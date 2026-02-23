"""
Countertop mask generation using TWO models:

1. Local Detectron2 Mask R-CNN  – instance segmentation (custom-trained)
2. Mask2Former ADE20K           – semantic segmentation (pre-trained)

The final mask is produced by combining both:
  final = (local_countertop  UNION  m2f_countertop)  MINUS  m2f_floor

This removes floor false-positives from the local model and fills in
areas under objects (which semantic segmentation covers but instance
segmentation misses).
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from .config import (
    MODEL_WEIGHTS,
    MODEL_CONFIG,
    NUM_CLASSES,
    CONFIDENCE,
    CLASS_NAMES,
    TARGET_CLASSES,
    M2F_MODEL_NAME,
    M2F_INCLUDE_IDS,
    M2F_FLOOR_ID,
    MASK_KERNEL_SIZE,
    MASK_CLOSE_ITER,
)


def build_predictor(
    weights: str | Path = MODEL_WEIGHTS,
    confidence: float = CONFIDENCE,
) -> DefaultPredictor:
    """Build and return a Detectron2 DefaultPredictor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = device

    # Register class metadata for visualization
    if "countertop_infer" not in MetadataCatalog:
        MetadataCatalog.get("countertop_infer").set(thing_classes=CLASS_NAMES)

    predictor = DefaultPredictor(cfg)
    print(f"Predictor ready  |  device={device}  |  weights={weights}")
    return predictor


def generate_mask(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    target_classes: list[int] | None = None,
) -> tuple[np.ndarray, object]:
    """
    Run inference and combine masks for the target classes.

    Parameters
    ----------
    image_bgr      : (H, W, 3) BGR image
    predictor      : DefaultPredictor
    target_classes : class indices to merge (default: config.TARGET_CLASSES)

    Returns
    -------
    mask      : (H, W) uint8, 0 or 255
    instances : detectron2 Instances (for visualization)
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pred_classes = instances.pred_classes.numpy()
    pred_masks = instances.pred_masks.numpy()       # (N, H, W) bool

    for i, cls_id in enumerate(pred_classes):
        if cls_id in target_classes:
            mask[pred_masks[i]] = 255

    return mask, instances


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = MASK_KERNEL_SIZE,
    close_iter: int = MASK_CLOSE_ITER,
) -> np.ndarray:
    """Morphological close to fill small holes and smooth edges."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)


# ─────────────────────────────────────────────────────────────────────
# MASK2FORMER ADE20K  (pre-trained semantic segmentation)
# ─────────────────────────────────────────────────────────────────────

def load_mask2former(model_name: str = M2F_MODEL_NAME):
    """Load the pre-trained Mask2Former ADE20K semantic model."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Mask2Former loaded  |  device={device}  |  model={model_name}")
    return processor, model, device


def m2f_segment(image_bgr: np.ndarray, processor, model, device) -> np.ndarray:
    """
    Run Mask2Former semantic segmentation.

    Returns
    -------
    seg_map : (H, W) int array of ADE20K class IDs
    """
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image_rgb.size[::-1]]
    )[0]
    return result.cpu().numpy()


def m2f_extract_mask(seg_map: np.ndarray, label_ids: list[int]) -> np.ndarray:
    """Extract binary mask (0/255) for given ADE20K label IDs."""
    mask = np.zeros_like(seg_map, dtype=np.uint8)
    for lid in label_ids:
        mask[seg_map == lid] = 255
    return mask


# ─────────────────────────────────────────────────────────────────────
# COMBINED MASK  (local model + Mask2Former)
# ─────────────────────────────────────────────────────────────────────

def generate_combined_mask(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    m2f_processor,
    m2f_model,
    m2f_device,
    target_classes: list[int] | None = None,
    m2f_include_ids: list[int] | None = None,
    m2f_floor_id: int = M2F_FLOOR_ID,
) -> dict:
    """
    Generate a refined countertop mask by combining two models.

    Strategy
    --------
    1. local_mask  = Detectron2 countertop predictions
    2. m2f_mask    = Mask2Former countertop/cabinet/island semantic mask
    3. m2f_floor   = Mask2Former floor semantic mask
    4. combined    = (local_mask  &  m2f_mask)  &  ~m2f_floor

    Only pixels that BOTH models agree on are kept (AND gate).
    Floor pixels are then subtracted as a safety net.

    Returns
    -------
    dict with keys:
        'local_mask', 'm2f_mask', 'm2f_floor', 'combined',
        'instances', 'seg_map'
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES
    if m2f_include_ids is None:
        m2f_include_ids = M2F_INCLUDE_IDS

    # 1. Local Detectron2 model
    local_mask, instances = generate_mask(image_bgr, predictor, target_classes)
    local_mask = clean_mask(local_mask)

    # 2. Mask2Former ADE20K
    seg_map = m2f_segment(image_bgr, m2f_processor, m2f_model, m2f_device)
    m2f_mask = m2f_extract_mask(seg_map, m2f_include_ids)
    m2f_mask = clean_mask(m2f_mask)
    m2f_floor = m2f_extract_mask(seg_map, [m2f_floor_id])

    # 3. Combine: INTERSECTION of both masks (AND), then subtract floor
    combined = np.minimum(local_mask, m2f_mask)   # AND gate
    combined[m2f_floor > 0] = 0                    # remove floor false-positives
    combined = clean_mask(combined)                 # clean up edges

    return {
        "local_mask": local_mask,
        "m2f_mask":   m2f_mask,
        "m2f_floor":  m2f_floor,
        "combined":   combined,
        "instances":  instances,
        "seg_map":    seg_map,
    }


def save_preview(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    instances,
    output_path: Path,
) -> None:
    """Save a 4-panel preview: original | predictions | mask | overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # 1 — Original
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 2 — All predictions
    v = Visualizer(
        image_rgb,
        MetadataCatalog.get("countertop_infer"),
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW,
    )
    vis_out = v.draw_instance_predictions(instances)
    axes[1].imshow(vis_out.get_image())
    axes[1].set_title(f"Predictions ({len(instances)})")
    axes[1].axis("off")

    # 3 — Binary mask
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Mask ({np.count_nonzero(mask)} px)")
    axes[2].axis("off")

    # 4 — Overlay
    overlay = image_rgb.copy()
    overlay[mask > 0] = [255, 50, 50]
    blended = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)
    axes[3].imshow(blended)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.suptitle(output_path.stem, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Preview saved: {output_path}")


def save_combined_preview(
    image_bgr: np.ndarray,
    result: dict,
    output_path: Path,
) -> None:
    """
    Save a 6-panel comparison preview:
    Row 1: Original | Local Mask | Mask2Former Mask
    Row 2: Floor Mask (excluded) | Combined Final | Final Overlay
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    local_mask = result["local_mask"]
    m2f_mask   = result["m2f_mask"]
    m2f_floor  = result["m2f_floor"]
    combined   = result["combined"]

    fig, axes = plt.subplots(2, 3, figsize=(24, 10))

    # Row 1
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(local_mask, cmap="gray")
    axes[0, 1].set_title(f"Local Model ({np.count_nonzero(local_mask)} px)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(m2f_mask, cmap="gray")
    axes[0, 2].set_title(f"Mask2Former ({np.count_nonzero(m2f_mask)} px)")
    axes[0, 2].axis("off")

    # Row 2
    axes[1, 0].imshow(m2f_floor, cmap="Reds")
    axes[1, 0].set_title(f"Floor (excluded, {np.count_nonzero(m2f_floor)} px)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(combined, cmap="gray")
    axes[1, 1].set_title(f"Combined Final ({np.count_nonzero(combined)} px)")
    axes[1, 1].axis("off")

    overlay = image_rgb.copy()
    overlay[combined > 0] = [255, 50, 50]
    blended = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title("Final Overlay")
    axes[1, 2].axis("off")

    plt.suptitle(output_path.stem, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Combined preview saved: {output_path}")
