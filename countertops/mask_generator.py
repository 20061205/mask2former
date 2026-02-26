"""
Countertop mask generation using TWO models:

1. Local Detectron2 Mask R-CNN  – instance segmentation (custom-trained)
2. SAM (Segment Anything)       – refines Detectron2 boxes into precise masks

Pipeline:  Detectron2 → bounding boxes → SAM → pixel-precise masks
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from .config import (
    MODEL_WEIGHTS,
    MODEL_CONFIG,
    NUM_CLASSES,
    CONFIDENCE,
    CLASS_NAMES,
    TARGET_CLASSES,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
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
# SAM  (Segment Anything Model – box-prompted refinement)
# ─────────────────────────────────────────────────────────────────────

def load_sam(
    checkpoint: str | Path = SAM_CHECKPOINT,
    model_type: str = SAM_MODEL_TYPE,
):
    """
    Load the SAM model and return a SamPredictor.

    Parameters
    ----------
    checkpoint : path to the SAM weights (e.g. sam_vit_h_4b8939.pth)
    model_type : model variant ('vit_h', 'vit_l', 'vit_b')

    Returns
    -------
    SamPredictor instance
    """
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"SAM loaded  |  device={device}  |  type={model_type}  |  weights={checkpoint}")
    return predictor


def generate_sam_mask(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    sam_predictor,
    target_classes: list[int] | None = None,
) -> dict:
    """
    Generate precise masks using Detectron2 boxes → SAM refinement.

    Flow
    ----
    1. Run Detectron2 → bounding boxes + class predictions
    2. Feed each box to SAM for pixel-precise mask
    3. Merge masks belonging to target classes

    Parameters
    ----------
    image_bgr      : (H, W, 3) BGR image
    predictor       : Detectron2 DefaultPredictor
    sam_predictor   : SAM SamPredictor
    target_classes  : class indices to include (default: config.TARGET_CLASSES)

    Returns
    -------
    dict with keys:
        'sam_mask'     : (H, W) uint8, merged SAM mask for target classes (0/255)
        'all_masks'    : list of (H, W) bool masks from SAM (one per detection)
        'all_classes'  : int array of class IDs per detection
        'boxes'        : (N, 4) float array of bounding boxes
        'scores'       : (N,) float array of confidence scores
        'instances'    : Detectron2 Instances (for visualization)
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    # 1. Detectron2 inference → boxes + classes
    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    boxes   = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores  = instances.scores.numpy()

    h, w = image_bgr.shape[:2]

    # 2. SAM refinement: feed each box
    sam_predictor.set_image(image_bgr)

    all_masks = []
    all_classes = []
    merged_mask = np.zeros((h, w), dtype=np.uint8)

    for i, box in enumerate(boxes):
        input_box = np.array(box)
        masks_sam, _scores_sam, _logits = sam_predictor.predict(
            box=input_box[None, :],
            multimask_output=False,
        )
        mask_bool = masks_sam[0]  # (H, W) bool
        all_masks.append(mask_bool)
        all_classes.append(classes[i])

        # Merge into target mask
        if classes[i] in target_classes:
            merged_mask[mask_bool] = 255

    merged_mask = clean_mask(merged_mask)

    return {
        "sam_mask":    merged_mask,
        "all_masks":   all_masks,
        "all_classes":  np.array(all_classes),
        "boxes":        boxes,
        "scores":       scores,
        "instances":    instances,
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


def save_sam_preview(
    image_bgr: np.ndarray,
    result: dict,
    output_path: Path,
) -> None:
    """
    Save a SAM-specific preview with colored instance overlays.

    Layout: Original | SAM Mask | SAM Overlay (colored per instance) | Final Overlay
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_mask   = result["sam_mask"]
    all_masks  = result["all_masks"]
    all_classes = result["all_classes"]
    boxes      = result["boxes"]

    fig, axes = plt.subplots(1, 4, figsize=(32, 6))

    # 1 — Original
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 2 — SAM mask (binary, target classes only)
    axes[1].imshow(sam_mask, cmap="gray")
    axes[1].set_title(f"SAM Mask ({np.count_nonzero(sam_mask)} px)")
    axes[1].axis("off")

    # 3 — Colored overlay (all instances) with class labels
    color_palette = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    overlay = image_rgb.copy()
    for i, mask_bool in enumerate(all_masks):
        color = color_palette[i % len(color_palette)]
        colored = np.zeros_like(image_rgb, dtype=np.uint8)
        colored[mask_bool] = color
        overlay = cv2.addWeighted(overlay, 1, colored, 0.45, 0)

        # Contour
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        # Class label
        if i < len(boxes):
            x1, y1 = boxes[i][:2].astype(int)
            cls_name = CLASS_NAMES[all_classes[i]] if all_classes[i] < len(CLASS_NAMES) else f"cls_{all_classes[i]}"
            cv2.putText(overlay, cls_name, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    axes[2].imshow(overlay)
    axes[2].set_title(f"SAM Instances ({len(all_masks)})")
    axes[2].axis("off")

    # 4 — Final overlay
    final_overlay = image_rgb.copy()
    final_overlay[sam_mask > 0] = [255, 50, 50]
    blended = cv2.addWeighted(image_rgb, 0.6, final_overlay, 0.4, 0)
    axes[3].imshow(blended)
    axes[3].set_title(f"Final Overlay ({np.count_nonzero(sam_mask)} px)")
    axes[3].axis("off")

    plt.suptitle(output_path.stem, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  SAM preview saved: {output_path}")
