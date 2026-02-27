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


def is_valid_countertop_mask(
    mask_bool: np.ndarray,
    box: np.ndarray,
    image_shape: tuple,
    min_area_ratio: float = 0.002,    # Lowered from 0.005 to catch smaller countertops
    max_area_ratio: float = 0.65,     # Increased from 0.5 to allow larger surfaces
    min_aspect_ratio: float = 1.2,    # Relaxed from 1.5 for more flexible shapes
) -> tuple[bool, str]:
    """
    Enhanced filter to identify countertop objects while excluding sinks, cabinet doors, and walls.

    Criteria (Enhanced)
    -------------------
    - Area: Must be significant but not too large (not walls, not tiny objects)
    - Aspect ratio: Countertops are typically wide/horizontal (relaxed constraints)
    - Position: Should be in middle portions of image (expanded range)
    - Shape: Should be relatively horizontal, not vertical
    - Compactness: Should be relatively solid (relaxed for irregular countertops)
    - Size consistency: Reasonable bounding box dimensions

    Returns
    -------
    (is_valid, reason) : bool and string explaining why rejected
    """
    h, w = image_shape[:2]
    total_pixels = h * w
    mask_area = np.count_nonzero(mask_bool)
    area_ratio = mask_area / total_pixels

    # Filter 1: Area constraints (relaxed)
    if area_ratio < min_area_ratio:
        return False, f"too_small ({area_ratio*100:.2f}%)"
    if area_ratio > max_area_ratio:
        return False, f"too_large ({area_ratio*100:.2f}%)"

    # Filter 2: Bounding box aspect ratio (more flexible)
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    if box_height > 0 and box_width > 0:
        aspect_ratio = box_width / box_height
        
        # Countertops should be wider than tall, but allow some flexibility
        if aspect_ratio < min_aspect_ratio:
            return False, f"not_horizontal (aspect={aspect_ratio:.2f})"
        
        # Filter out extremely wide thin objects (likely edges/artifacts)
        if aspect_ratio > 15.0:
            return False, f"too_wide (aspect={aspect_ratio:.2f})"
        
        # Check minimum dimensions (avoid tiny detections)
        if box_width < 50 or box_height < 20:  # pixels
            return False, f"too_small_box (w={box_width:.0f}, h={box_height:.0f})"

    # Filter 3: Vertical position check (expanded range)
    # Cabinet doors and walls are often at edges/top
    box_center_y = (y1 + y2) / 2
    relative_y = box_center_y / h
    
    # Countertops typically in middle to upper-middle of frame
    # Expanded range: allow 10% to 90% (was 15% to 85%)
    if relative_y < 0.10:
        return False, f"too_high (y={relative_y:.2f})"
    if relative_y > 0.90:
        return False, f"too_low (y={relative_y:.2f})"

    # Filter 4: Check mask compactness (relaxed for irregular countertops)
    # Calculate bounding box fill ratio
    bbox_area = box_width * box_height
    if bbox_area > 0:
        fill_ratio = mask_area / bbox_area
        
        # Relaxed from 0.3 to 0.2 to allow more irregular shapes
        if fill_ratio < 0.2:
            return False, f"not_solid (fill={fill_ratio:.2f})"
        
        # But reject if it's TOO uniform (might be wall/floor)
        if fill_ratio > 0.98 and area_ratio > 0.4:
            return False, f"too_uniform (fill={fill_ratio:.2f}, likely wall)"

    # Filter 5: Check horizontal extent (countertops should span reasonable width)
    relative_width = box_width / w
    if relative_width < 0.1:  # Must be at least 10% of image width
        return False, f"too_narrow (width={relative_width*100:.1f}%)"

    return True, "valid"


def generate_sam_mask(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    sam_predictor,
    target_classes: list[int] | None = None,
    filter_non_countertops: bool = True,
) -> dict:
    """
    Generate precise masks using Detectron2 boxes → SAM refinement.

    Flow
    ----
    1. Run Detectron2 → bounding boxes + class predictions
    2. Feed each box to SAM for pixel-precise mask
    3. Filter out non-countertop objects (sinks, walls, cabinet doors)
    4. Merge masks belonging to target classes

    Parameters
    ----------
    image_bgr              : (H, W, 3) BGR image
    predictor              : Detectron2 DefaultPredictor
    sam_predictor          : SAM SamPredictor
    target_classes         : class indices to include (default: config.TARGET_CLASSES)
    filter_non_countertops : if True, apply strict filtering for countertops only

    Returns
    -------
    dict with keys:
        'sam_mask'       : (H, W) uint8, merged SAM mask for target classes (0/255)
        'all_masks'      : list of (H, W) bool masks from SAM (one per detection)
        'all_classes'    : int array of class IDs per detection
        'boxes'          : (N, 4) float array of bounding boxes
        'scores'         : (N,) float array of confidence scores
        'instances'      : Detectron2 Instances (for visualization)
        'filtered_count' : number of masks filtered out
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
    filtered_count = 0
    filter_reasons = []

    for i, box in enumerate(boxes):
        input_box = np.array(box)
        masks_sam, _scores_sam, _logits = sam_predictor.predict(
            box=input_box[None, :],
            multimask_output=False,
        )
        mask_bool = masks_sam[0]  # (H, W) bool
        all_masks.append(mask_bool)
        all_classes.append(classes[i])

        # Merge into target mask with filtering
        if classes[i] in target_classes:
            # Apply countertop-specific filtering
            if filter_non_countertops:
                is_valid, reason = is_valid_countertop_mask(
                    mask_bool, box, image_bgr.shape
                )
                if is_valid:
                    merged_mask[mask_bool] = 255
                else:
                    filtered_count += 1
                    filter_reasons.append(f"Instance {i}: {reason}")
            else:
                # No filtering, include all target class masks
                merged_mask[mask_bool] = 255

    merged_mask = clean_mask(merged_mask)

    # Print filtering summary if any were filtered
    if filtered_count > 0:
        print(f"  Filtered {filtered_count} non-countertop object(s):")
        for reason in filter_reasons[:5]:  # Show first 5
            print(f"    - {reason}")
        if len(filter_reasons) > 5:
            print(f"    ... and {len(filter_reasons) - 5} more")

    return {
        "sam_mask":       merged_mask,
        "all_masks":      all_masks,
        "all_classes":    np.array(all_classes),
        "boxes":          boxes,
        "scores":         scores,
        "instances":      instances,
        "filtered_count": filtered_count,
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


def visualize_sam_countertop_labels(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    sam_predictor,
    output_path: Path,
    target_classes: list[int] | None = None,
    show_overlay: bool = True,
) -> dict:
    """
    Visualize SAM countertop instances with numbered labels.

    Layout (3 panels):
    - Original Image
    - Labeled Countertop Instances (each with unique color and number)
    - Overlay on Original (optional)

    Parameters
    ----------
    image_bgr      : (H, W, 3) BGR image
    predictor      : Detectron2 DefaultPredictor (MaskRCNN)
    sam_predictor  : SAM SamPredictor
    output_path    : where to save the visualization
    target_classes : class indices to include (default: config.TARGET_CLASSES)
    show_overlay   : if True, show 3-panel layout; if False, 2-panel

    Returns
    -------
    dict with SAM results
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if target_classes is None:
        target_classes = TARGET_CLASSES

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Generate SAM masks with filtering
    sam_result = generate_sam_mask(image_bgr, predictor, sam_predictor, target_classes, filter_non_countertops=True)
    
    all_masks = sam_result["all_masks"]
    all_classes = sam_result["all_classes"]
    sam_boxes = sam_result["boxes"]
    sam_mask = sam_result["sam_mask"]

    # Filter to only target classes (countertops)
    countertop_masks = []
    countertop_boxes = []
    countertop_indices = []
    
    for i, (mask_bool, cls) in enumerate(zip(all_masks, all_classes)):
        if cls in target_classes:
            countertop_masks.append(mask_bool)
            countertop_boxes.append(sam_boxes[i])
            countertop_indices.append(i)

    # Color palette
    color_palette = [
        [255, 100, 100],   # Light red
        [100, 255, 100],   # Light green
        [100, 100, 255],   # Light blue
        [255, 255, 100],   # Yellow
        [255, 100, 255],   # Magenta
        [100, 255, 255],   # Cyan
        [255, 150, 100],   # Orange
        [150, 100, 255],   # Purple
        [100, 255, 150],   # Mint
    ]

    # Create figure
    num_panels = 3 if show_overlay else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(8*num_panels, 6))
    if num_panels == 1:
        axes = [axes]

    # ─────────────────────────────────────────────────────────────────
    # Panel 1: Original Image
    # ─────────────────────────────────────────────────────────────────
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # Panel 2: Labeled Countertop Instances
    # ─────────────────────────────────────────────────────────────────
    labeled_image = np.zeros_like(image_rgb)
    
    for idx, (mask_bool, box) in enumerate(zip(countertop_masks, countertop_boxes)):
        # Assign color
        color = color_palette[idx % len(color_palette)]
        labeled_image[mask_bool] = color
        
        # Draw contours
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        labeled_image_bgr = cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
        cv2.drawContours(labeled_image_bgr, contours, -1, (255, 255, 255), 3)
        labeled_image = cv2.cvtColor(labeled_image_bgr, cv2.COLOR_BGR2RGB)
        
        # Add label number at center
        M = cv2.moments(mask_bool.astype(np.uint8))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback to box center
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
        
        label_text = f"#{idx + 1}"
        
        # Draw label with background
        labeled_image_bgr = cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
        
        # Get text size for background
        font_scale = 1.5
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw black background rectangle
        padding = 10
        cv2.rectangle(
            labeled_image_bgr,
            (cx - text_width//2 - padding, cy - text_height//2 - padding),
            (cx + text_width//2 + padding, cy + text_height//2 + padding + baseline),
            (0, 0, 0),
            -1
        )
        
        # Draw white text
        cv2.putText(
            labeled_image_bgr,
            label_text,
            (cx - text_width//2, cy + text_height//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        labeled_image = cv2.cvtColor(labeled_image_bgr, cv2.COLOR_BGR2RGB)
    
    axes[1].imshow(labeled_image)
    axes[1].set_title(f"SAM Countertop Labels ({len(countertop_masks)} instances)", 
                      fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # Panel 3: Overlay on Original (optional)
    # ─────────────────────────────────────────────────────────────────
    if show_overlay:
        overlay_image = image_rgb.copy()
        
        for idx, (mask_bool, box) in enumerate(zip(countertop_masks, countertop_boxes)):
            # Semi-transparent color overlay
            color = color_palette[idx % len(color_palette)]
            colored = np.zeros_like(image_rgb, dtype=np.uint8)
            colored[mask_bool] = color
            overlay_image = cv2.addWeighted(overlay_image, 1, colored, 0.5, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
            cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 3)
            overlay_image = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            
            # Add label
            M = cv2.moments(mask_bool.astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
            
            label_text = f"Countertop #{idx + 1}"
            
            overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
            
            # Draw label with shadow effect
            cv2.putText(overlay_bgr, label_text, (cx - 80, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(overlay_bgr, label_text, (cx - 80, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            overlay_image = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        
        axes[2].imshow(overlay_image)
        axes[2].set_title(f"Overlay ({np.count_nonzero(sam_mask)} pixels)", 
                          fontsize=14, fontweight="bold")
        axes[2].axis("off")

    # Add summary text
    summary = f"Detected {len(countertop_masks)} countertop instance(s)"
    fig.text(0.5, 0.02, summary, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.suptitle(f"SAM Countertop Labels - {output_path.stem}", 
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  SAM countertop labels saved: {output_path}")
    print(f"  Found {len(countertop_masks)} countertop instance(s)")
    
    return sam_result


def visualize_maskrcnn_and_sam(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    sam_predictor,
    output_path: Path,
    target_classes: list[int] | None = None,
) -> dict:
    """
    Comprehensive visualization comparing MaskRCNN and SAM results.

    Layout (2 rows × 3 columns):
    Row 1: Original | MaskRCNN Detections | MaskRCNN Masks
    Row 2: SAM Labeled Masks | SAM Colored Instances | Final Merged Overlay

    Parameters
    ----------
    image_bgr      : (H, W, 3) BGR image
    predictor      : Detectron2 DefaultPredictor (MaskRCNN)
    sam_predictor  : SAM SamPredictor
    output_path    : where to save the visualization
    target_classes : class indices to include (default: config.TARGET_CLASSES)

    Returns
    -------
    dict with 'maskrcnn_result' and 'sam_result'
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if target_classes is None:
        target_classes = TARGET_CLASSES

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Get MaskRCNN predictions
    # ═══════════════════════════════════════════════════════════════════
    mask_rcnn, instances = generate_mask(image_bgr, predictor, target_classes)
    mask_rcnn_clean = clean_mask(mask_rcnn)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Get SAM refined masks
    # ═══════════════════════════════════════════════════════════════════
    sam_result = generate_sam_mask(image_bgr, predictor, sam_predictor, target_classes, filter_non_countertops=True)

    # ═══════════════════════════════════════════════════════════════════
    # Create visualization
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # ─────────────────────────────────────────────────────────────────
    # ROW 1: MaskRCNN Results
    # ─────────────────────────────────────────────────────────────────

    # [0,0] Original Image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # [0,1] MaskRCNN Detections with bounding boxes
    axes[0, 1].imshow(image_rgb)
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color = color_map.get(int(cls), 'cyan')
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
        axes[0, 1].add_patch(rect)
        
        # Label
        cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls_{cls}"
        label = f"{cls_name} {score:.2f}"
        axes[0, 1].text(x1, y1 - 5, label, color='white', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    axes[0, 1].set_title(f"MaskRCNN Detections ({len(instances)} objects)", 
                         fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # [0,2] MaskRCNN Masks (binary)
    axes[0, 2].imshow(mask_rcnn_clean, cmap="gray")
    axes[0, 2].set_title(f"MaskRCNN Masks ({np.count_nonzero(mask_rcnn_clean)} px)", 
                         fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # ROW 2: SAM Results
    # ─────────────────────────────────────────────────────────────────
    
    sam_mask = sam_result["sam_mask"]
    all_masks = sam_result["all_masks"]
    all_classes = sam_result["all_classes"]
    sam_boxes = sam_result["boxes"]

    # [1,0] SAM Labeled Masks (each instance in different color)
    color_palette = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 255, 0], [255, 128, 0], [128, 0, 255],
    ]
    labeled_sam = np.zeros_like(image_rgb)
    for i, mask_bool in enumerate(all_masks):
        color = color_palette[i % len(color_palette)]
        labeled_sam[mask_bool] = color
    
    axes[1, 0].imshow(labeled_sam)
    axes[1, 0].set_title(f"SAM Labeled Masks ({len(all_masks)} instances)", 
                         fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # [1,1] SAM Colored Instances with contours and labels
    sam_overlay = image_rgb.copy()
    for i, mask_bool in enumerate(all_masks):
        color = tuple(color_palette[i % len(color_palette)])
        colored = np.zeros_like(image_rgb, dtype=np.uint8)
        colored[mask_bool] = color
        sam_overlay = cv2.addWeighted(sam_overlay, 1, colored, 0.5, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(sam_overlay, contours, -1, (255, 255, 255), 2)
        
        # Add instance labels
        if i < len(sam_boxes):
            x1, y1 = sam_boxes[i][:2].astype(int)
            cls_name = CLASS_NAMES[all_classes[i]] if all_classes[i] < len(CLASS_NAMES) else f"cls_{all_classes[i]}"
            label = f"#{i+1}: {cls_name}"
            cv2.putText(sam_overlay, label, (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(sam_overlay, label, (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    axes[1, 1].imshow(sam_overlay)
    axes[1, 1].set_title(f"SAM Colored Instances", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # [1,2] Final Merged Overlay
    final_overlay = image_rgb.copy()
    final_overlay[sam_mask > 0] = [255, 50, 50]
    blended = cv2.addWeighted(image_rgb, 0.6, final_overlay, 0.4, 0)
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title(f"Final Merged Mask ({np.count_nonzero(sam_mask)} px)", 
                         fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    # ═══════════════════════════════════════════════════════════════════
    # Add comparison statistics
    # ═══════════════════════════════════════════════════════════════════
    stats_text = (
        f"MaskRCNN: {len(instances)} detections, {np.count_nonzero(mask_rcnn_clean)} pixels\n"
        f"SAM: {len(all_masks)} refined masks, {np.count_nonzero(sam_mask)} pixels\n"
        f"Target classes: {target_classes}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f"MaskRCNN vs SAM Comparison - {output_path.stem}", 
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Comprehensive visualization saved: {output_path}")
    
    return {
        "maskrcnn_result": {
            "mask": mask_rcnn_clean,
            "instances": instances,
        },
        "sam_result": sam_result,
    }


def visualize_three_model_masks(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    sam_predictor,
    mask2former_processor,
    mask2former_model,
    mask2former_device,
    output_path: Path,
    target_classes: list[int] | None = None,
    surface_names: list[str] = None,
    filter_sam: bool = True,
) -> dict:
    """
    Compare black & white masks from 3 models side by side:
    1. MaskRCNN (custom trained)
    2. SAM (refined from MaskRCNN boxes)
    3. Mask2Former (semantic segmentation)

    Layout (2 rows × 2 columns):
    Row 1: Original Image | MaskRCNN Mask (B&W)
    Row 2: SAM Mask (B&W) | Mask2Former Mask (B&W)

    Parameters
    ----------
    image_bgr              : (H, W, 3) BGR image
    predictor              : Detectron2 DefaultPredictor (MaskRCNN)
    sam_predictor          : SAM SamPredictor
    mask2former_processor  : Mask2Former processor
    mask2former_model      : Mask2Former model
    mask2former_device     : torch device
    output_path            : where to save the visualization
    target_classes         : class indices for MaskRCNN (default: config.TARGET_CLASSES)
    surface_names          : surfaces for Mask2Former (default: countertop, table, cabinet)
    filter_sam             : if True, filter out non-countertop SAM masks (default: True)

    Returns
    -------
    dict with all three model results
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from surfaces import extract_surface_mask, clean_mask as clean_m2f_mask

    if target_classes is None:
        target_classes = TARGET_CLASSES
    
    if surface_names is None:
        surface_names = ["countertop", "table", "cabinet"]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: MaskRCNN predictions
    # ═══════════════════════════════════════════════════════════════════
    print("  [1/3] Running MaskRCNN...")
    mask_rcnn, instances = generate_mask(image_bgr, predictor, target_classes)
    mask_rcnn_clean = clean_mask(mask_rcnn)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: SAM refined masks
    # ═══════════════════════════════════════════════════════════════════
    print("  [2/3] Running SAM refinement...")
    sam_result = generate_sam_mask(image_bgr, predictor, sam_predictor, target_classes, filter_non_countertops=filter_sam)
    sam_mask = sam_result["sam_mask"]

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Mask2Former semantic segmentation
    # ═══════════════════════════════════════════════════════════════════
    print("  [3/3] Running Mask2Former...")
    pil_image = Image.fromarray(image_rgb)
    inputs = mask2former_processor(images=pil_image, return_tensors="pt").to(mask2former_device)
    
    with torch.no_grad():
        outputs = mask2former_model(**inputs)
    
    segmentation_map = mask2former_processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[pil_image.size[::-1]],
    )[0].cpu().numpy()

    # Combine masks for specified surfaces
    mask2former_combined = np.zeros_like(segmentation_map, dtype=np.uint8)
    surfaces_found = []
    
    for surface_name in surface_names:
        try:
            surface_mask = extract_surface_mask(segmentation_map, surface_name)
            mask2former_combined = np.maximum(mask2former_combined, surface_mask)
            if np.count_nonzero(surface_mask) > 0:
                surfaces_found.append(surface_name)
        except ValueError:
            continue
    
    mask2former_clean = clean_m2f_mask(mask2former_combined)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Combine SAM and Mask2Former with AND gate
    # ═══════════════════════════════════════════════════════════════════
    print("  [4/4] Combining SAM + Mask2Former (AND gate)...")
    # AND operation: only regions where BOTH models detected countertops
    combined_and_mask = cv2.bitwise_and(sam_mask, mask2former_clean)
    combined_pixels = np.count_nonzero(combined_and_mask)

    # ═══════════════════════════════════════════════════════════════════
    # Create visualization (2 rows × 3 columns)
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # ─────────────────────────────────────────────────────────────────
    # [0,0] Original Image
    # ─────────────────────────────────────────────────────────────────
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # [0,1] MaskRCNN Mask (B&W)
    # ─────────────────────────────────────────────────────────────────
    axes[0, 1].imshow(mask_rcnn_clean, cmap="gray", vmin=0, vmax=255)
    rcnn_pixels = np.count_nonzero(mask_rcnn_clean)
    rcnn_pct = (rcnn_pixels / mask_rcnn_clean.size) * 100
    axes[0, 1].set_title(
        f"MaskRCNN Mask\n{len(instances)} detections | {rcnn_pixels:,} px ({rcnn_pct:.1f}%)", 
        fontsize=12, fontweight="bold"
    )
    axes[0, 1].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # [0,2] SAM Mask (B&W)
    # ─────────────────────────────────────────────────────────────────
    axes[0, 2].imshow(sam_mask, cmap="gray", vmin=0, vmax=255)
    sam_pixels = np.count_nonzero(sam_mask)
    sam_pct = (sam_pixels / sam_mask.size) * 100
    axes[0, 2].set_title(
        f"SAM Refined Mask\n{len(sam_result['all_masks'])} instances | {sam_pixels:,} px ({sam_pct:.1f}%)", 
        fontsize=12, fontweight="bold"
    )
    axes[0, 2].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # [1,0] Mask2Former Mask (B&W)
    # ─────────────────────────────────────────────────────────────────
    axes[1, 0].imshow(mask2former_clean, cmap="gray", vmin=0, vmax=255)
    m2f_pixels = np.count_nonzero(mask2former_clean)
    m2f_pct = (m2f_pixels / mask2former_clean.size) * 100
    surfaces_str = ", ".join(surfaces_found) if surfaces_found else "none"
    axes[1, 0].set_title(
        f"Mask2Former Semantic Mask\n{surfaces_str} | {m2f_pixels:,} px ({m2f_pct:.1f}%)", 
        fontsize=12, fontweight="bold"
    )
    axes[1, 0].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # [1,1] Combined AND Mask (B&W)
    # ─────────────────────────────────────────────────────────────────
    axes[1, 1].imshow(combined_and_mask, cmap="gray", vmin=0, vmax=255)
    combined_pct = (combined_pixels / combined_and_mask.size) * 100
    axes[1, 1].set_title(
        f"Combined (SAM ∩ Mask2Former)\n{combined_pixels:,} px ({combined_pct:.1f}%)", 
        fontsize=12, fontweight="bold", color='darkgreen'
    )
    axes[1, 1].axis("off")

    # ─────────────────────────────────────────────────────────────────
    # [1,2] Overlay comparison on original
    # ─────────────────────────────────────────────────────────────────
    overlay_comparison = image_rgb.copy()
    # SAM only (yellow)
    sam_only = cv2.bitwise_and(sam_mask, cv2.bitwise_not(mask2former_clean))
    overlay_comparison[sam_only > 0] = [255, 255, 0]
    # Mask2Former only (blue)
    m2f_only = cv2.bitwise_and(mask2former_clean, cv2.bitwise_not(sam_mask))
    overlay_comparison[m2f_only > 0] = [0, 150, 255]
    # Both (green) 
    overlay_comparison[combined_and_mask > 0] = [0, 255, 0]
    blended = cv2.addWeighted(image_rgb, 0.5, overlay_comparison, 0.5, 0)
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title(
        f"Overlay Comparison\nGreen=Both | Yellow=SAM only | Blue=Mask2Former only", 
        fontsize=11, fontweight="bold"
    )
    axes[1, 2].axis("off")

    # ═══════════════════════════════════════════════════════════════════
    # Add comparison summary
    # ═══════════════════════════════════════════════════════════════════
    stats_text = (
        f"Model Comparison Summary:\n"
        f"1. MaskRCNN (custom): {len(instances)} objects, {rcnn_pixels:,} pixels ({rcnn_pct:.1f}%)\n"
        f"2. SAM (refined): {len(sam_result['all_masks'])} instances, {sam_pixels:,} pixels ({sam_pct:.1f}%)\n"
        f"3. Mask2Former (ADE20K): {surfaces_str}, {m2f_pixels:,} pixels ({m2f_pct:.1f}%)\n"
        f"4. Combined (AND): {combined_pixels:,} pixels ({combined_pct:.1f}%) ← Final Mask"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.9))

    plt.suptitle(f"3-Model Mask Comparison with AND Gate - {output_path.stem}", 
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 3-Model comparison saved: {output_path}")
    print(f"  ✅ Combined AND mask: {combined_pixels:,} pixels ({combined_pct:.1f}%)")
    
    # Also save the combined mask separately
    combined_mask_path = output_path.parent / f"{output_path.stem}_combined_mask.png"
    cv2.imwrite(str(combined_mask_path), combined_and_mask)
    print(f"  ✅ Combined mask saved: {combined_mask_path}")
    
    return {
        "maskrcnn_result": {
            "mask": mask_rcnn_clean,
            "instances": instances,
            "pixels": rcnn_pixels,
        },
        "sam_result": {
            "mask": sam_mask,
            "all_masks": sam_result["all_masks"],
            "pixels": sam_pixels,
        },
        "mask2former_result": {
            "mask": mask2former_clean,
            "surfaces": surfaces_found,
            "pixels": m2f_pixels,
        },
        "combined_result": {
            "mask": combined_and_mask,
            "pixels": combined_pixels,
            "percentage": combined_pct,
        },
    }
