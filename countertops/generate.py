"""
Generate countertop masks using Detectron2 Mask R-CNN + SAM.

Usage
-----
  cd "E:\\tile viz\\mask2former"

  # SAM mode (default): Detectron2 boxes → SAM precise masks
  python -m countertops.generate --preview
  python -m countertops.generate --image rooms/kitchens/kit1.jpg --preview

  # Local model only (faster, no SAM)
  python -m countertops.generate --mode local --preview
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import INPUT_DIR, OUTPUT_DIR, CONFIDENCE, CLASS_NAMES, TARGET_CLASSES
from .mask_generator import (
    build_predictor,
    generate_mask,
    clean_mask,
    save_preview,
    load_sam,
    generate_sam_mask,
    save_sam_preview,
)


def process_image(img_path: Path, predictor, output_dir: Path, preview: bool,
                   sam_pred=None, mode="sam"):
    """
    Run inference on one image and save the mask.

    Parameters
    ----------
    sam_pred : SamPredictor or None
    mode     : 'sam' | 'local'
    """
    print(f"\nProcessing: {img_path.name}  (mode={mode})")

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  ERROR: could not read {img_path}")
        return

    sam_result = None

    if mode == "sam" and sam_pred is not None:
        # ── SAM mode: Detectron2 boxes → SAM masks ──
        sam_result = generate_sam_mask(image_bgr, predictor, sam_pred)
        final_mask = sam_result["sam_mask"]
        instances = sam_result["instances"]

        print(f"  SAM mask: {np.count_nonzero(final_mask)} px  "
              f"({len(sam_result['all_masks'])} instances)")

    else:
        # ── Local-only mode ──
        mask, instances = generate_mask(image_bgr, predictor)
        final_mask = clean_mask(mask)

    # Print detected instances
    pred_classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    for i, (cls, sc) in enumerate(zip(pred_classes, scores)):
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
        tag = " [included]" if cls in TARGET_CLASSES else ""
        print(f"  Instance {i}: {label} ({sc:.3f}){tag}")

    # Save binary mask
    mask_path = output_dir / f"{img_path.stem}_countertop_mask.png"
    cv2.imwrite(str(mask_path), final_mask)
    print(f"  Mask saved: {mask_path}  ({np.count_nonzero(final_mask)} white px)")

    # Optional preview
    if preview:
        if sam_result is not None:
            preview_path = output_dir / f"{img_path.stem}_sam_preview.png"
            save_sam_preview(image_bgr, sam_result, preview_path)
        else:
            preview_path = output_dir / f"{img_path.stem}_preview.png"
            save_preview(image_bgr, final_mask, instances, preview_path)


def main():
    parser = argparse.ArgumentParser(description="Generate countertop masks")
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help="Folder of kitchen images (default: rooms/kitchens)")
    parser.add_argument("--image", type=Path, default=None,
                        help="Single image to process (overrides --input)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Where to save masks (default: countertops/masks)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE,
                        help="Detection confidence threshold")
    parser.add_argument("--preview", action="store_true",
                        help="Save preview images alongside masks")
    parser.add_argument("--mode", choices=["sam", "local"],
                        default="sam",
                        help="sam=Detectron2+SAM (default), local=Detectron2 only")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load models based on mode
    predictor = build_predictor(confidence=args.confidence)
    sam_pred = None

    if args.mode == "sam":
        sam_pred = load_sam()

    print(f"\nMode: {args.mode}")

    if args.image:
        process_image(args.image, predictor, args.output, args.preview,
                      sam_pred=sam_pred, mode=args.mode)
    else:
        exts = ("*.jpg", "*.jpeg", "*.png")
        image_paths = []
        for ext in exts:
            image_paths.extend(sorted(args.input.glob(ext)))
        print(f"Found {len(image_paths)} images in {args.input}")

        for img_path in image_paths:
            process_image(img_path, predictor, args.output, args.preview,
                          sam_pred=sam_pred, mode=args.mode)

    print(f"\nDone! Masks saved to: {args.output}")


if __name__ == "__main__":
    main()
