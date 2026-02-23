"""
Generate countertop masks using TWO models for best accuracy.

Usage
-----
  cd "E:\\tile viz\\mask2former"

  # Combined mode (default): local + Mask2Former
  python -m countertops.generate --preview
  python -m countertops.generate --image rooms/kitchens/kit1.jpg --preview

  # Local model only (faster, no Mask2Former download)
  python -m countertops.generate --mode local --preview

  # Mask2Former only
  python -m countertops.generate --mode m2f --preview
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
    load_mask2former,
    generate_combined_mask,
    save_combined_preview,
)


def process_image(img_path: Path, predictor, output_dir: Path, preview: bool,
                   m2f=None):
    """
    Run inference on one image and save the mask.

    Parameters
    ----------
    m2f : tuple (processor, model, device) or None
        If provided, uses combined mode (local + Mask2Former).
        If None, uses local model only.
    """
    print(f"\nProcessing: {img_path.name}")

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  ERROR: could not read {img_path}")
        return

    if m2f is not None:
        # ── Combined mode ──
        m2f_proc, m2f_model, m2f_device = m2f
        result = generate_combined_mask(
            image_bgr, predictor, m2f_proc, m2f_model, m2f_device
        )
        final_mask = result["combined"]

        local_px = np.count_nonzero(result["local_mask"])
        m2f_px   = np.count_nonzero(result["m2f_mask"])
        floor_px = np.count_nonzero(result["m2f_floor"])
        final_px = np.count_nonzero(final_mask)
        print(f"  Local:{local_px}px  M2F:{m2f_px}px  Floor(removed):{floor_px}px  → Final:{final_px}px")

        instances = result["instances"]
    else:
        # ── Local-only mode ──
        mask, instances = generate_mask(image_bgr, predictor)
        final_mask = clean_mask(mask)
        result = None

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
        if result is not None:
            preview_path = output_dir / f"{img_path.stem}_combined_preview.png"
            save_combined_preview(image_bgr, result, preview_path)
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
    parser.add_argument("--mode", choices=["combined", "local", "m2f"],
                        default="combined",
                        help="combined=both models (default), local=Detectron2 only, m2f=Mask2Former only")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load models based on mode
    predictor = None
    m2f = None

    if args.mode in ("combined", "local"):
        predictor = build_predictor(confidence=args.confidence)

    if args.mode in ("combined", "m2f"):
        m2f_proc, m2f_model, m2f_device = load_mask2former()
        m2f = (m2f_proc, m2f_model, m2f_device)

    if args.mode == "m2f":
        # In m2f-only mode we still need a predictor for the interface,
        # but we can skip it by handling separately
        predictor = build_predictor(confidence=args.confidence)

    print(f"\nMode: {args.mode}")

    if args.image:
        process_image(args.image, predictor, args.output, args.preview, m2f=m2f)
    else:
        exts = ("*.jpg", "*.jpeg", "*.png")
        image_paths = []
        for ext in exts:
            image_paths.extend(sorted(args.input.glob(ext)))
        print(f"Found {len(image_paths)} images in {args.input}")

        for img_path in image_paths:
            process_image(img_path, predictor, args.output, args.preview, m2f=m2f)

    print(f"\nDone! Masks saved to: {args.output}")


if __name__ == "__main__":
    main()
