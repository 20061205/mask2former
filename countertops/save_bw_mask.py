"""
Generate and save final merged countertop mask as black and white image.

This script outputs only the binary mask (black background, white countertop regions)
without any visualization overlays.

Usage
-----
  cd "E:\\tile viz\\mask2former"

  # Single image - saves mask as B&W
  python -m countertops.save_bw_mask --image rooms/kitchens/kit1.jpg

  # Custom output location
  python -m countertops.save_bw_mask --image rooms/kitchens/kit4.jpg \
         --output outputs/kit4_bw_mask.png --confidence 0.5

  # Process all images in a folder
  python -m countertops.save_bw_mask --input rooms/kitchens \
         --output outputs/masks

  # Use MaskRCNN only (no SAM refinement)
  python -m countertops.save_bw_mask --image rooms/kitchens/kit1.jpg --mode local
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import sys

from .config import INPUT_DIR, OUTPUT_DIR, CONFIDENCE
from .mask_generator import (
    build_predictor,
    load_sam,
    generate_mask,
    clean_mask,
    generate_sam_mask,
)


def save_final_mask(
    img_path: Path,
    output_path: Path,
    predictor,
    sam_predictor=None,
    mode="sam",
):
    """
    Generate and save the final merged mask as black and white image.

    Parameters
    ----------
    img_path      : input image path
    output_path   : where to save the B&W mask
    predictor     : Detectron2 predictor
    sam_predictor : SAM predictor (optional, for SAM mode)
    mode          : 'sam' or 'local'

    Returns
    -------
    bool : True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Processing: {img_path.name}")
    print(f"{'='*70}")

    # Read image
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  ❌ ERROR: Could not read {img_path}")
        return False

    try:
        # Generate mask based on mode
        if mode == "sam" and sam_predictor is not None:
            # SAM mode: Detectron2 + SAM refinement
            print(f"  Mode: SAM (Detectron2 + SAM refinement with filtering)")
            sam_result = generate_sam_mask(image_bgr, predictor, sam_predictor, filter_non_countertops=True)
            final_mask = sam_result["sam_mask"]
            num_instances = len(sam_result["all_masks"])
            print(f"  Detected: {num_instances} countertop instance(s)")
        else:
            # Local mode: Detectron2 only
            print(f"  Mode: Local (Detectron2 only)")
            mask, instances = generate_mask(image_bgr, predictor)
            final_mask = clean_mask(mask)
            num_instances = len(instances)
            print(f"  Detected: {num_instances} object(s)")

        # Save as black and white image
        cv2.imwrite(str(output_path), final_mask)
        
        # Calculate statistics
        white_pixels = np.count_nonzero(final_mask)
        total_pixels = final_mask.shape[0] * final_mask.shape[1]
        percentage = (white_pixels / total_pixels) * 100

        print(f"  ✅ Mask saved: {output_path}")
        print(f"     Size: {final_mask.shape[1]}x{final_mask.shape[0]} pixels")
        print(f"     White pixels: {white_pixels:,} ({percentage:.2f}%)")
        print(f"     Black pixels: {total_pixels - white_pixels:,}")
        
        return True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Save final merged countertop mask as black and white image"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single image to process (e.g., rooms/kitchens/kit1.jpg)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DIR,
        help="Input folder with images (default: rooms/kitchens)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path or folder (default: outputs/masks/<image>_mask.png)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE,
        help="Detection confidence threshold (default: from config)",
    )
    parser.add_argument(
        "--mode",
        choices=["sam", "local"],
        default="sam",
        help="sam=Detectron2+SAM (default), local=Detectron2 only",
    )

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════
    # Load models
    # ═══════════════════════════════════════════════════════════════════
    print("\n🚀 Loading models...")
    print("─" * 70)
    
    try:
        predictor = build_predictor(confidence=args.confidence)
        sam_predictor = None
        
        if args.mode == "sam":
            sam_predictor = load_sam()
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        sys.exit(1)

    print("✅ Models loaded successfully!\n")

    # ═══════════════════════════════════════════════════════════════════
    # Process images
    # ═══════════════════════════════════════════════════════════════════
    
    if args.image:
        # Single image mode
        if not args.image.exists():
            print(f"❌ ERROR: Image not found: {args.image}")
            sys.exit(1)

        if args.output:
            output_path = args.output
        else:
            output_dir = Path("outputs") / "masks"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{args.image.stem}_mask.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = save_final_mask(
            args.image,
            output_path,
            predictor,
            sam_predictor,
            args.mode,
        )

        if success:
            print(f"\n✅ Done! Black & white mask saved to: {output_path}")
        else:
            print(f"\n❌ Processing failed.")
            sys.exit(1)

    else:
        # Batch mode
        input_dir = args.input
        if not input_dir.exists():
            print(f"❌ ERROR: Input folder not found: {input_dir}")
            sys.exit(1)

        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"❌ ERROR: No images found in {input_dir}")
            sys.exit(1)

        # Setup output directory
        if args.output:
            output_dir = args.output
        else:
            output_dir = Path("outputs") / "masks"
        
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process all images
        print(f"📁 Found {len(image_files)} images in {input_dir}")
        print(f"📂 Output directory: {output_dir}\n")

        success_count = 0
        for img_path in image_files:
            output_path = output_dir / f"{img_path.stem}_mask.png"
            
            if save_final_mask(
                img_path,
                output_path,
                predictor,
                sam_predictor,
                args.mode,
            ):
                success_count += 1

        print(f"\n{'='*70}")
        print(f"✅ Completed: {success_count}/{len(image_files)} images processed successfully")
        print(f"📂 Results saved to: {output_dir}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
