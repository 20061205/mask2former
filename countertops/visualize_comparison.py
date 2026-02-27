"""
Visualize MaskRCNN and SAM comparison for countertop detection.

This script creates a comprehensive 6-panel visualization showing:
- Row 1: Original | MaskRCNN Detections | MaskRCNN Masks
- Row 2: SAM Labeled Masks | SAM Colored Instances | Final Merged Overlay

Usage
-----
  cd "E:\\tile viz\\mask2former"

  # Single image
  python -m countertops.visualize_comparison --image rooms/kitchens/kit1.jpg

  # Custom output
  python -m countertops.visualize_comparison --image rooms/kitchens/kit4.jpg \
         --output outputs/kit4_comparison.jpg --confidence 0.5

  # Process all images in a folder
  python -m countertops.visualize_comparison --input rooms/kitchens \
         --output outputs/comparisons
"""

import argparse
from pathlib import Path
import cv2
import sys

from .config import INPUT_DIR, OUTPUT_DIR, CONFIDENCE
from .mask_generator import (
    build_predictor,
    load_sam,
    visualize_maskrcnn_and_sam,
)


def process_single_image(
    img_path: Path,
    output_path: Path,
    predictor,
    sam_predictor,
    confidence: float,
    save_mask: bool = False,
):
    """Process a single image and create comparison visualization."""
    print(f"\n{'='*70}")
    print(f"Processing: {img_path.name}")
    print(f"{'='*70}")

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  ERROR: Could not read {img_path}")
        return False

    try:
        result = visualize_maskrcnn_and_sam(
            image_bgr=image_bgr,
            predictor=predictor,
            sam_predictor=sam_predictor,
            output_path=output_path,
        )

        # Print summary
        maskrcnn = result["maskrcnn_result"]
        sam = result["sam_result"]
        
        print(f"\n  📊 Summary:")
        print(f"     MaskRCNN: {len(maskrcnn['instances'])} detections")
        print(f"     SAM: {len(sam['all_masks'])} refined masks")
        print(f"     Output: {output_path}")
        
        # Save B&W mask if requested
        if save_mask:
            mask_path = output_path.parent / f"{output_path.stem}_bw_mask.png"
            final_mask = sam["sam_mask"]
            cv2.imwrite(str(mask_path), final_mask)
            import numpy as np
            white_px = np.count_nonzero(final_mask)
            print(f"     B&W Mask: {mask_path} ({white_px} white pixels)")
        
        return True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MaskRCNN and SAM comparison for countertop detection"
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
        help="Output path or folder (default: outputs/comparison_<image>.jpg)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE,
        help="Detection confidence threshold (default: from config)",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save final merged mask as black and white PNG",
    )

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════
    # Load models
    # ═══════════════════════════════════════════════════════════════════
    print("\n🚀 Loading models...")
    print("─" * 70)
    
    try:
        predictor = build_predictor(confidence=args.confidence)
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
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"comparison_{args.image.stem}.jpg"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = process_single_image(
            args.image,
            output_path,
            predictor,
            sam_predictor,
            args.confidence,
            args.save_mask,
        )

        if success:
            print(f"\n✅ Done! Visualization saved to: {output_path}")
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
            output_dir = Path("outputs") / "comparisons"
        
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process all images
        print(f"📁 Found {len(image_files)} images in {input_dir}")
        print(f"📂 Output directory: {output_dir}\n")

        success_count = 0
        for img_path in image_files:
            output_path = output_dir / f"comparison_{img_path.stem}.jpg"
            
            if process_single_image(
                img_path,
                output_path,
                predictor,
                sam_predictor,
                args.confidence,
                args.save_mask,
            ):
                success_count += 1

        print(f"\n{'='*70}")
        print(f"✅ Completed: {success_count}/{len(image_files)} images processed successfully")
        print(f"📂 Results saved to: {output_dir}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
