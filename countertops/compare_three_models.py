"""
Visualize masks from 3 different models side-by-side.

Shows black & white masks from:
1. MaskRCNN (custom trained for countertops)
2. SAM (refined from MaskRCNN boxes)
3. Mask2Former (semantic segmentation - ADE20K)

Usage
-----
  cd "E:\\tile viz\\mask2former"

  # Single image with default surfaces (countertop, table, cabinet)
  python -m countertops.compare_three_models --image rooms/kitchens/kit1.jpg

  # Custom surfaces for Mask2Former
  python -m countertops.compare_three_models --image rooms/kitchens/kit4.jpg \
         --surfaces countertop table cabinet shelf

  # Custom output and confidence
  python -m countertops.compare_three_models --image rooms/kitchens/kit4.jpg \
         --output outputs/kit4_3models.jpg --confidence 0.5

  # Disable SAM filtering (include all detected regions)
  python -m countertops.compare_three_models --image rooms/kitchens/kit4.jpg --no-filter

  # Process all images in a folder
  python -m countertops.compare_three_models --input rooms/kitchens \
         --output outputs/3model_comparisons
"""

import argparse
from pathlib import Path
import cv2
import sys

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from countertops.config import INPUT_DIR, OUTPUT_DIR, CONFIDENCE
from countertops.mask_generator import (
    build_predictor,
    load_sam,
    visualize_three_model_masks,
)
from model import load_model


def process_single_image(
    img_path: Path,
    output_path: Path,
    predictor,
    sam_predictor,
    mask2former_processor,
    mask2former_model,
    mask2former_device,
    surfaces: list[str],
    filter_sam: bool = True,
):
    """Process a single image and create 3-model comparison."""
    print(f"\n{'='*70}")
    print(f"Processing: {img_path.name}")
    print(f"{'='*70}")

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  ❌ ERROR: Could not read {img_path}")
        return False

    try:
        result = visualize_three_model_masks(
            image_bgr=image_bgr,
            predictor=predictor,
            sam_predictor=sam_predictor,
            mask2former_processor=mask2former_processor,
            mask2former_model=mask2former_model,
            mask2former_device=mask2former_device,
            output_path=output_path,
            surface_names=surfaces,
            filter_sam=filter_sam,
        )

        print(f"\n  📊 Summary:")
        print(f"     MaskRCNN: {result['maskrcnn_result']['pixels']:,} pixels")
        print(f"     SAM: {result['sam_result']['pixels']:,} pixels")
        print(f"     Mask2Former: {result['mask2former_result']['pixels']:,} pixels")
        print(f"     Combined AND: {result['combined_result']['pixels']:,} pixels ({result['combined_result']['percentage']:.1f}%)")
        print(f"     Output: {output_path}")
        
        return True

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare masks from 3 models: MaskRCNN, SAM, Mask2Former"
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
        help="Output path or folder (default: outputs/3models_<image>.jpg)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE,
        help="Detection confidence threshold for MaskRCNN (default: from config)",
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        default=["countertop", "table", "cabinet"],
        help="Surfaces for Mask2Former (default: countertop table cabinet)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable SAM mask filtering (include all detected objects)",
    )

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════════
    # Load all 3 models
    # ═══════════════════════════════════════════════════════════════════
    print("\n🚀 Loading models...")
    print("─" * 70)
    
    try:
        print("  [1/3] Loading MaskRCNN...")
        predictor = build_predictor(confidence=args.confidence)
        
        print("  [2/3] Loading SAM...")
        sam_predictor = load_sam()
        
        print("  [3/3] Loading Mask2Former...")
        mask2former_processor, mask2former_model, mask2former_device = load_model()
        
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("✅ All models loaded successfully!\n")
    print(f"Mask2Former surfaces: {', '.join(args.surfaces)}")

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
            output_path = output_dir / f"3models_{args.image.stem}.jpg"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = process_single_image(
            args.image,
            output_path,
            predictor,
            sam_predictor,
            mask2former_processor,
            mask2former_model,
            mask2former_device,
            args.surfaces,
            filter_sam=not args.no_filter,
        )

        if success:
            print(f"\n✅ Done! 3-Model comparison saved to: {output_path}")
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
            output_dir = Path("outputs") / "3model_comparisons"
        
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process all images
        print(f"📁 Found {len(image_files)} images in {input_dir}")
        print(f"📂 Output directory: {output_dir}\n")

        success_count = 0
        for img_path in image_files:
            output_path = output_dir / f"3models_{img_path.stem}.jpg"
            
            if process_single_image(
                img_path,
                output_path,
                predictor,
                sam_predictor,
                mask2former_processor,
                mask2former_model,
                mask2former_device,
                args.surfaces,
                filter_sam=not args.no_filter,
            ):
                success_count += 1

        print(f"\n{'='*70}")
        print(f"✅ Completed: {success_count}/{len(image_files)} images processed successfully")
        print(f"📂 Results saved to: {output_dir}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
