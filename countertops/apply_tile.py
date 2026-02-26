"""
Apply tiles to countertop areas using pre-generated masks or live generation.

Usage
-----
  cd "E:\\tile viz\\mask2former"

  # Use a pre-generated mask
  python -m countertops.apply_tile --room rooms/kitchens/kit4.jpg \
         --tile tiles/tile1.jpg \
         --mask countertops/masks/kit4_countertop_mask.png

  # Generate mask on-the-fly (combined mode) and apply tile
  python -m countertops.apply_tile --room rooms/kitchens/kit4.jpg \
         --tile tiles/tile1.jpg --live

  # Customise tile size, grout, rotation
  python -m countertops.apply_tile --room rooms/kitchens/kit4.jpg \
         --tile "tiles/granite tile.jpg" --live \
         --tile-size 400 --grout 3 --rotation 5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import (
    INPUT_DIR,
    OUTPUT_DIR,
    TILE_DIR,
    TILE_SIZE,
    GROUT_WIDTH,
    ROTATION_ANGLE,
    CONFIDENCE,
)
from .tile_applicator import apply_tile, save_tile_preview


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Apply tile texture to countertop mask area"
    )
    parser.add_argument("--room", type=Path, required=True,
                        help="Path to kitchen / room image")
    parser.add_argument("--tile", type=Path, required=True,
                        help="Path to tile texture image")
    parser.add_argument("--mask", type=Path, default=None,
                        help="Path to pre-generated countertop mask PNG. "
                             "If omitted, use --live to generate on the fly.")
    parser.add_argument("--live", action="store_true",
                        help="Generate combined mask live (requires both models)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path (default: countertops/masks/<room>_tiled.png)")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE,
                        help=f"Tile cell size in pixels (default: {TILE_SIZE})")
    parser.add_argument("--grout", type=int, default=GROUT_WIDTH,
                        help=f"Grout line width (default: {GROUT_WIDTH})")
    parser.add_argument("--rotation", type=float, default=ROTATION_ANGLE,
                        help=f"Tile rotation angle in degrees (default: {ROTATION_ANGLE})")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE,
                        help=f"Detection confidence for live mode (default: {CONFIDENCE})")
    parser.add_argument("--preview", action="store_true",
                        help="Save a 3-panel comparison preview")
    args = parser.parse_args()

    # ── Load room and tile images ────────────────────────────────────
    room_bgr = load_image(args.room)
    tile_bgr = load_image(args.tile)
    print(f"Room : {args.room}  ({room_bgr.shape[1]}x{room_bgr.shape[0]})")
    print(f"Tile : {args.tile}  ({tile_bgr.shape[1]}x{tile_bgr.shape[0]})")

    # ── Get or generate mask ─────────────────────────────────────────
    if args.mask:
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {args.mask}")
        print(f"Mask : {args.mask}  ({np.count_nonzero(mask)} white px)")
    elif args.live:
        print("Generating SAM mask (Detectron2 + SAM)...")
        from .mask_generator import (
            build_predictor,
            load_sam,
            generate_sam_mask,
        )
        predictor = build_predictor(confidence=args.confidence)
        sam_pred = load_sam()
        result = generate_sam_mask(room_bgr, predictor, sam_pred)
        mask = result["sam_mask"]
        print(f"Mask generated  ({np.count_nonzero(mask)} white px)")

        # Save the mask too
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        mask_path = OUTPUT_DIR / f"{args.room.stem}_countertop_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"Mask saved: {mask_path}")
    else:
        parser.error("Provide --mask <path> or use --live to generate one.")

    # ── Resize mask if needed ────────────────────────────────────────
    if mask.shape[:2] != room_bgr.shape[:2]:
        mask = cv2.resize(mask, (room_bgr.shape[1], room_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # ── Apply tile ───────────────────────────────────────────────────
    print(f"Applying tile  (size={args.tile_size}, grout={args.grout}, "
          f"rotation={args.rotation}°)...")
    result_bgr = apply_tile(
        room_bgr, mask, tile_bgr,
        tile_size=args.tile_size,
        grout=args.grout,
        rotation=args.rotation,
    )

    # ── Save output ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        out_path = OUTPUT_DIR / f"{args.room.stem}_tiled.png"

    cv2.imwrite(str(out_path), result_bgr)
    print(f"\nResult saved: {out_path}")

    # ── Optional preview ─────────────────────────────────────────────
    if args.preview:
        preview_path = out_path.with_name(out_path.stem + "_preview.png")
        save_tile_preview(room_bgr, mask, result_bgr, preview_path)

    print("Done!")


if __name__ == "__main__":
    main()
