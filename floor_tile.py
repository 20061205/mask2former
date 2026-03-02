"""
CLI entry-point for Mask2Former tile visualisation.

Supports multiple surfaces: floor, wall, countertop, stairway, cabinet, etc.
Supports real-world dimension-aware tile count (floor area + tile size → exact grid).

Usage examples
--------------
  # Tile the floor only (default)
  python pipeline.py --room room.jpg --tile tile.jpg

  # Tile floor with real-world dimensions (10'×12' room, 24"×24" tiles → 60 tiles)
  python pipeline.py --room room.jpg --tile tile.jpg \\
      --floor-width 10 --floor-length 12 --tile-width 24 --tile-height 24

  # Tile floor + stairway
  python pipeline.py --room room.jpg --tile tile.jpg --surfaces floor stairway

  # Tile kitchen countertop
  python pipeline.py --room kitchen.jpg --tile marble.jpg --surfaces countertop

  # List all available surface names
  python pipeline.py --list-surfaces
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import SURFACE_IDS, DEFAULT_SURFACES, DEFAULT_ROTATION_ANGLE, DEFAULT_CAMERA_TILT
from model import load_model, segment_image
from surfaces import extract_all_masks, combine_masks
from tile_engine import (
    build_full_tile_grid,
    composite_tile_on_surface,
    calculate_tile_pixel_size,
)


# ── Debug / preview helpers ──────────────────────────────────────────

def save_segmentation_preview(room_image, masks: dict, output_path: Path):
    n = len(masks) + 1  # +1 for original
    plt.figure(figsize=(5 * n, 5))

    plt.subplot(1, n, 1)
    plt.imshow(room_image)
    plt.title("Original")
    plt.axis("off")

    for i, (name, m) in enumerate(masks.items(), start=2):
        plt.subplot(1, n, i)
        plt.imshow(m, cmap="gray")
        plt.title(f"{name.capitalize()} Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_debug_images(outdir: Path, surface_name: str,
                      mask, full_tile, room_bgr):
    prefix = f"debug_{surface_name}"
    cv2.imwrite(str(outdir / f"{prefix}_1_mask.png"), mask)
    cv2.imwrite(str(outdir / f"{prefix}_2_tile_grid.png"), full_tile)

    cutout = room_bgr.copy()
    cutout[mask > 0] = 0
    cv2.imwrite(str(outdir / f"{prefix}_3_room_cutout.png"), cutout)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mask2Former tile visualiser – floor, wall, countertop, stairs & more"
    )
    parser.add_argument("--room", help="Path to room image")
    parser.add_argument("--tile", help="Path to tile image")
    parser.add_argument(
        "--surfaces", nargs="+", default=DEFAULT_SURFACES,
        help=f"Surfaces to tile. Available: {list(SURFACE_IDS.keys())}"
    )
    parser.add_argument("--rotation", type=float, default=DEFAULT_ROTATION_ANGLE,
                        help="Tile rotation angle in degrees")
    parser.add_argument("--camera-tilt", type=float, default=DEFAULT_CAMERA_TILT,
                        help="Camera tilt factor (1.0=flat, 1.3=default, 2.0=strong perspective)")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug images")
    parser.add_argument("--list-surfaces", action="store_true",
                        help="Print available surface names and exit")

    # ── Real-world dimension args ────────────────────────────────────
    parser.add_argument("--floor-width", type=float, default=None,
                        help="Real floor width in feet (e.g. 10)")
    parser.add_argument("--floor-length", type=float, default=None,
                        help="Real floor length in feet (e.g. 12)")
    parser.add_argument("--tile-width", type=float, default=None,
                        help="Single tile width in inches (e.g. 24)")
    parser.add_argument("--tile-height", type=float, default=None,
                        help="Single tile height in inches (e.g. 24)")
    args = parser.parse_args()

    # ── List surfaces ────────────────────────────────────────────────
    if args.list_surfaces:
        print("Available surfaces (ADE20K label IDs):")
        for name, lid in SURFACE_IDS.items():
            print(f"  {name:15s}  →  ID {lid}")
        return

    if not args.room or not args.tile:
        parser.error("--room and --tile are required (unless --list-surfaces)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────
    room_pil = Image.open(args.room).convert("RGB")
    tile_pil = Image.open(args.tile).convert("RGB")
    room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)
    tile_bgr = cv2.cvtColor(np.array(tile_pil), cv2.COLOR_RGB2BGR)

    # ── Segment ──────────────────────────────────────────────────────
    processor, model, device = load_model()
    seg_map = segment_image(room_pil, processor, model, device)
    print(f"Segmentation complete – unique labels: {np.unique(seg_map).tolist()}")

    # ── Extract masks ────────────────────────────────────────────────
    masks = extract_all_masks(seg_map, args.surfaces, clean=True)

    # Save preview
    save_segmentation_preview(room_pil, masks, outdir / "segmentation_preview.png")
    for name, m in masks.items():
        Image.fromarray(m).save(outdir / f"mask_{name}.png")
    print(f"Saved {len(masks)} mask(s): {list(masks.keys())}")

    # ── Combine all selected surfaces into one mask ──────────────────
    combined_mask = combine_masks(masks)

    # ── Build tile grid & composite ──────────────────────────────────
    # Determine tile pixel size: from real-world dims or default
    tile_w_px = None
    tile_h_px = None

    dim_args = (args.floor_width, args.floor_length,
                args.tile_width, args.tile_height)
    if all(d is not None for d in dim_args):
        print("Computing tile size from real-world dimensions...")
        tile_w_px, tile_h_px, n_across, n_down = calculate_tile_pixel_size(
            combined_mask,
            args.floor_width, args.floor_length,
            args.tile_width, args.tile_height,
        )
        print(f"  Total tiles: ~{n_across * n_down} "
              f"({n_across} × {n_down})")
    elif any(d is not None for d in dim_args):
        parser.error("Provide ALL four dimension args: "
                     "--floor-width, --floor-length, --tile-width, --tile-height")

    print("Building tile grid...")
    full_tile = build_full_tile_grid(
        room_bgr, combined_mask, tile_bgr,
        rotation_angle=args.rotation,
        tile_w=tile_w_px,
        tile_h=tile_h_px,
        camera_tilt=args.camera_tilt,
    )

    if args.debug:
        save_debug_images(outdir, "combined", combined_mask, full_tile, room_bgr)

    result_bgr = composite_tile_on_surface(room_bgr, combined_mask, full_tile)

    # ── Save (maximum quality) ────────────────────────────────────────
    out_name = "tile_applied_" + "_".join(args.surfaces) + ".png"
    cv2.imwrite(str(outdir / out_name), result_bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])   # 0-9; lower = bigger/faster
    # Also save a high-quality JPEG for quick viewing
    jpg_name = "tile_applied_" + "_".join(args.surfaces) + ".jpg"
    cv2.imwrite(str(outdir / jpg_name), result_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"Saved: {(outdir / out_name).resolve()}")
    print(f"Saved: {(outdir / jpg_name).resolve()}")
    print(f"All outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
