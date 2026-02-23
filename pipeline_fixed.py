"""
Fixed pipeline for staircase tile visualization with per-step homographies.

Treats each step as an independent surface with separate tread and riser processing.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import SURFACE_IDS, DEFAULT_SURFACES, DEFAULT_ROTATION_ANGLE
from model import load_model, segment_image
from surfaces import extract_surface_mask, clean_mask, combine_masks
from tile_engine import build_full_tile_grid, composite_tile_on_surface
from staircase_engine_v2 import process_staircase


def save_segmentation_preview(room_image, masks: dict, output_path: Path):
    """Save preview of segmentation masks."""
    n = len(masks) + 1
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 5 * rows))
    
    plt.subplot(rows, cols, 1)
    plt.imshow(room_image)
    plt.title("Original")
    plt.axis("off")
    
    for i, (name, m) in enumerate(masks.items(), start=2):
        plt.subplot(rows, cols, i)
        plt.imshow(m, cmap="gray")
        plt.title(f"{name}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fixed staircase tile visualiser with per-step homographies"
    )
    parser.add_argument("--room", help="Path to room image")
    parser.add_argument("--tile", help="Path to tile image")
    parser.add_argument(
        "--surfaces", nargs="+", default=DEFAULT_SURFACES,
        help=f"Surfaces to tile. Available: {list(SURFACE_IDS.keys())}"
    )
    parser.add_argument("--rotation", type=float, default=DEFAULT_ROTATION_ANGLE,
                        help="Tile rotation angle in degrees")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug images")
    parser.add_argument("--list-surfaces", action="store_true",
                        help="Print available surface names and exit")
    args = parser.parse_args()

    # List surfaces
    if args.list_surfaces:
        print("Available surfaces (ADE20K label IDs):")
        for name, lid in SURFACE_IDS.items():
            print(f"  {name:15s}  →  ID {lid}")
        print("\nNote: 'screen' (ID 59) is used for staircase detection")
        return

    if not args.room or not args.tile:
        parser.error("--room and --tile are required (unless --list-surfaces)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load images
    room_pil = Image.open(args.room).convert("RGB")
    tile_pil = Image.open(args.tile).convert("RGB")
    room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)
    tile_bgr = cv2.cvtColor(np.array(tile_pil), cv2.COLOR_RGB2BGR)

    # Segment
    processor, model, device = load_model()
    seg_map = segment_image(room_pil, processor, model, device)
    print(f"Segmentation complete – unique labels: {np.unique(seg_map).tolist()}")

    # Process surfaces
    result_bgr = room_bgr.copy()
    masks = {}
    
    # Handle staircase (screen) separately with per-step processing
    if 'screen' in args.surfaces:
        print("Processing staircase with advanced per-step logic...")
        
        # Extract staircase mask
        stair_mask = extract_surface_mask(seg_map, 'screen')
        stair_mask = clean_mask(stair_mask)
        
        if np.sum(stair_mask > 0) > 0:
            # Use the new staircase engine for per-step logic
            result_bgr = process_staircase(result_bgr, stair_mask, tile_bgr)
            print("Applied tiles to staircase steps using advanced logic")
        else:
            print("No staircase detected")
    
    # Handle other surfaces (floor, wall, etc.) with original method
    other_surfaces = [s for s in args.surfaces if s != 'screen']
    if other_surfaces:
        print(f"Processing other surfaces: {other_surfaces}")
        
        # Extract masks for other surfaces
        for surface_name in other_surfaces:
            mask = extract_surface_mask(seg_map, surface_name)
            mask = clean_mask(mask)
            masks[surface_name] = mask
        
        # Save preview of other surface masks
        if masks:
            save_segmentation_preview(room_pil, masks, outdir / "other_surfaces_preview.png")
            for name, m in masks.items():
                Image.fromarray(m).save(outdir / f"mask_{name}.png")
            
            # Combine other surface masks
            combined_mask = combine_masks(masks)
            
            # Apply tiles to other surfaces using original method
            full_tile = build_full_tile_grid(
                room_bgr, combined_mask, tile_bgr,
                rotation_angle=args.rotation
            )
            
            if args.debug:
                cv2.imwrite(str(outdir / "debug_other_tiles.png"), full_tile)
            
            # Composite other surface tiles
            result_bgr = composite_tile_on_surface(result_bgr, combined_mask, full_tile)
            print("Applied tiles to other surfaces")

    # Save final result
    surfaces_str = "_".join(args.surfaces)
    out_name = f"fixed_tile_applied_{surfaces_str}.png"
    cv2.imwrite(str(outdir / out_name), result_bgr)
    print(f"Saved: {(outdir / out_name).resolve()}")
    print(f"All outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
