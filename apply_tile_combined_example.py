"""
Example script demonstrating tile application with combined AND mask.

This uses the highest-accuracy mask by combining SAM + Mask2Former predictions.
The tile application preserves:
- 3D depth (edge shadows)
- Original lighting and gradients
- Specular highlights
- Micro-shadows and surface reflections

Usage:
------
cd "E:\\tile viz\\mask2former"
python apply_tile_combined_example.py
"""

from pathlib import Path
import sys

# Example images
ROOMS = [
    "rooms/kitchens/kit1.jpg",
    "rooms/kitchens/kit4.jpg",
]

TILES = [
    "tiles/tile1.jpg",
    "tiles/granite tile.jpg",
    "tiles/tile3.jpg",
]

def main():
    """Run tile application examples."""
    
    print("=" * 70)
    print("TILE APPLICATION WITH COMBINED AND MASK (SAM + Mask2Former)")
    print("=" * 70)
    print("\nThis demonstrates tile application using the highest-accuracy")
    print("combined mask from SAM + Mask2Former AND gate.")
    print("\nThe tile application preserves:")
    print("  ✓ 3D depth (edge shadows)")
    print("  ✓ Original lighting and gradients")
    print("  ✓ Specular highlights")
    print("  ✓ Micro-shadows and surface reflections")
    print()
    
    # Example 1: Basic usage
    print("\n" + "─" * 70)
    print("EXAMPLE 1: Basic combined mask + tile application")
    print("─" * 70)
    room = ROOMS[0]
    tile = TILES[0]
    
    cmd = (
        f'python -m countertops.apply_tile '
        f'--room "{room}" '
        f'--tile "{tile}" '
        f'--combined '
        f'--preview'
    )
    print(f"\nCommand:\n{cmd}\n")
    print("This will:")
    print("  1. Generate combined AND mask (SAM + Mask2Former)")
    print("  2. Save 3-model comparison visualization")
    print("  3. Apply tile texture with lighting preservation")
    print("  4. Save preview showing: original | mask | tiled result")
    
    # Example 2: Custom tile settings
    print("\n" + "─" * 70)
    print("EXAMPLE 2: Custom tile size, grout, and rotation")
    print("─" * 70)
    
    cmd = (
        f'python -m countertops.apply_tile '
        f'--room "{ROOMS[1]}" '
        f'--tile "{TILES[1]}" '
        f'--combined '
        f'--tile-size 350 '
        f'--grout 5 '
        f'--rotation 15 '
        f'--surfaces countertop table '
        f'--preview'
    )
    print(f"\nCommand:\n{cmd}\n")
    print("This will:")
    print("  1. Use larger tiles (350px) with wider grout (5px)")
    print("  2. Rotate tiles 15 degrees for visual interest")
    print("  3. Only mask countertop and table surfaces")
    print("  4. Preserve all lighting, shadows, and highlights")
    
    # Example 3: Multiple tiles
    print("\n" + "─" * 70)
    print("EXAMPLE 3: Try multiple tile textures")
    print("─" * 70)
    print("\nYou can batch process with different tiles:\n")
    
    for i, tile in enumerate(TILES, 1):
        cmd = (
            f'python -m countertops.apply_tile '
            f'--room "{ROOMS[0]}" '
            f'--tile "{tile}" '
            f'--combined '
            f'--output "outputs/kit1_tile{i}.png"'
        )
        print(f"  {i}. {cmd}")
    
    print("\n" + "─" * 70)
    print("KEY FLAGS:")
    print("─" * 70)
    print("  --combined     : Use SAM + Mask2Former AND gate (highest accuracy)")
    print("  --live         : Use SAM only (faster)")
    print("  --mask <path>  : Use pre-generated mask file")
    print("  --tile-size N  : Tile cell size in pixels (default: 300)")
    print("  --grout N      : Grout line width in pixels (default: 2)")
    print("  --rotation D   : Tile rotation angle in degrees (default: 0)")
    print("  --surfaces ... : Surfaces for Mask2Former (default: countertop table cabinet)")
    print("  --preview      : Save 3-panel comparison image")
    print()
    
    print("\n" + "=" * 70)
    print("To run Example 1, press Enter (or Ctrl+C to exit)")
    print("=" * 70)
    
    try:
        input()
        import subprocess
        room = ROOMS[0]
        tile = TILES[0]
        cmd = [
            sys.executable, "-m", "countertops.apply_tile",
            "--room", room,
            "--tile", tile,
            "--combined",
            "--preview"
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nExiting without running example.")
        sys.exit(0)


if __name__ == "__main__":
    main()
