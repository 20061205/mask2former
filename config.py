"""
Configuration: model name, ADE20K surface label IDs, and defaults.
"""

# ── Model ────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/mask2former-swin-base-ade-semantic"

# ── ADE20K label IDs for tileable surfaces ───────────────────────────
# Full list: https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8
SURFACE_IDS = {
    "floor":      3,
    "wall":       0,
    "countertop": [70, 45, 73],  # countertop + table-top-like surfaces
    "stairway":   53,            # stairs / staircase
    "cabinet":    10,            # kitchen cabinet fronts
    "ceiling":    5,
    "table":      15,
    "shelf":      24,
}

# Surfaces that are tileable by default
DEFAULT_SURFACES = ["floor"]

# ── Tile engine defaults ─────────────────────────────────────────────
DEFAULT_TILE_SIZE = 120
DEFAULT_GROUT_WIDTH = 3
DEFAULT_ROTATION_ANGLE = 0.0
TILE_GRID_SCALE = 3        # oversized pattern multiplier


# ── Mask cleaning defaults ───────────────────────────────────────────
MASK_KERNEL_SIZE = 5
MASK_CLOSE_ITERATIONS = 2
