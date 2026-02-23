"""
Configuration for countertop mask generation.
Uses two models:
  1. Local Detectron2 Mask R-CNN  (trained on custom kitchen dataset)
  2. Mask2Former ADE20K semantic  (pre-trained, broad scene understanding)
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
MODEL_WEIGHTS = Path(__file__).parent / "model_final.pth"
INPUT_DIR     = Path(__file__).parent.parent / "rooms" / "kitchens"
OUTPUT_DIR    = Path(__file__).parent / "masks"

# ═══════════════════════════════════════════════════════════════════════
# LOCAL DETECTRON2 MODEL (custom-trained)
# ═══════════════════════════════════════════════════════════════════════
MODEL_CONFIG  = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES   = 4
CONFIDENCE    = 0.5

# Class names (0-indexed, matching COCO annotation category order)
# COCO ids: 1=cabinet, 2=countertop, 3=floor, 4=wall
CLASS_NAMES = ["cabinet", "countertop", "floor", "wall"]

# Which local model classes to include in the countertop mask
TARGET_CLASSES = [2]  # 2 = countertop

# ═══════════════════════════════════════════════════════════════════════
# MASK2FORMER ADE20K MODEL (pre-trained)
# ═══════════════════════════════════════════════════════════════════════
M2F_MODEL_NAME = "facebook/mask2former-swin-base-ade-semantic"

# ADE20K label IDs for countertop-related surfaces (to INCLUDE)
M2F_COUNTERTOP_IDS = [70, 45, 73]   # countertop, table, kitchen_island
M2F_CABINET_IDS    = [10]            # cabinet
M2F_INCLUDE_IDS    = M2F_COUNTERTOP_IDS + M2F_CABINET_IDS  # combined include

# ADE20K label IDs to EXCLUDE (used to remove false positives)
M2F_FLOOR_ID  = 3
M2F_WALL_ID   = 0

# ═══════════════════════════════════════════════════════════════════════
# MASK CLEANING
# ═══════════════════════════════════════════════════════════════════════
MASK_KERNEL_SIZE    = 1
MASK_CLOSE_ITER     = 2

# ═══════════════════════════════════════════════════════════════════════
# TILE APPLICATION
# ═══════════════════════════════════════════════════════════════════════
TILE_DIR            = Path(__file__).parent.parent / "tiles"
TILE_SIZE           = 100           # tile cell size in pixels (slab-like)
GROUT_WIDTH         = 2             # thin grout for countertop slab look
ROTATION_ANGLE      = 10.0           # tile rotation angle (degrees)
TILE_GRID_SCALE     = 3             # oversized pattern multiplier
FEATHER_RADIUS      = 11            # edge blending radius
SHADOW_BASE         = 0.65          # edge shadow darkening (0=black, 1=none)
HIGHLIGHT_STRENGTH  = 0.30          # how much original specular to keep
DETAIL_STRENGTH     = 0.35          # room micro-shadow transfer strength
