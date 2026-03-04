"""
Core tile-application logic, wrapping the existing pipeline modules.
Model is loaded once at startup and reused across requests.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add project root to path so we can import existing modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import load_model, segment_image
from floor.surfaces import extract_all_masks, combine_masks
from floor.tile_engine import (
    build_full_tile_grid,
    composite_tile_on_surface,
    calculate_tile_pixel_size,
)
from floor.config import SURFACE_IDS, DEFAULT_CAMERA_TILT


# ── Upload storage ───────────────────────────────────────────────────
UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


def save_inputs(room_bytes: bytes, tile_bytes: bytes, prefix: str = "") -> dict:
    """Save uploaded room & tile images to uploads/ folder. Returns saved paths."""
    import uuid
    uid = uuid.uuid4().hex[:8]
    tag = f"{prefix}_{uid}" if prefix else uid

    room_path = UPLOADS_DIR / f"{tag}_room.png"
    tile_path = UPLOADS_DIR / f"{tag}_tile.png"
    room_path.write_bytes(room_bytes)
    tile_path.write_bytes(tile_bytes)

    return {"room": str(room_path), "tile": str(tile_path), "id": tag}


# ── Image decoding helper ────────────────────────────────────────────

def decode_images(room_bytes: bytes, tile_bytes: bytes):
    """Decode uploaded image bytes to BGR numpy arrays."""
    room_bgr = cv2.imdecode(np.frombuffer(room_bytes, np.uint8), cv2.IMREAD_COLOR)
    tile_bgr = cv2.imdecode(np.frombuffer(tile_bytes, np.uint8), cv2.IMREAD_COLOR)
    return room_bgr, tile_bgr


# ── Singleton model ──────────────────────────────────────────────────
_processor = _model = _device = None


def get_model():
    global _processor, _model, _device
    if _processor is None:
        _processor, _model, _device = load_model()
    return _processor, _model, _device


def list_surfaces() -> dict[str, int | list[int]]:
    return dict(SURFACE_IDS)


def apply_tiles(
    room_bytes: bytes,
    tile_bytes: bytes,
    surfaces: list[str],
    rotation: float = 0.0,
    floor_width: float | None = None,
    floor_length: float | None = None,
    tile_width: float | None = None,
    tile_height: float | None = None,
    camera_tilt: float = DEFAULT_CAMERA_TILT,
) -> tuple[bytes, dict | None]:
    """
    Run the full pipeline and return (result_png_bytes, dimension_info | None).
    """
    # Decode images
    room_arr = np.frombuffer(room_bytes, np.uint8)
    tile_arr = np.frombuffer(tile_bytes, np.uint8)
    room_bgr = cv2.imdecode(room_arr, cv2.IMREAD_COLOR)
    tile_bgr = cv2.imdecode(tile_arr, cv2.IMREAD_COLOR)
    room_pil = Image.fromarray(cv2.cvtColor(room_bgr, cv2.COLOR_BGR2RGB))

    # Segment
    processor, model, device = get_model()
    seg_map = segment_image(room_pil, processor, model, device)

    # Masks
    masks = extract_all_masks(seg_map, surfaces, clean=True)
    combined_mask = combine_masks(masks)

    # Dimension-aware tile sizing
    tile_w_px = tile_h_px = None
    dim_info = None
    dims = (floor_width, floor_length, tile_width, tile_height)

    if all(d is not None for d in dims):
        tw, th, na, nd = calculate_tile_pixel_size(
            combined_mask, floor_width, floor_length,
            tile_width, tile_height,
        )
        tile_w_px, tile_h_px = tw, th
        dim_info = {
            "tile_w_px": tw, "tile_h_px": th,
            "tiles_across": na, "tiles_down": nd,
            "total_tiles": na * nd,
        }

    # Build grid & composite
    full_tile = build_full_tile_grid(
        room_bgr, combined_mask, tile_bgr,
        rotation_angle=rotation,
        tile_w=tile_w_px, tile_h=tile_h_px,
        camera_tilt=camera_tilt,
    )
    result_bgr = composite_tile_on_surface(room_bgr, combined_mask, full_tile)

    # Encode to PNG bytes
    _, png = cv2.imencode(".png", result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return png.tobytes(), dim_info
