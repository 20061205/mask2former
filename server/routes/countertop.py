"""Countertop tile routes (Detectron2 -> SAM)."""

import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from ..schemas import TileResponse
from ..services import decode_images, save_inputs

router = APIRouter(prefix="/api/countertop", tags=["countertop"])

# -- Lazy-loaded countertop models ----------------------------------------
_predictor = None
_sam_predictor = None


def _get_countertop_models():
    global _predictor, _sam_predictor
    if _predictor is None:
        from countertops.mask_generator import build_predictor, load_sam
        _predictor = build_predictor()
        _sam_predictor = load_sam()
    return _predictor, _sam_predictor


@router.post("/apply", response_class=Response)
async def apply_countertop_tile(
    room: UploadFile = File(..., description="Kitchen image"),
    tile: UploadFile = File(..., description="Tile / slab image"),
    tile_size: int = Form(600, description="Tile size in pixels"),
    grout: int = Form(2),
    rotation: float = Form(10.0),
):
    """Generate countertop mask on-the-fly and apply tile. Returns PNG."""
    from countertops.mask_generator import generate_sam_mask
    from countertops.tile_applicator import apply_tile

    room_bytes = await room.read()
    tile_bytes = await tile.read()
    save_inputs(room_bytes, tile_bytes, prefix="countertop")
    room_bgr, tile_bgr = decode_images(room_bytes, tile_bytes)
    predictor, sam_pred = _get_countertop_models()

    result = generate_sam_mask(room_bgr, predictor, sam_pred)
    mask = result["sam_mask"]

    out = apply_tile(room_bgr, mask, tile_bgr,
                     tile_size=tile_size, grout=grout, rotation=rotation)

    _, png = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return Response(content=png.tobytes(), media_type="image/png")


@router.post("/apply-json", response_model=TileResponse)
async def apply_countertop_tile_json(
    room: UploadFile = File(...),
    tile: UploadFile = File(...),
    tile_size: int = Form(600),
    grout: int = Form(2),
    rotation: float = Form(10.0),
):
    """Apply countertop tile and return JSON metadata."""
    from countertops.mask_generator import generate_sam_mask
    from countertops.tile_applicator import apply_tile

    room_bytes = await room.read()
    tile_bytes = await tile.read()
    save_inputs(room_bytes, tile_bytes, prefix="countertop")
    room_bgr, tile_bgr = decode_images(room_bytes, tile_bytes)
    predictor, sam_pred = _get_countertop_models()

    result = generate_sam_mask(room_bgr, predictor, sam_pred)
    mask = result["sam_mask"]

    out = apply_tile(room_bgr, mask, tile_bgr,
                     tile_size=tile_size, grout=grout, rotation=rotation)

    _, png = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    filename = f"countertop_{uuid.uuid4().hex[:8]}.png"
    (out_dir / filename).write_bytes(png.tobytes())

    return TileResponse(
        message="Countertop tile applied successfully",
        surfaces=["countertop"],
        dimensions=None,
        output_url=f"/outputs/{filename}",
    )


@router.post("/mask", response_class=Response)
async def generate_countertop_mask(
    room: UploadFile = File(..., description="Kitchen image"),
):
    """Generate and return the countertop mask as PNG."""
    from countertops.mask_generator import generate_sam_mask

    room_bytes = await room.read()
    arr = np.frombuffer(room_bytes, np.uint8)
    room_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    predictor, sam_pred = _get_countertop_models()

    result = generate_sam_mask(room_bgr, predictor, sam_pred)
    mask = result["sam_mask"]

    _, png = cv2.imencode(".png", mask)
    return Response(content=png.tobytes(), media_type="image/png")
