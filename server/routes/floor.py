"""Floor & wall tile routes."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from ..schemas import TileResponse, TileInfo
from ..services import apply_tiles, list_surfaces, save_inputs

router = APIRouter(prefix="/api/floor", tags=["floor"])


@router.get("/surfaces")
def get_surfaces():
    """List all available tileable surfaces."""
    return list_surfaces()


@router.post("/apply", response_class=Response)
async def apply_floor_tile(
    room: UploadFile = File(..., description="Room image"),
    tile: UploadFile = File(..., description="Tile image"),
    surfaces: str = Form("floor", description="Comma-separated surface names"),
    rotation: float = Form(0.0),
    floor_width: float | None = Form(None, description="Surface width in feet"),
    floor_length: float | None = Form(None, description="Surface length in feet"),
    tile_width: float | None = Form(None, description="Tile width in inches"),
    tile_height: float | None = Form(None, description="Tile height in inches"),
):
    """Apply tiles to floor/wall. Returns PNG image."""
    room_bytes = await room.read()
    tile_bytes = await tile.read()
    save_inputs(room_bytes, tile_bytes, prefix="floor")
    surface_list = [s.strip() for s in surfaces.split(",")]

    result_png, _ = apply_tiles(
        room_bytes, tile_bytes, surface_list, rotation,
        floor_width, floor_length, tile_width, tile_height,
    )
    return Response(content=result_png, media_type="image/png")


@router.post("/apply-json", response_model=TileResponse)
async def apply_floor_tile_json(
    room: UploadFile = File(...),
    tile: UploadFile = File(...),
    surfaces: str = Form("floor"),
    rotation: float = Form(0.0),
    floor_width: float | None = Form(None),
    floor_length: float | None = Form(None),
    tile_width: float | None = Form(None),
    tile_height: float | None = Form(None),
):
    """Apply tiles and return JSON metadata + save result."""
    room_bytes = await room.read()
    tile_bytes = await tile.read()
    save_inputs(room_bytes, tile_bytes, prefix="floor")
    surface_list = [s.strip() for s in surfaces.split(",")]

    result_png, dim_info = apply_tiles(
        room_bytes, tile_bytes, surface_list, rotation,
        floor_width, floor_length, tile_width, tile_height,
    )

    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    filename = f"floor_{uuid.uuid4().hex[:8]}.png"
    (out_dir / filename).write_bytes(result_png)

    return TileResponse(
        message="Floor tiles applied successfully",
        surfaces=surface_list,
        dimensions=TileInfo(**dim_info) if dim_info else None,
        output_url=f"/outputs/{filename}",
    )
