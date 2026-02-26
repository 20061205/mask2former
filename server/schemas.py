from pydantic import BaseModel, Field


class TileRequest(BaseModel):
    surfaces: list[str] = Field(default=["floor"], description="Surfaces to tile")
    rotation: float = Field(default=0.0, description="Tile rotation angle in degrees")

    # Real-world dimensions (all optional, but must be all-or-none)
    floor_width: float | None = Field(None, description="Floor width in feet")
    floor_length: float | None = Field(None, description="Floor length in feet")
    tile_width: float | None = Field(None, description="Tile width in inches")
    tile_height: float | None = Field(None, description="Tile height in inches")


class TileInfo(BaseModel):
    tile_w_px: int
    tile_h_px: int
    tiles_across: int
    tiles_down: int
    total_tiles: int


class TileResponse(BaseModel):
    message: str
    surfaces: list[str]
    dimensions: TileInfo | None = None
    output_url: str
