"""Send a tile request to the server and save the result."""

import requests
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────
SERVER = "http://localhost:8000"
ROOM_IMG = r"E:\tile viz\mask2former\rooms\kitchens\kit4.jpg"
TILE_IMG = r"E:\tile viz\mask2former\tiles\tile1.jpg"
OUTPUT_DIR = Path("outputs")

# ── Request ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)

r = requests.post(
    f"{SERVER}/api/countertop/apply",
    files={
        "room": open(ROOM_IMG, "rb"),
        "tile": open(TILE_IMG, "rb"),
    },
    data={
        "tile_size": "600",
        "grout": "2",
        "rotation": "10",
    },
)

print(f"Status: {r.status_code}")

if r.status_code == 200:
    out_path = OUTPUT_DIR / "countertop_result.png"
    out_path.write_bytes(r.content)
    print(f"Saved: {out_path.resolve()}")
else:
    print(f"Error: {r.text}")
