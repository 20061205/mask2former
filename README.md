pip install git+https://github.com/facebookresearch/detectron2.git
countertops/             # Countertop-specific module (dual-model)
# Tile Visualizer — Mask2Former

Realistic tile visualization on room surfaces (floor, wall, countertop, etc.) using Mask2Former semantic segmentation.

---

## Installation

```bash
# Create virtual environment
python -m venv .mask

# Activate (Windows)
.mask\Scripts\activate

# Install dependencies
pip install -r requirements-lock.txt

# Install detectron2 from source (required for countertop module)
pip install git+https://github.com/facebookresearch/detectron2.git
```

---

## Quick Start


### Floor Tiling Example
```bash
python floor/floor_tile.py --room rooms/room1.jpg --tile tiles/tile1.jpg
```

### Countertop Mask Generation
```bash
python -m countertops.generate --image rooms/kitchens/kit1.jpg --preview
```

### Apply Tile to Countertop
```bash
python -m countertops.apply_tile --room rooms/kitchens/kit4.jpg --tile tiles/tile1.jpg --live --preview
```

---

## Project Structure

```
floor/
  floor_tile.py          # Main CLI — floor, wall, and other surface tiling
  tile_engine.py         # Tile grid, lighting transfer, compositing
  surfaces.py            # Surface mask extraction & cleaning
  config.py              # ADE20K label IDs, tile defaults

countertops/           # Countertop-specific module (dual-model)
  generate.py          # Mask generation CLI
  apply_tile.py        # Tile application CLI
  mask_generator.py    # Detectron2 + Mask2Former AND-gate masks
  tile_applicator.py   # Countertop tile rendering engine
  config.py            # Countertop-specific config

rooms/                   # Input room images
tiles/                   # Input tile images
outputs/                 # Generated results
model.py                 # Mask2Former model loading & segmentation

```

---

## Floor Tiles

### With real-world dimensions (exact tile count)
Provide floor area (in **feet**) and tile size (in **inches**):
```bash
python -m floor.floor_tile --room path/to/room.jpg --tile path/to/tile.jpg --floor-width 10 --floor-length 12 --tile-width 24 --tile-height 12
```

This computes exactly how many tiles fit (e.g. 10′×12′ floor with 24″×24″ tiles → 5×6 = 30 tiles) and sizes each tile in pixels to match.


---

---


## Countertop Tiles

Uses a multi-model approach for accurate countertop mask generation:

- **Mask2Former ADE20K**: Semantic segmentation for room surfaces.
- **MaskRCNN**: Custom model for countertop detection.
- **SAM (Segment Anything Model)**: Added for enhanced mask precision and generalization.

First MaskRCNN (trained model) detect the countertop regions. Then that pixel possions send to SAM and it exacty extract the countertop masks. The mask of SAM model and mask of Mask2Former are combined (AND-gate logic) to produce robust countertop masks.


### Generate countertop mask


To use the SAM model, ensure the SAM weights file (e.g. `sam_vit_h_4b8939.pth`) is present in the `countertops/` directory. The script will automatically use it if available.


### Apply tile to countertop

```bash

# Generate mask on-the-fly and apply (uses all available models)
python -m countertops.apply_tile --room path/to/kitchen.jpg --tile path/to/tile.jpg --live --preview
```

```


### Customize countertop tile settings

```bash
python -m countertops.apply_tile --room rooms/kitchens/kit4.jpg --tile "tiles/granite tile.jpg" --live --tile-size 400 --grout 3 --rotation 5
```

---

### SAM Model Setup

- Download the official SAM weights (e.g. `sam_vit_h_4b8939.pth`) and place them in the `countertops/` directory.
- The scripts will automatically detect and use SAM for mask generation if the weights are available.
- For more information, see [SAM GitHub](https://github.com/facebookresearch/segment-anything).

---

## Available Surfaces

```bash
python floor/floor_tile.py --list-surfaces
```

| Surface     | ADE20K ID |
|-------------|-----------|
| floor       | 3         |
| wall        | 0         |
| countertop  | 70, 45, 73|
| stairway    | 53        |
| cabinet     | 10        |
| ceiling     | 5         |
| table       | 15        |
| shelf       | 24        |

---

## CLI Reference — `floor_tile.py`

| Argument           | Description                              | Default     |
|--------------------|------------------------------------------|-------------|
| `--room`           | Path to room image                       | *(required)*|
| `--tile`           | Path to tile image                       | *(required)*|
| `--surfaces`       | Surface(s) to tile                       | `floor`     |
| `--rotation`       | Tile rotation angle (degrees)            | `10.0`      |
| `--floor-width`    | Real surface width in feet               | —           |
| `--floor-length`   | Real surface length in feet              | —           |
| `--tile-width`     | Tile width in inches                     | —           |
| `--tile-height`    | Tile height in inches                    | —           |
| `--outdir`         | Output directory                         | `outputs`   |
| `--debug`          | Save intermediate debug images           | `false`     |
| `--list-surfaces`  | Print available surfaces and exit        | —           |

> All four dimension args (`--floor-width`, `--floor-length`, `--tile-width`, `--tile-height`) must be provided together.

---

## Output

Results are saved to the `outputs/` directory:

- `tile_applied_floor.png` — lossless PNG
- `tile_applied_floor.jpg` — high-quality JPEG (q97)
- `segmentation_preview.png` — mask visualization
- `mask_floor.png` — extracted surface mask
