import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


MODEL_NAME = "facebook/mask2former-swin-base-ade-semantic"
WALL_ID = 0
FLOOR_ID = 3


def load_model(model_name: str = MODEL_NAME):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def segment_room(image: Image.Image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
    )[0]

    segmentation_map = result.cpu().numpy()
    wall_mask = (segmentation_map == WALL_ID).astype(np.uint8) * 255
    floor_mask = (segmentation_map == FLOOR_ID).astype(np.uint8) * 255
    return segmentation_map, wall_mask, floor_mask


def tile_to_room_size(room_image: Image.Image, tile_image: Image.Image):
    tile_np = np.array(tile_image)
    room_h, room_w = room_image.size[1], room_image.size[0]

    tile_h, tile_w = tile_np.shape[:2]
    repeat_y = room_h // tile_h + 1
    repeat_x = room_w // tile_w + 1

    big_tile = np.tile(tile_np, (repeat_y, repeat_x, 1))
    return big_tile[:room_h, :room_w]


def apply_tile(room_image: Image.Image, floor_mask: np.ndarray, tile_image: Image.Image):
    room_np = np.array(room_image)
    big_tile = tile_to_room_size(room_image, tile_image)
    floor_mask_bool = floor_mask.astype(bool)

    result_img = room_np.copy()
    result_img[floor_mask_bool] = big_tile[floor_mask_bool]
    return result_img


def save_preview(room_image, wall_mask, floor_mask, output_path: Path):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(room_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(wall_mask, cmap="gray")
    plt.title("Wall Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(floor_mask, cmap="gray")
    plt.title("Floor Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Mask2Former basic room segmentation and floor tile overlay")
    parser.add_argument("--room", required=True, help="Path to room image")
    parser.add_argument("--tile", required=True, help="Path to tile image")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    room_image = Image.open(args.room).convert("RGB")
    tile_image = Image.open(args.tile).convert("RGB")

    processor, model, device = load_model()
    _, wall_mask, floor_mask = segment_room(room_image, processor, model, device)

    result_img = apply_tile(room_image, floor_mask, tile_image)

    Image.fromarray(wall_mask).save(outdir / "wall_mask.png")
    Image.fromarray(floor_mask).save(outdir / "floor_mask.png")
    Image.fromarray(result_img).save(outdir / "tile_applied_basic.png")
    save_preview(room_image, wall_mask, floor_mask, outdir / "segmentation_preview.png")

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
