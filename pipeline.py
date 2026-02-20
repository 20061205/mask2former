import argparse
from pathlib import Path

import cv2
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


def segment_room(image_rgb: Image.Image, processor, model, device):
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image_rgb.size[::-1]],
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


def apply_tile_basic(room_image: Image.Image, floor_mask: np.ndarray, tile_image: Image.Image):
    room_np = np.array(room_image)
    big_tile = tile_to_room_size(room_image, tile_image)
    floor_mask_bool = floor_mask.astype(bool)

    result_img = room_np.copy()
    result_img[floor_mask_bool] = big_tile[floor_mask_bool]
    return result_img


def clean_floor_mask(floor_mask, kernel_size=25, iterations=3):
    """Fill small gaps and expand floor mask using morphological operations."""
    mask = (floor_mask > 0).astype(np.uint8)
    
    # Close small holes (dilate then erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Dilate to expand floor coverage and connect nearby regions
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return (mask * 255).astype(np.uint8)


def save_segmentation_preview(room_image, wall_mask, floor_mask, output_path: Path):
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


def create_tile_pattern(width, height, tile_img, tile_size=120, grout=3):
    grout_color = 220
    pattern = np.ones((height, width, 3), dtype=np.uint8) * grout_color

    tile_resized = cv2.resize(tile_img, (tile_size - grout, tile_size - grout))

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)

            y_end = min(y + tile_size - grout, height)
            x_end = min(x + tile_size - grout, width)

            pattern[y:y_end, x:x_end] = tile_var[: y_end - y, : x_end - x]

    return pattern


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_floor_corners(mask):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)

    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) < 4:
        return None

    pts = approx.reshape(-1, 2)

    if len(pts) > 4:
        rect = cv2.minAreaRect(hull)
        pts = cv2.boxPoints(rect)

    pts = order_points_clockwise(pts)
    return np.float32(pts)


def apply_tile_realistic(room_bgr, floor_mask, tile_bgr, rotation_angle=20):
    h, w = room_bgr.shape[:2]
    mask = (floor_mask > 0).astype(np.uint8) * 255

    dst_pts = get_floor_corners(mask)
    if dst_pts is None:
        return room_bgr

    original_flat_w = 1200
    original_flat_h = 1200
    expanded_dim = int(np.ceil(np.sqrt(original_flat_w**2 + original_flat_h**2)))

    tile_pattern = create_tile_pattern(expanded_dim, expanded_dim, tile_bgr)

    offset_x = (expanded_dim - original_flat_w) / 2
    offset_y = (expanded_dim - original_flat_h) / 2

    src_pts = np.float32(
        [
            [offset_x, offset_y],
            [offset_x + original_flat_w, offset_y],
            [offset_x + original_flat_w, offset_y + original_flat_h],
            [offset_x, offset_y + original_flat_h],
        ]
    )

    center_x = np.mean(src_pts[:, 0])
    center_y = np.mean(src_pts[:, 1])
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)

    rotated_src_pts = cv2.transform(src_pts.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
    h_matrix = cv2.getPerspectiveTransform(rotated_src_pts, dst_pts)
    warped_tiles = cv2.warpPerspective(tile_pattern, h_matrix, (w, h))

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lighting = cv2.GaussianBlur(gray, (31, 31), 0) / 255.0
    lighting = 0.6 + 0.4 * lighting

    realistic_tiles = warped_tiles.astype(np.float32) * lighting[:, :, None]
    realistic_tiles = np.clip(realistic_tiles, 0, 255).astype(np.uint8)

    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_norm = mask_blur / 255.0
    mask_3 = np.stack([mask_norm] * 3, axis=-1)

    result = room_bgr * (1 - mask_3) + realistic_tiles * mask_3
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Single-file Mask2Former floor tile visualizer")
    parser.add_argument("--room", required=True, help="Path to room image")
    parser.add_argument("--tile", required=True, help="Path to tile image")
    parser.add_argument("--mode", choices=["basic", "realistic"], default="realistic", help="Visualization mode")
    parser.add_argument("--rotation", type=float, default=20.0, help="Rotation angle for realistic mode")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    room_image = Image.open(args.room).convert("RGB")
    tile_image = Image.open(args.tile).convert("RGB")

    processor, model, device = load_model()
    _, wall_mask, floor_mask = segment_room(room_image, processor, model, device)

    Image.fromarray(wall_mask).save(outdir / "wall_mask.png")
    Image.fromarray(floor_mask).save(outdir / "floor_mask.png")
    save_segmentation_preview(room_image, wall_mask, floor_mask, outdir / "segmentation_preview.png")

    if args.mode == "basic":
        result_rgb = apply_tile_basic(room_image, floor_mask, tile_image)
        Image.fromarray(result_rgb).save(outdir / "tile_applied_basic.png")
        print(f"Saved basic result to: {(outdir / 'tile_applied_basic.png').resolve()}")
    else:
        room_bgr = cv2.cvtColor(np.array(room_image), cv2.COLOR_RGB2BGR)
        tile_bgr = cv2.cvtColor(np.array(tile_image), cv2.COLOR_RGB2BGR)
        result_bgr = apply_tile_realistic(room_bgr, floor_mask, tile_bgr, rotation_angle=args.rotation)
        cv2.imwrite(str(outdir / "tile_applied_realistic.png"), result_bgr)
        print(f"Saved realistic result to: {(outdir / 'tile_applied_realistic.png').resolve()}")

    print(f"Saved masks/preview to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
