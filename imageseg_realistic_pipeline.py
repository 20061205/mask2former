import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


MODEL_NAME = "facebook/mask2former-swin-base-ade-semantic"
FLOOR_ID = 3


def load_model(model_name: str = MODEL_NAME):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def get_floor_mask(image_rgb: Image.Image, processor, model, device):
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image_rgb.size[::-1]],
    )[0]

    segmentation_map = result.cpu().numpy()
    floor_mask = (segmentation_map == FLOOR_ID).astype(np.uint8) * 255
    return floor_mask


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


def apply_realistic_tile(original_img, mask, tile_img, rotation_angle=20):
    h, w = original_img.shape[:2]
    mask = (mask > 0).astype(np.uint8) * 255

    dst_pts = get_floor_corners(mask)
    if dst_pts is None:
        return original_img

    original_flat_w = 1200
    original_flat_h = 1200
    expanded_dim = int(np.ceil(np.sqrt(original_flat_w**2 + original_flat_h**2)))

    tile_pattern = create_tile_pattern(expanded_dim, expanded_dim, tile_img)

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

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    lighting = cv2.GaussianBlur(gray, (31, 31), 0) / 255.0
    lighting = 0.6 + 0.4 * lighting

    realistic_tiles = warped_tiles.astype(np.float32) * lighting[:, :, None]
    realistic_tiles = np.clip(realistic_tiles, 0, 255).astype(np.uint8)

    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_norm = mask_blur / 255.0
    mask_3 = np.stack([mask_norm] * 3, axis=-1)

    result = original_img * (1 - mask_3) + realistic_tiles * mask_3
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Mask2Former realistic floor tile visualization")
    parser.add_argument("--room", required=True, help="Path to room image")
    parser.add_argument("--tile", required=True, help="Path to tile image")
    parser.add_argument("--rotation", type=float, default=20.0, help="Tile rotation angle")
    parser.add_argument("--out", default="outputs/tile_applied_realistic.png", help="Output image path")
    args = parser.parse_args()

    room_pil = Image.open(args.room).convert("RGB")
    tile_pil = Image.open(args.tile).convert("RGB")

    processor, model, device = load_model()
    floor_mask = get_floor_mask(room_pil, processor, model, device)

    room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)
    tile_bgr = cv2.cvtColor(np.array(tile_pil), cv2.COLOR_RGB2BGR)

    result_bgr = apply_realistic_tile(room_bgr, floor_mask, tile_bgr, rotation_angle=args.rotation)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result_bgr)

    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
