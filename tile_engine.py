"""
Tile pattern creation, rotation, perspective grid, and compositing.
"""

import cv2
import numpy as np

from config import (
    DEFAULT_TILE_SIZE,
    DEFAULT_GROUT_WIDTH,
    TILE_GRID_SCALE,
)


# ── Tile pattern generation ──────────────────────────────────────────

def create_tile_pattern(width: int, height: int, tile_img: np.ndarray,
                        tile_size: int = DEFAULT_TILE_SIZE,
                        grout: int = DEFAULT_GROUT_WIDTH) -> np.ndarray:
    """
    Create a flat tile grid with grout lines and slight per-tile colour
    variation for realism.
    """
    grout_color = 220
    pattern = np.ones((height, width, 3), dtype=np.uint8) * grout_color
    tile_resized = cv2.resize(tile_img, (tile_size - grout, tile_size - grout))

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)

            y_end = min(y + tile_size - grout, height)
            x_end = min(x + tile_size - grout, width)
            pattern[y:y_end, x:x_end] = tile_var[:y_end - y, :x_end - x]

    return pattern


# ── Full-image perspective tile grid ─────────────────────────────────

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_surface_corners(mask: np.ndarray):
    """Find the 4 corner points of a surface mask via contour approx."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
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

    return np.float32(order_points_clockwise(pts))


def build_full_tile_grid(room_bgr: np.ndarray,
                         surface_mask: np.ndarray,
                         tile_bgr: np.ndarray,
                         rotation_angle: float = 10.0,
                         tile_size: int = DEFAULT_TILE_SIZE,
                         grout: int = DEFAULT_GROUT_WIDTH,
                         scale: int = TILE_GRID_SCALE) -> np.ndarray:
    """
    Create a perspective-angled, rotated tile grid that fills the
    **entire image** (same dimensions as room_bgr).

    Steps
    -----
    1. Build an oversized flat tile pattern.
    2. Rotate it.
    3. Use the surface corners to compute a perspective transform,
       then use the INVERSE mapping so every output pixel is filled.
    4. Apply room lighting for realism.
    """
    h, w = room_bgr.shape[:2]

    # 1. Oversized rotated tile pattern
    big_w, big_h = w * scale, h * scale
    tile_pattern = create_tile_pattern(big_w, big_h, tile_bgr,
                                       tile_size=tile_size, grout=grout)

    center_rot = (big_w // 2, big_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center_rot, rotation_angle, 1.0)
    tile_pattern = cv2.warpAffine(tile_pattern, rot_mat, (big_w, big_h),
                                  borderMode=cv2.BORDER_REFLECT)

    # 2. Perspective mapping via surface corners
    dst_pts = get_surface_corners(surface_mask)
    use_perspective = False

    if dst_pts is not None:
        ox = (big_w - w) // 2
        oy = (big_h - h) // 2
        src_quad = np.float32([
            [ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]
        ])
        M = cv2.getPerspectiveTransform(src_quad, dst_pts)

        # Try to invert safely
        try:
            M_inv = np.linalg.inv(M)
            test_pt = M_inv @ np.array([w / 2, h / 2, 1.0])
            if abs(test_pt[2]) > 1e-6 and np.all(np.isfinite(test_pt)):
                use_perspective = True
        except np.linalg.LinAlgError:
            pass

    if use_perspective:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        ones = np.ones_like(xs)
        coords = np.stack([xs, ys, ones], axis=-1)
        src_coords = coords @ M_inv.T
        # Avoid division by zero
        denom = src_coords[:, :, 2].copy()
        denom[np.abs(denom) < 1e-8] = 1e-8
        src_coords[:, :, 0] /= denom
        src_coords[:, :, 1] /= denom
        map_x = np.clip(src_coords[:, :, 0], 0, big_w - 1).astype(np.float32)
        map_y = np.clip(src_coords[:, :, 1], 0, big_h - 1).astype(np.float32)

        full_tile = cv2.remap(tile_pattern, map_x, map_y,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        # Fallback: crop centre of rotated pattern (no perspective)
        ox = (big_w - w) // 2
        oy = (big_h - h) // 2
        full_tile = tile_pattern[oy:oy + h, ox:ox + w]

    # 3. Apply room lighting
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lighting = cv2.GaussianBlur(gray, (31, 31), 0) / 255.0
    lighting = 0.6 + 0.4 * lighting
    full_tile = np.clip(full_tile.astype(np.float32) * lighting[:, :, None],
                        0, 255).astype(np.uint8)

    return full_tile


# ── Compositing ──────────────────────────────────────────────────────

def composite_tile_on_surface(room_bgr: np.ndarray,
                              surface_mask: np.ndarray,
                              full_tile_image: np.ndarray,
                              feather_radius: int = 11) -> np.ndarray:
    """
    Composite: tile grid is the base layer, then the non-surface parts
    of the room image are pasted on top with feathered edges.
    """
    non_surface = 255 - surface_mask
    non_surface_smooth = cv2.GaussianBlur(non_surface, (feather_radius, feather_radius), 0)
    alpha = non_surface_smooth.astype(np.float32) / 255.0
    alpha_3 = np.stack([alpha] * 3, axis=-1)

    result = full_tile_image * (1.0 - alpha_3) + room_bgr * alpha_3
    return np.clip(result, 0, 255).astype(np.uint8)
