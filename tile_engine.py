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

    # 3. Transfer the room's lighting / shadows onto the tile
    full_tile = transfer_room_lighting(room_bgr, full_tile)

    return full_tile


def transfer_room_lighting(room_bgr: np.ndarray,
                           tile_bgr: np.ndarray) -> np.ndarray:
    """
    Transfer the original room image's lighting, shadows, and
    reflections onto the tile grid using LAB-space luminance matching.

    Steps
    -----
    1. Convert both to LAB colour space.
    2. Extract room's L channel (brightness map).
    3. Build a *lighting ratio* from the room's low-freq brightness.
    4. Multiply tile's L channel by the ratio → preserves original
       light gradients, shadows, and specular highlights.
    5. Blend a high-frequency detail layer from the room to add
       subtle surface reflections and micro-shadows.
    """
    h, w = room_bgr.shape[:2]
    tile_bgr = tile_bgr[:h, :w]   # safety crop

    room_lab = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tile_lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    room_L = room_lab[:, :, 0]
    tile_L = tile_lab[:, :, 0]

    # Low-frequency lighting envelope from the room
    room_L_blur = cv2.GaussianBlur(room_L, (61, 61), 0)
    tile_L_blur = cv2.GaussianBlur(tile_L, (61, 61), 0)

    # Lighting ratio: how much brighter/darker is each room pixel vs average
    tile_L_blur[tile_L_blur < 1] = 1
    ratio = room_L_blur / tile_L_blur
    ratio = np.clip(ratio, 0.3, 2.5)

    # Apply lighting ratio to tile
    tile_L_lit = tile_L * ratio

    # High-frequency detail from room (micro-shadows, reflections)
    room_detail = room_L - room_L_blur
    detail_strength = 0.35
    tile_L_lit = tile_L_lit + room_detail * detail_strength

    tile_L_lit = np.clip(tile_L_lit, 0, 255)
    tile_lab[:, :, 0] = tile_L_lit

    result = cv2.cvtColor(tile_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


# ── Compositing ──────────────────────────────────────────────────────

def composite_tile_on_surface(room_bgr: np.ndarray,
                              surface_mask: np.ndarray,
                              full_tile_image: np.ndarray,
                              feather_radius: int = 11) -> np.ndarray:
    """
    Realistic composite:
      1. Feathered alpha along mask edges for smooth transition.
      2. Shadow darkening near edges to preserve 3D depth / corners.
      3. Specular highlight preservation from the original room.
    """
    h, w = room_bgr.shape[:2]

    # ── 1. Feathered alpha mask ──────────────────────────────────────
    #  Use distance transform for smooth, natural-looking falloff
    binary = (surface_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Feather zone: pixels within feather_radius of the edge blend
    feather = np.clip(dist / max(feather_radius, 1), 0.0, 1.0)
    # Slight Gaussian smooth on top for extra softness
    feather = cv2.GaussianBlur(feather.astype(np.float32), (5, 5), 0)

    # ── 2. Edge shadow / 3D depth preservation ───────────────────────
    #  Darken tiles near mask edges → simulates the shadow at corners
    #  where the countertop meets the vertical face.
    shadow_width = max(feather_radius * 2, 15)
    shadow_zone = np.clip(dist / shadow_width, 0.0, 1.0)
    # Shadow factor: darkest at the very edge, normal further in
    shadow_factor = 0.65 + 0.35 * shadow_zone   # 0.65 at edge → 1.0 inside
    shadow_factor3 = np.stack([shadow_factor] * 3, axis=-1)

    tile_shadowed = np.clip(
        full_tile_image.astype(np.float32) * shadow_factor3, 0, 255
    ).astype(np.uint8)

    # ── 3. Specular / highlight preservation ─────────────────────────
    #  Keep bright specular spots from the original room so reflections
    #  look natural (e.g. light glinting off a polished surface).
    room_gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tile_gray = cv2.cvtColor(tile_shadowed, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Detect strong highlights in the original room
    highlight_thresh = np.percentile(room_gray[binary > 0], 92) if binary.sum() > 0 else 230
    highlight_mask = (room_gray > highlight_thresh).astype(np.float32)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (11, 11), 0)

    # Blend highlights: where original was very bright, let some of that through
    highlight_strength = 0.3
    blended = tile_shadowed.astype(np.float32) * (1.0 - highlight_mask[:, :, None] * highlight_strength) + \
              room_bgr.astype(np.float32) * (highlight_mask[:, :, None] * highlight_strength)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # ── 4. Final composite ───────────────────────────────────────────
    alpha_3 = np.stack([feather] * 3, axis=-1)
    result = blended.astype(np.float32) * alpha_3 + room_bgr.astype(np.float32) * (1.0 - alpha_3)

    return np.clip(result, 0, 255).astype(np.uint8)
