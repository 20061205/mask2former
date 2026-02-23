"""
Realistic tile application onto countertop mask regions.

Pipeline
--------
1. Build an oversized tile grid with grout lines and per-tile variation.
2. Rotate and perspective-warp to match the countertop geometry.
3. Transfer the room's original lighting (LAB luminance matching).
4. Preserve shadows near mask edges for 3D depth.
5. Keep specular highlights from the original surface.
6. Feathered alpha blend for seamless edge transitions.
"""

import cv2
import numpy as np
from pathlib import Path

from .config import (
    TILE_SIZE,
    GROUT_WIDTH,
    ROTATION_ANGLE,
    TILE_GRID_SCALE,
    FEATHER_RADIUS,
    SHADOW_BASE,
    HIGHLIGHT_STRENGTH,
    DETAIL_STRENGTH,
)


# ─────────────────────────────────────────────────────────────────────
# 1. TILE PATTERN GENERATION
# ─────────────────────────────────────────────────────────────────────

def create_tile_pattern(
    width: int,
    height: int,
    tile_img: np.ndarray,
    tile_size: int = TILE_SIZE,
    grout: int = GROUT_WIDTH,
) -> np.ndarray:
    """
    Create a flat tile grid with grout lines and slight per-tile
    colour variation for realism.
    """
    grout_color = 200
    pattern = np.ones((height, width, 3), dtype=np.uint8) * grout_color
    cell = tile_size - grout
    tile_resized = cv2.resize(tile_img, (cell, cell), interpolation=cv2.INTER_AREA)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)

            y_end = min(y + cell, height)
            x_end = min(x + cell, width)
            th = y_end - y
            tw = x_end - x
            pattern[y:y_end, x:x_end] = tile_var[:th, :tw]

    return pattern


# ─────────────────────────────────────────────────────────────────────
# 2. PERSPECTIVE / GEOMETRY
# ─────────────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points clockwise: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _get_surface_corners(mask: np.ndarray):
    """Find the 4 corner points of the mask region."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    eps = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, eps, True)

    if len(approx) < 4:
        return None

    pts = approx.reshape(-1, 2)
    if len(pts) > 4:
        rect = cv2.minAreaRect(hull)
        pts = cv2.boxPoints(rect)

    return np.float32(_order_points(pts))


def build_tile_grid(
    room_bgr: np.ndarray,
    mask: np.ndarray,
    tile_bgr: np.ndarray,
    tile_size: int = TILE_SIZE,
    grout: int = GROUT_WIDTH,
    rotation: float = ROTATION_ANGLE,
    scale: int = TILE_GRID_SCALE,
) -> np.ndarray:
    """
    Create a rotated tile grid that fills the entire image.

    For countertops (irregular shapes — L-shaped, multi-piece) we skip
    perspective warping and instead use a flat rotated grid.  The room's
    lighting transfer handles the realism afterwards.
    """
    h, w = room_bgr.shape[:2]
    big_w, big_h = w * scale, h * scale

    # Oversized flat tile pattern
    pattern = create_tile_pattern(big_w, big_h, tile_bgr,
                                  tile_size=tile_size, grout=grout)

    # Rotate around centre
    if abs(rotation) > 0.1:
        center = (big_w // 2, big_h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
        pattern = cv2.warpAffine(pattern, rot_mat, (big_w, big_h),
                                 borderMode=cv2.BORDER_REFLECT)

    # Crop centre to room dimensions (no perspective — keeps all edges)
    ox = (big_w - w) // 2
    oy = (big_h - h) // 2
    full_tile = pattern[oy:oy + h, ox:ox + w]

    return full_tile


# ─────────────────────────────────────────────────────────────────────
# 3. LIGHTING TRANSFER
# ─────────────────────────────────────────────────────────────────────

def transfer_lighting(
    room_bgr: np.ndarray,
    tile_bgr: np.ndarray,
    detail_strength: float = DETAIL_STRENGTH,
) -> np.ndarray:
    """
    Transfer the room's lighting onto the tile using LAB luminance.

    - Low-frequency ratio: preserves broad light gradients and shadows.
    - High-frequency detail: adds micro-shadows and surface reflections.
    """
    h, w = room_bgr.shape[:2]
    tile_bgr = tile_bgr[:h, :w]

    room_lab = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tile_lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    room_L = room_lab[:, :, 0]
    tile_L = tile_lab[:, :, 0]

    # Low-freq lighting envelope
    room_L_blur = cv2.GaussianBlur(room_L, (61, 61), 0)
    tile_L_blur = cv2.GaussianBlur(tile_L, (61, 61), 0)
    tile_L_blur[tile_L_blur < 1] = 1

    ratio = np.clip(room_L_blur / tile_L_blur, 0.3, 2.5)
    tile_L_lit = tile_L * ratio

    # High-freq detail (micro-shadows, reflections)
    room_detail = room_L - room_L_blur
    tile_L_lit = tile_L_lit + room_detail * detail_strength

    tile_lab[:, :, 0] = np.clip(tile_L_lit, 0, 255)
    return cv2.cvtColor(tile_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────
# 4. COMPOSITING  (feathered blend + edge shadow + highlights)
# ─────────────────────────────────────────────────────────────────────

def composite(
    room_bgr: np.ndarray,
    mask: np.ndarray,
    tile_lit: np.ndarray,
    feather_radius: int = FEATHER_RADIUS,
    shadow_base: float = SHADOW_BASE,
    highlight_strength: float = HIGHLIGHT_STRENGTH,
) -> np.ndarray:
    """
    Blend the lit tile onto the room image with:
      - Very narrow feather at mask edges (anti-aliasing, not fade-out)
      - Thin edge shadow for 3D depth at countertop borders
      - Specular highlight preservation
    """
    h, w = room_bgr.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    # ── Alpha mask: blur the binary mask slightly for anti-aliased edges
    #    (NOT distance-based fade — tiles must reach all corners/edges)
    alpha = cv2.GaussianBlur(binary.astype(np.float32) * 255,
                             (feather_radius | 1, feather_radius | 1), 0) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    # Ensure interior is fully opaque (only the 1-2px edge is softened)
    alpha[binary > 0] = np.maximum(alpha[binary > 0], 0.95)

    # ── Edge shadow (3D depth) — thin band only ──────────────────────
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    shadow_width = 8  # very thin shadow band at edges
    shadow_zone = np.clip(dist / shadow_width, 0.0, 1.0)
    shadow_factor = shadow_base + (1.0 - shadow_base) * shadow_zone
    shadow_3ch = np.stack([shadow_factor] * 3, axis=-1)

    tile_shadowed = np.clip(
        tile_lit.astype(np.float32) * shadow_3ch, 0, 255
    ).astype(np.uint8)

    # ── Specular highlights from original ────────────────────────────
    room_gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if binary.sum() > 0:
        thresh = np.percentile(room_gray[binary > 0], 95)
    else:
        thresh = 240.0
    hl_mask = (room_gray > thresh).astype(np.float32)
    hl_mask = cv2.GaussianBlur(hl_mask, (7, 7), 0)

    blended = (
        tile_shadowed.astype(np.float32) * (1.0 - hl_mask[:, :, None] * highlight_strength)
        + room_bgr.astype(np.float32) * (hl_mask[:, :, None] * highlight_strength)
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # ── Final alpha composite ────────────────────────────────────────
    alpha_3 = np.stack([alpha] * 3, axis=-1)
    result = blended.astype(np.float32) * alpha_3 + room_bgr.astype(np.float32) * (1.0 - alpha_3)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────

def apply_tile(
    room_bgr: np.ndarray,
    mask: np.ndarray,
    tile_bgr: np.ndarray,
    tile_size: int = TILE_SIZE,
    grout: int = GROUT_WIDTH,
    rotation: float = ROTATION_ANGLE,
) -> np.ndarray:
    """
    Full pipeline: tile grid → lighting transfer → composite.

    Parameters
    ----------
    room_bgr  : original room image (BGR)
    mask      : binary countertop mask (0/255)
    tile_bgr  : tile texture image (BGR)
    tile_size : pixel size of each tile cell
    grout     : grout line width in pixels
    rotation  : tile rotation angle in degrees

    Returns
    -------
    result : room image with tile applied to mask region (BGR)
    """
    # Build perspective tile grid
    tile_grid = build_tile_grid(room_bgr, mask, tile_bgr,
                                tile_size=tile_size, grout=grout,
                                rotation=rotation)

    # Transfer room lighting onto tiles
    tile_lit = transfer_lighting(room_bgr, tile_grid)

    # Composite onto room
    result = composite(room_bgr, mask, tile_lit)

    return result


# ─────────────────────────────────────────────────────────────────────
# 6. PREVIEW
# ─────────────────────────────────────────────────────────────────────

def save_tile_preview(
    room_bgr: np.ndarray,
    mask: np.ndarray,
    result_bgr: np.ndarray,
    output_path: Path,
) -> None:
    """Save a 3-panel comparison: original | mask | tiled result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    axes[0].imshow(cv2.cvtColor(room_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Room")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Countertop Mask")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Tiled Countertop")
    axes[2].axis("off")

    plt.suptitle(output_path.stem, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Tile preview saved: {output_path}")
