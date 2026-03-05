"""
Perspective-correct tile mapping using homography.

Creates physically realistic tile textures that follow floor perspective,
with proper vanishing-point convergence and distance-based scaling.

Modules
-------
- extract_floor_polygon(mask)   → 4-point floor quadrilateral
- create_tiled_texture(...)     → flat repeating tile canvas
- compute_homography(...)       → 3×3 homography matrix
- blend_with_lighting(...)      → lighting-aware masked composite

The public helpers ``build_full_tile_grid``, ``composite_tile_on_surface``,
and ``calculate_tile_pixel_size`` are kept for backward compatibility with
the CLI (``floor_tile.py``) and the FastAPI server.
"""

import cv2
import numpy as np

from floor.config import (
    DEFAULT_TILE_SIZE,
    DEFAULT_GROUT_WIDTH,
    TILE_GRID_SCALE,
    DEFAULT_CAMERA_TILT,
)


# =====================================================================
# 1.  Floor polygon extraction  (row-based mask analysis)
# =====================================================================

def extract_floor_polygon(mask: np.ndarray) -> np.ndarray | None:
    """
    Extract a **perspective-correct trapezoid** from the binary floor mask
    by directly measuring the mask's horizontal extent at the top and
    bottom of the floor region.

    Why row-based?
    --------------
    Contour-approximation (``approxPolyDP``) often oversimplifies
    irregular masks, and its ``minAreaRect`` fallback returns a
    *rectangle* — which kills the perspective effect entirely.

    Instead we scan the mask row-by-row:
    * **Top zone** – the furthest rows from the camera (narrower).
    * **Bottom zone** – the nearest rows to the camera (wider).
    * Left / right extremes in each zone give us a natural trapezoid
      whose width difference encodes the camera's perspective.

    Returns
    -------
    np.ndarray of shape (4, 2) float32 ordered TL, TR, BR, BL,
    or *None* if the mask is too small / empty.
    """
    binary = (mask > 0).astype(np.uint8)
    h_img, w_img = binary.shape[:2]

    # ── Rows that contain any mask pixel ─────────────────────────────
    row_any = np.any(binary > 0, axis=1)
    active_rows = np.where(row_any)[0]
    if len(active_rows) < 10:
        return None

    top_row = int(active_rows[0])
    bot_row = int(active_rows[-1])
    span = bot_row - top_row
    if span < 20:
        return None

    # ── Sample top & bottom 8 % bands ────────────────────────────────
    band = max(int(span * 0.08), 5)

    # Top band (far from camera)
    top_zone = binary[top_row : top_row + band, :]
    top_cols = np.where(np.any(top_zone > 0, axis=0))[0]
    if len(top_cols) < 2:
        return None

    # Bottom band (near camera)
    bot_start = max(bot_row - band, 0)
    bot_zone = binary[bot_start : bot_row + 1, :]
    bot_cols = np.where(np.any(bot_zone > 0, axis=0))[0]
    if len(bot_cols) < 2:
        return None

    # ── Build the four corners ───────────────────────────────────────
    # Use 3rd / 97th percentile to ignore tiny mask spurs
    tl_x = float(np.percentile(top_cols, 3))
    tr_x = float(np.percentile(top_cols, 97))
    bl_x = float(np.percentile(bot_cols, 3))
    br_x = float(np.percentile(bot_cols, 97))

    top_y = float(top_row + band // 2)
    bot_y = float(bot_row - band // 2)

    quad = np.float32([
        [tl_x, top_y],   # TL – far-left
        [tr_x, top_y],   # TR – far-right
        [br_x, bot_y],   # BR – near-right
        [bl_x, bot_y],   # BL – near-left
    ])

    # Sanity check
    if cv2.contourArea(quad) < 100:
        return None

    top_w = tr_x - tl_x
    bot_w = br_x - bl_x
    print(f"  Floor quad: top_w={top_w:.0f}  bot_w={bot_w:.0f}  "
          f"height={bot_y - top_y:.0f}  "
          f"perspective ratio={bot_w / max(top_w, 1):.2f}")
    return quad


# =====================================================================
# 1b. Camera tilt amplification
# =====================================================================

def _amplify_perspective(quad: np.ndarray, tilt: float) -> np.ndarray:
    """
    Simulate a higher / more tilted-down camera by narrowing the far
    (top) edge of the floor quadrilateral.

    In a real scene, tilting the camera upward makes the far end of
    the floor appear proportionally narrower relative to the near end.
    This function replicates that effect by pulling the top-left and
    top-right corners toward the top-edge midpoint.

    Parameters
    ----------
    quad : (4, 2) float32  –  TL, TR, BR, BL
    tilt : float
        1.0  = no change (use raw mask shape).
        1.3  = mild upward tilt (default) – noticeable but natural.
        1.5  = moderate – far tiles visibly smaller.
        2.0+ = dramatic bird's-eye perspective.

    Returns
    -------
    Adjusted (4, 2) float32 quad.
    """
    if tilt <= 1.0:
        return quad

    tl, tr, br, bl = quad
    top_mid = (tl + tr) / 2.0

    # Pull top corners toward the midpoint → narrower far edge
    shrink = 1.0 / tilt
    new_tl = top_mid + (tl - top_mid) * shrink
    new_tr = top_mid + (tr - top_mid) * shrink

    return np.float32([new_tl, new_tr, br, bl])


# =====================================================================
# 2.  Tiled texture canvas
# =====================================================================

def create_tiled_texture(
    tile_img: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    grout: int = DEFAULT_GROUT_WIDTH,
    tile_w: int | None = None,
    tile_h: int | None = None,
) -> np.ndarray:
    """
    Create a large flat repeating tile canvas with grout lines and
    subtle per-tile colour variation for realism.

    Parameters
    ----------
    tile_img  : BGR tile image
    canvas_w  : output canvas width  (pixels)
    canvas_h  : output canvas height (pixels)
    tile_size : base square tile size; overridden by *tile_w*/*tile_h*
    grout     : grout-line width in pixels
    tile_w    : explicit tile width  (overrides *tile_size*)
    tile_h    : explicit tile height (overrides *tile_size*)

    Returns
    -------
    BGR canvas of shape ``(canvas_h, canvas_w, 3)``.
    """
    tw = tile_w if tile_w is not None else tile_size
    th = tile_h if tile_h is not None else tile_size

    grout_color = 220
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * grout_color

    # High-quality resize
    target_w = max(tw - grout, 1)
    target_h = max(th - grout, 1)
    src_h, src_w = tile_img.shape[:2]
    interp = (cv2.INTER_AREA
              if (target_w < src_w or target_h < src_h)
              else cv2.INTER_LANCZOS4)
    tile_resized = cv2.resize(tile_img, (target_w, target_h),
                              interpolation=interp)

    for y in range(0, canvas_h, th):
        for x in range(0, canvas_w, tw):
            variation = 0.97 + 0.06 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)

            y_end = min(y + th - grout, canvas_h)
            x_end = min(x + tw - grout, canvas_w)
            canvas[y:y_end, x:x_end] = tile_var[:y_end - y, :x_end - x]

    return canvas


# =====================================================================
# 3.  Homography computation & perspective warp
# =====================================================================

def compute_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray | None:
    """
    Compute the 3×3 homography that maps *src_pts* → *dst_pts*.

    Parameters
    ----------
    src_pts : (4, 2) float32 – corners of the flat tile rectangle
              ordered  TL, TR, BR, BL.
    dst_pts : (4, 2) float32 – floor quadrilateral in image space,
              same ordering.

    Returns
    -------
    3×3 homography matrix, or *None* on failure.
    """
    H, status = cv2.findHomography(src_pts, dst_pts)
    if H is None or not np.all(np.isfinite(H)):
        return None
    return H


def warp_tile_to_floor(
    tile_texture: np.ndarray,
    homography: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """
    Warp the flat tile texture into image space via ``cv2.warpPerspective``.

    Because the homography maps *source texture coords → image coords*,
    tiles that lie further from the camera are automatically compressed,
    producing natural vanishing-point convergence.

    Parameters
    ----------
    tile_texture : BGR tile canvas
    homography   : 3×3 matrix from :func:`compute_homography`
    output_size  : ``(width, height)`` of the room image

    Returns
    -------
    Warped tile image of size ``(height, width, 3)``.
    """
    w, h = output_size
    return cv2.warpPerspective(
        tile_texture, homography, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _flat_floor_dimensions(floor_quad: np.ndarray) -> tuple[float, float]:
    """
    Estimate the "bird's-eye-view" (overhead) width and height of the
    floor from its perspective quadrilateral.

    Uses the **geometric mean** of the top and bottom edge widths as
    the flat width.  This is critical for perspective strength:
    * If flat_w == bottom_w  → tiles at the near edge are 1:1 (no
      expansion), so the scale difference between near and far is
      limited to the compression at the far end.
    * Using the mean centres the expansion/compression around the
      middle of the floor, giving a more balanced and perceptually
      stronger perspective gradient.
    """
    tl, tr, br, bl = floor_quad
    top_w = float(np.linalg.norm(tr - tl))
    bot_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))

    # Geometric mean keeps the total tile area roughly correct while
    # distributing the foreshortening evenly between near and far ends.
    mean_w = float(np.sqrt(top_w * max(bot_w, 1)))
    mean_h = max(left_h, right_h)
    return mean_w, mean_h


# =====================================================================
# 4.  Lighting transfer & blending
# =====================================================================

def transfer_room_lighting(room_bgr: np.ndarray,
                           tile_bgr: np.ndarray) -> np.ndarray:
    """
    Transfer the original room image's lighting, shadows, and
    reflections onto the tile grid using LAB-space luminance matching.

    Steps
    -----
    1. Convert both images to LAB colour space.
    2. Build a low-frequency *lighting ratio* from the room's L channel.
    3. Multiply the tile's L channel by the ratio → preserves original
       light gradients, shadows, and specular highlights.
    4. Add a scaled high-frequency detail layer from the room for
       subtle surface reflections and micro-shadows.
    5. Sharpen slightly to recover crispness lost during blending.
    """
    h, w = room_bgr.shape[:2]
    tile_bgr = tile_bgr[:h, :w]  # safety crop

    room_lab = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    tile_lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    room_L = room_lab[:, :, 0]
    tile_L = tile_lab[:, :, 0]

    ksize = 61
    room_L_blur = cv2.GaussianBlur(room_L, (ksize, ksize), 0)
    tile_L_blur = cv2.GaussianBlur(tile_L, (ksize, ksize), 0)

    tile_L_blur[tile_L_blur < 1] = 1
    ratio = np.clip(room_L_blur / tile_L_blur, 0.4, 2.2)

    tile_L_lit = tile_L * ratio
    room_detail = room_L - room_L_blur
    tile_L_lit += room_detail * 0.25
    tile_L_lit = np.clip(tile_L_lit, 0, 255)
    tile_lab[:, :, 0] = tile_L_lit

    result = cv2.cvtColor(
        np.clip(tile_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR,
    )

    sharp_kernel = np.array([[0, -0.3, 0],
                             [-0.3, 2.2, -0.3],
                             [0, -0.3, 0]], dtype=np.float32)
    result = cv2.filter2D(result, -1, sharp_kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_with_lighting(
    room_bgr: np.ndarray,
    warped_tile: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 11,
) -> np.ndarray:
    """
    Composite the perspective-warped tile into the room image while
    preserving the original lighting, shadows, and highlights.

    Pipeline
    --------
    1. **Feathered alpha** – distance-transform-based soft edge.
    2. **Edge shadow**     – darkens tiles near mask boundary for
       3-D depth illusion at corners / edges.
    3. **Specular preservation** – blends bright room highlights back
       in so light glints and reflections look natural.
    4. **Alpha composite** – feathered blend of tile and room.

    Parameters
    ----------
    room_bgr     : original room image (BGR)
    warped_tile   : perspective-warped tile (BGR, same size)
    mask         : binary surface mask (0 / 255)
    feather_radius : edge-feathering width in pixels

    Returns
    -------
    Final composited BGR image.
    """
    h, w = room_bgr.shape[:2]
    warped_tile = warped_tile[:h, :w]

    # ── 1. Feathered alpha mask ──────────────────────────────────────
    binary = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    feather = np.clip(dist / max(feather_radius, 1), 0.0, 1.0)
    feather = cv2.GaussianBlur(feather.astype(np.float32), (5, 5), 0)

    # ── 2. Edge shadow / 3-D depth ──────────────────────────────────
    shadow_width = max(feather_radius * 2, 15)
    shadow_zone = np.clip(dist / shadow_width, 0.0, 1.0)
    shadow_factor = 0.65 + 0.35 * shadow_zone
    shadow_factor3 = np.stack([shadow_factor] * 3, axis=-1)

    tile_shadowed = np.clip(
        warped_tile.astype(np.float32) * shadow_factor3, 0, 255,
    ).astype(np.uint8)

    # ── 3. Specular / highlight preservation ─────────────────────────
    room_gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    highlight_thresh = (
        np.percentile(room_gray[binary > 0], 92)
        if binary.sum() > 0
        else 230
    )
    highlight_mask = (room_gray > highlight_thresh).astype(np.float32)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (11, 11), 0)

    hl = highlight_mask[:, :, None] * 0.3
    blended = (tile_shadowed.astype(np.float32) * (1.0 - hl)
               + room_bgr.astype(np.float32) * hl)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # ── 4. Final alpha composite ─────────────────────────────────────
    alpha_3 = np.stack([feather] * 3, axis=-1)
    result = (blended.astype(np.float32) * alpha_3
              + room_bgr.astype(np.float32) * (1.0 - alpha_3))

    return np.clip(result, 0, 255).astype(np.uint8)


# =====================================================================
# Real-world dimension → pixel mapping  (unchanged)
# =====================================================================

def calculate_tile_pixel_size(
    surface_mask: np.ndarray,
    floor_width_ft: float,
    floor_length_ft: float,
    tile_width_in: float,
    tile_height_in: float,
) -> tuple[int, int, int, int]:
    """
    Convert real-world tile dimensions to pixel sizes.

    Pixel sizes are computed against the **bird's-eye (flat) floor
    extent** — not the perspective bounding box — because the tiled
    texture is built on a flat canvas that is later warped.

    The number of tiles across/down is determined solely from the
    physical dimensions.  The pixel size of each tile is then chosen
    so that those tiles fill the flat floor extent exactly.  This
    means the result is **independent of the room size the user
    provides** being a perfect match for the image — the tile *count*
    is what the user controls; the pixel size auto-adapts to the mask.

    Returns ``(tile_w_px, tile_h_px, tiles_across, tiles_down)``.
    """
    # ── Number of tiles from physical dimensions ─────────────────────
    tiles_across = floor_width_ft * 12 / tile_width_in
    tiles_down   = floor_length_ft * 12 / tile_height_in

    # ── Flat floor extent from the mask quad (perspective-corrected) ──
    floor_quad = extract_floor_polygon(surface_mask)
    if floor_quad is not None:
        flat_w, flat_h = _flat_floor_dimensions(floor_quad)
        print(f"  Bird's-eye floor estimate: {flat_w:.0f} × {flat_h:.0f} px")
    else:
        # Fallback: use mask bounding box (less accurate)
        ys, xs = np.where(surface_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Empty surface mask – cannot compute tile size")
        flat_w = float(xs.max() - xs.min())
        flat_h = float(ys.max() - ys.min())
        print(f"  (fallback) Mask bbox: {flat_w:.0f} × {flat_h:.0f} px")

    # ── Tile pixel size that fills the flat canvas ───────────────────
    tile_w_px = max(int(round(flat_w / tiles_across)), 4)
    tile_h_px = max(int(round(flat_h / tiles_down)), 4)

    # Keep tile aspect ratio matching the real-world tile shape.
    # If the floor quad aspect ratio disagrees with the stated room
    # dimensions, adapt the tile size so the tile *shape* stays correct.
    real_tile_aspect = tile_width_in / tile_height_in
    pixel_tile_aspect = tile_w_px / max(tile_h_px, 1)
    if abs(pixel_tile_aspect - real_tile_aspect) / max(real_tile_aspect, 0.01) > 0.15:
        # Use the smaller dimension as the anchor and fix the other
        avg_size = (tile_w_px + tile_h_px) / 2.0
        tile_h_px = max(int(round(avg_size)), 4)
        tile_w_px = max(int(round(tile_h_px * real_tile_aspect)), 4)
        print(f"  Aspect-ratio correction applied (real tile {real_tile_aspect:.2f})")

    print(f"  Floor: {floor_width_ft}′ × {floor_length_ft}′  |  "
          f"Tile: {tile_width_in}″ × {tile_height_in}″")
    print(f"  Tiles needed: {tiles_across:.1f} across × {tiles_down:.1f} down")
    print(f"  Pixel tile size: {tile_w_px} × {tile_h_px} px  "
          f"(flat floor {flat_w:.0f}×{flat_h:.0f})")

    return tile_w_px, tile_h_px, int(np.ceil(tiles_across)), int(np.ceil(tiles_down))


# =====================================================================
# Public API  (backward-compatible with floor_tile.py & server)
# =====================================================================

def build_full_tile_grid(
    room_bgr: np.ndarray,
    surface_mask: np.ndarray,
    tile_bgr: np.ndarray,
    rotation_angle: float = 10.0,
    tile_size: int = DEFAULT_TILE_SIZE,
    grout: int = DEFAULT_GROUT_WIDTH,
    scale: int = TILE_GRID_SCALE,
    tile_w: int | None = None,
    tile_h: int | None = None,
    camera_tilt: float = DEFAULT_CAMERA_TILT,
) -> np.ndarray:
    """
    Create a **perspective-correct, homography-warped** tile grid that
    fills the entire image (same dimensions as *room_bgr*).

    Perspective pipeline
    --------------------
    1. Extract a 4-point floor quadrilateral from the surface mask.
    2. Compute the flat ("birds-eye") floor dimensions from the quad.
    3. Build an oversized flat tile canvas and optionally rotate it.
    4. Compute a homography from the canvas rectangle → floor quad.
    5. ``cv2.warpPerspective`` – tiles naturally shrink toward the
       vanishing point and grout lines converge correctly.
    6. Transfer room lighting / shadows onto the warped tile.

    Falls back to a simple centre-crop (no perspective) if the polygon
    extraction or homography computation fails.

    Parameters
    ----------
    tile_w / tile_h override tile_size when provided
    (rectangular / dimension-aware tiles).
    """
    h, w = room_bgr.shape[:2]

    # ── 1. Extract floor quadrilateral ───────────────────────────────
    floor_quad = extract_floor_polygon(surface_mask)
    use_homography = floor_quad is not None

    # ── 1b. Amplify perspective to simulate higher camera ────────────
    if use_homography and camera_tilt > 1.0:
        floor_quad = _amplify_perspective(floor_quad, camera_tilt)
        tl, tr, br, bl = floor_quad
        far_w  = float(np.linalg.norm(tr - tl))
        near_w = float(np.linalg.norm(br - bl))
        print(f"  Camera tilt {camera_tilt:.1f}× applied  │  "
              f"far-edge → {far_w:.0f} px  near-edge → {near_w:.0f} px  "
              f"ratio {near_w / max(far_w, 1):.1f}×")

    # ── 2. Decide canvas size ────────────────────────────────────────
    if use_homography:
        flat_w, flat_h = _flat_floor_dimensions(floor_quad)
        # Scale up so rotation / edge margin don't clip tiles
        canvas_w = int(flat_w * scale)
        canvas_h = int(flat_h * scale)
        # Ensure a sane minimum
        min_side = max((tile_w or tile_size), (tile_h or tile_size)) * 3
        canvas_w = max(canvas_w, min_side)
        canvas_h = max(canvas_h, min_side)
        print(f"  Bird's-eye floor: {flat_w:.0f} × {flat_h:.0f} px  "
              f"→ canvas {canvas_w} × {canvas_h}")
    else:
        canvas_w = w * scale
        canvas_h = h * scale

    # ── 3. Build flat tile texture ───────────────────────────────────
    tile_texture = create_tiled_texture(
        tile_bgr, canvas_w, canvas_h,
        tile_size=tile_size, grout=grout,
        tile_w=tile_w, tile_h=tile_h,
    )

    # ── 4. Optional rotation ─────────────────────────────────────────
    if rotation_angle != 0:
        center = (canvas_w // 2, canvas_h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        tile_texture = cv2.warpAffine(
            tile_texture, rot_mat, (canvas_w, canvas_h),
            borderMode=cv2.BORDER_REFLECT,
        )

    # ── 5. Perspective warp via homography ───────────────────────────
    if use_homography:
        # Source rectangle: centred region of the canvas whose size
        # matches the bird's-eye floor extent.
        cx, cy = canvas_w / 2.0, canvas_h / 2.0
        half_fw, half_fh = flat_w / 2.0, flat_h / 2.0
        src_pts = np.float32([
            [cx - half_fw, cy - half_fh],   # TL
            [cx + half_fw, cy - half_fh],   # TR
            [cx + half_fw, cy + half_fh],   # BR
            [cx - half_fw, cy + half_fh],   # BL
        ])

        H = compute_homography(src_pts, floor_quad)
        if H is not None:
            full_tile = warp_tile_to_floor(tile_texture, H, (w, h))
            # Report perspective scaling factor
            tl, tr, br, bl = floor_quad
            near_w = float(np.linalg.norm(br - bl))
            far_w  = float(np.linalg.norm(tr - tl))
            print(f"  ✓ Homography applied  |  near-edge {near_w:.0f} px  "
                  f"far-edge {far_w:.0f} px  "
                  f"→ {near_w / max(far_w, 1):.1f}× perspective ratio")
        else:
            print("  ⚠ Homography failed – falling back to centre-crop")
            full_tile = _centre_crop(tile_texture, w, h)
    else:
        print("  ⚠ No floor polygon found – falling back to centre-crop")
        full_tile = _centre_crop(tile_texture, w, h)

    # ── 6. Transfer room lighting ────────────────────────────────────
    full_tile = transfer_room_lighting(room_bgr, full_tile)

    return full_tile


def _centre_crop(texture: np.ndarray, w: int, h: int) -> np.ndarray:
    """Crop the centre of *texture* to ``(h, w)``."""
    th, tw = texture.shape[:2]
    ox = max((tw - w) // 2, 0)
    oy = max((th - h) // 2, 0)
    crop = texture[oy:oy + h, ox:ox + w]
    # Pad if the texture was smaller than the target
    if crop.shape[0] < h or crop.shape[1] < w:
        padded = np.zeros((h, w, 3), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        return padded
    return crop


# ── Compositing (backward-compatible wrapper) ────────────────────────

def composite_tile_on_surface(
    room_bgr: np.ndarray,
    surface_mask: np.ndarray,
    full_tile_image: np.ndarray,
    feather_radius: int = 11,
) -> np.ndarray:
    """
    Realistic composite: feathered edges, edge shadow, and specular
    highlight preservation.  Delegates to :func:`blend_with_lighting`.
    """
    return blend_with_lighting(
        room_bgr, full_tile_image, surface_mask,
        feather_radius=feather_radius,
    )
