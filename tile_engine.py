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
                        tile_width: int = None,
                        tile_height: int = None,
                        grout: int = DEFAULT_GROUT_WIDTH,
                        orientation: str = "horizontal") -> np.ndarray:
    """
    Create a flat tile grid with proper grout lines and slight per-tile colour
    variation for realism.
    """
    grout_color = 220
    pattern = np.ones((height, width, 3), dtype=np.uint8) * grout_color
    
    # Determine tile dimensions
    if tile_width and tile_height:
        tile_w, tile_h = tile_width, tile_height
    else:
        tile_w, tile_h = tile_size - grout, tile_size - grout
    
    # IMPORTANT: Resize tile image to match user-specified dimensions
    # This ensures tile maintains the user's aspect ratio
    tile_resized = cv2.resize(tile_img, (tile_w, tile_h))
    
    # Create tile pattern with BOTH horizontal and vertical grout lines
    for y in range(0, height, tile_h + grout):
        for x in range(0, width, tile_w + grout):
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)

            y_end = min(y + tile_h, height)
            x_end = min(x + tile_w, width)
            
            # Apply tile to pattern
            if y_end > y and x_end > x:
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
    """Find the 4 corner points of a surface mask via contour approx, specifically for 3D wall planes."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    
    # Use approxPolyDP with a smaller epsilon to find the true geometric corners of the wall plane
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx.reshape(-1, 2)

    if len(pts) == 4:
        return np.float32(order_points_clockwise(pts))
    
    # If not 4 points, take the 4 points that most closely form a quadrilateral
    # by taking the points with extreme coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    
    return rect


def build_tile_grid_for_plane(room_bgr: np.ndarray,
                              plane_mask: np.ndarray,
                              tile_bgr: np.ndarray,
                              rotation_angle: float,
                              tile_size: int,
                              grout: int,
                              scale: int) -> np.ndarray:
    """
    Build tile grid for a single wall plane with proper perspective.
    """
    h, w = room_bgr.shape[:2]
    
    # 1. Oversized rotated tile pattern
    big_w, big_h = w * scale, h * scale
    tile_pattern = create_tile_pattern(big_w, big_h, tile_bgr,
                                       tile_size=tile_size, tile_width=tile_width, 
                                       tile_height=tile_height, grout=grout, orientation=orientation)

    center_rot = (big_w // 2, big_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center_rot, rotation_angle, 1.0)
    tile_pattern = cv2.warpAffine(tile_pattern, rot_mat, (big_w, big_h),
                                  borderMode=cv2.BORDER_REFLECT)

    # 2. Perspective mapping for this plane (always enabled for wall planes)
    dst_pts = get_surface_corners(plane_mask)
    use_perspective = True # Always attempt perspective for wall planes
    
    print(f"Wall plane corners found: {dst_pts}")

    if dst_pts is None:
        print("No valid corners found, using fallback")
        # Fallback: crop centre of rotated pattern (no perspective)
        ox = (big_w - w) // 2
        oy = (big_h - h) // 2
        return tile_pattern[oy:oy + h, ox:ox + w]
    
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
        use_perspective = False

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

        plane_tile = cv2.remap(tile_pattern, map_x, map_y,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        # Fallback: crop centre of rotated pattern (no perspective)
        ox = (big_w - w) // 2
        oy = (big_h - h) // 2
        plane_tile = tile_pattern[oy:oy + h, ox:ox + w]

    return plane_tile


def detect_wall_planes_enhanced(wall_mask: np.ndarray) -> list:
    """
    Enhanced wall plane detection with better separation and orientation analysis.
    """
    h, w = wall_mask.shape
    binary = (wall_mask > 0).astype(np.uint8) * 255
    
    print(f"Enhanced wall detection - mask shape: {binary.shape}, non-zero: {np.sum(binary > 0)}")
    
    # 1. Morphological operations to clean and separate planes
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 2. Edge detection to find plane boundaries
    edges = cv2.Canny(cleaned, 50, 150)
    
    # 3. Find contours with hierarchy analysis
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours with hierarchy analysis")
    
    if not contours:
        return []
    
    wall_planes = []
    
    # 4. Process each contour with enhanced analysis
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        min_area = w * h * 0.03  # Lower threshold for better detection
        
        print(f"Contour {i}: area = {area}, threshold = {min_area}")
        
        if area < min_area:
            print(f"  - Skipping small contour")
            continue
            
        # 5. Enhanced corner detection
        corners = get_enhanced_wall_corners(cnt, w, h)
        if corners is None:
            print(f"  - No corners found, skipping")
            continue
            
        # 6. Enhanced orientation analysis
        orientation = analyze_wall_orientation_enhanced(corners, w, h)
        print(f"  - Enhanced orientation: {orientation}")
        
        # 7. Create precise mask for this plane
        plane_mask = np.zeros_like(wall_mask)
        cv2.fillPoly(plane_mask, [cnt], 255)
        
        # 8. Validate plane quality
        if validate_wall_plane(plane_mask, corners):
            wall_planes.append({
                'mask': plane_mask,
                'corners': corners,
                'orientation': orientation,
                'area': area,
                'contour': cnt,
                'plane_id': len(wall_planes)
            })
            print(f"  - Validated and added wall plane: {orientation}")
        else:
            print(f"  - Plane validation failed, skipping")
    
    print(f"Enhanced detection returning {len(wall_planes)} wall planes")
    return wall_planes


def get_enhanced_wall_corners(contour, img_w: int, img_h: int) -> np.ndarray:
    """
    Enhanced corner detection with multiple methods for better accuracy.
    """
    # Method 1: Convex hull with adaptive approximation
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    
    # Try multiple epsilon values for best fit
    best_corners = None
    min_diff = float('inf')
    
    for scale in [0.01, 0.02, 0.03, 0.015, 0.025]:
        approx = cv2.approxPolyDP(hull, perimeter * scale, True)
        if len(approx) == 4:
            corners = np.float32(order_points_clockwise(approx.reshape(-1, 2)))
            
            # Check corner quality
            diff = check_corner_quality(corners, img_w, img_h)
            if diff < min_diff:
                min_diff = diff
                best_corners = corners
    
    # Method 2: Minimum area rectangle as fallback
    if best_corners is None:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        best_corners = np.float32(order_points_clockwise(box))
        print(f"  - Using minAreaRect fallback")
    
    return best_corners


def analyze_wall_orientation_enhanced(corners: np.ndarray, img_w: int, img_h: int) -> str:
    """
    Enhanced wall orientation analysis with more precise angle calculation.
    """
    # Calculate all edge vectors
    edges = [
        corners[1] - corners[0],  # top
        corners[2] - corners[1],  # right
        corners[3] - corners[2],  # bottom
        corners[0] - corners[3]   # left
    ]
    
    # Calculate angles for all edges
    angles = []
    for edge in edges:
        angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
        angles.append(angle)
    
    # Normalize angles to [0, 180]
    angles = [abs(a) % 180 for a in angles]
    
    # Calculate dominant orientation
    horizontal_angle = min(angles[0], angles[2])
    vertical_angle = min(angles[1], angles[3])
    
    # Enhanced orientation determination
    if horizontal_angle < 10 and 80 < vertical_angle < 100:
        return 'front'
    elif horizontal_angle > 80:
        return 'left'
    elif horizontal_angle < 10 and vertical_angle < 10:
        return 'right'
    elif 30 < horizontal_angle < 60:
        return 'angled_left'
    elif 120 < horizontal_angle < 150:
        return 'angled_right'
    else:
        return 'angled_complex'


def validate_wall_plane(mask: np.ndarray, corners: np.ndarray) -> bool:
    """
    Validate wall plane quality and geometry.
    """
    # Check mask coverage
    if np.sum(mask > 0) < 1000:  # Minimum pixel coverage
        return False
    
    # Check corner validity
    if np.any(np.isnan(corners)) or np.any(np.isinf(corners)):
        return False
    
    # Check corner ordering (should be clockwise)
    area = cv2.contourArea(corners.astype(np.int32))
    if area <= 0:
        return False
    
    # Check aspect ratio (avoid extremely thin planes)
    edge_lengths = [
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
        np.linalg.norm(corners[0] - corners[3])
    ]
    
    max_edge = max(edge_lengths)
    min_edge = min(edge_lengths)
    
    if min_edge < max_edge * 0.1:  # Too thin
        return False
    
    return True


def check_corner_quality(corners: np.ndarray, img_w: int, img_h: int) -> float:
    """
    Check the quality of detected corners.
    """
    # Check if corners are within image bounds
    if np.any(corners < 0) or np.any(corners[:, 0] >= img_w) or np.any(corners[:, 1] >= img_h):
        return float('inf')
    
    # Check for reasonable aspect ratio
    edge_lengths = [
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
        np.linalg.norm(corners[3] - corners[2]),
        np.linalg.norm(corners[0] - corners[3])
    ]
    
    max_edge = max(edge_lengths)
    min_edge = min(edge_lengths)
    
    # Return quality metric (lower is better)
    aspect_ratio = max_edge / min_edge if min_edge > 0 else float('inf')
    return abs(aspect_ratio - 2.0)  # Prefer aspect ratios around 2:1


def calculate_wall_vanishing_point(corners: np.ndarray, wall_orientation: str) -> np.ndarray:
    """
    Calculate wall-specific vanishing point for proper perspective transformation.
    Each wall plane should have its own vanishing point, not a shared or floor-derived one.
    """
    # Extract edge vectors
    top_edge = corners[1] - corners[0]
    right_edge = corners[2] - corners[1]
    bottom_edge = corners[3] - corners[2]
    left_edge = corners[0] - corners[3]
    
    # Calculate edge midpoints
    top_mid = (corners[0] + corners[1]) / 2
    right_mid = (corners[1] + corners[2]) / 2
    bottom_mid = (corners[2] + corners[3]) / 2
    left_mid = (corners[3] + corners[0]) / 2
    
    # Calculate vanishing point based on wall orientation
    if wall_orientation in ['front', 'right']:
        # For front/right walls, vanishing point is based on horizontal perspective
        # Extend horizontal edges to find vanishing point
        if abs(top_edge[0]) > 1e-6:  # Non-zero horizontal component
            h_vanish_x = top_mid[0] + top_edge[0] * 10
            h_vanish_y = top_mid[1] + top_edge[1] * 10
        else:
            h_vanish_x, h_vanish_y = top_mid
            
        # Vertical vanishing point (typically at infinity for straight walls)
        v_vanish_x = (left_mid[0] + right_mid[0]) / 2
        v_vanish_y = min(left_mid[1], right_mid[1]) - 1000  # Far above
        
    elif wall_orientation in ['left', 'angled_left', 'angled_right']:
        # For left/angled walls, calculate perspective differently
        # Use diagonal perspective
        h_vanish_x = top_mid[0] - top_edge[0] * 10
        h_vanish_y = top_mid[1] - top_edge[1] * 10
        
        v_vanish_x = (left_mid[0] + right_mid[0]) / 2
        v_vanish_y = min(left_mid[1], right_mid[1]) - 1000
        
    else:  # angled_complex or default
        # For complex angles, use average perspective
        h_vanish_x = (top_mid[0] + bottom_mid[0]) / 2
        h_vanish_y = (top_mid[1] + bottom_mid[1]) / 2 - 500
        
        v_vanish_x = (left_mid[0] + right_mid[0]) / 2
        v_vanish_y = min(left_mid[1], right_mid[1]) - 1000
    
    # Return the primary vanishing point (horizontal perspective is more important for walls)
    return np.array([h_vanish_x, h_vanish_y])


def add_wall_separation_grout_lines(tile_surface: np.ndarray, wall_planes: list, grout_width: int) -> np.ndarray:
    """
    Add grout lines along wall separation lines where different wall planes meet.
    """
    result = tile_surface.copy()
    h, w = result.shape[:2]
    
    # Create separation mask for all wall plane boundaries
    separation_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Find edges between different wall planes
    for i, plane1 in enumerate(wall_planes):
        mask1 = plane1['mask']
        
        for j, plane2 in enumerate(wall_planes):
            if i >= j:  # Skip self and already processed pairs
                continue
                
            mask2 = plane2['mask']
            
            # Find boundary between these two planes
            # Dilate masks slightly to find overlap areas (boundaries)
            kernel = np.ones((3, 3), np.uint8)
            dilated1 = cv2.dilate(mask1, kernel, iterations=1)
            dilated2 = cv2.dilate(mask2, kernel, iterations=1)
            
            # Boundary is where dilated masks overlap but original masks don't
            boundary = cv2.bitwise_and(dilated1, dilated2)
            boundary = cv2.bitwise_and(boundary, cv2.bitwise_not(cv2.bitwise_or(mask1, mask2)))
            
            # Add to separation mask
            separation_mask = cv2.bitwise_or(separation_mask, boundary)
    
    # Expand separation lines to grout width
    if grout_width > 1:
        kernel_size = max(3, grout_width * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        separation_mask = cv2.dilate(separation_mask, kernel, iterations=1)
    
    # Apply grout color to separation lines
    grout_color = 220
    result[separation_mask > 0] = grout_color
    
    return result


def clip_grout_lines_to_mask(tile_surface: np.ndarray, surface_mask: np.ndarray) -> np.ndarray:
    """
    Clip grout lines strictly within surface mask to prevent bleeding onto adjacent surfaces.
    """
    # Create a clean result with the same dimensions
    result = np.zeros_like(tile_surface)
    
    # Ensure mask is binary (0 or 255)
    binary_mask = (surface_mask > 0).astype(np.uint8)
    
    # Apply mask strictly - only pixels within mask are preserved
    mask_3d = np.stack([binary_mask] * 3, axis=-1)
    result = np.where(mask_3d > 0, tile_surface, result)
    
    return result


def create_wall_oriented_tile_grid(tile_img: np.ndarray, tile_width: int, tile_height: int, 
                                   grout_width: int, grid_width: int, grid_height: int,
                                   wall_orientation: str = "vertical", tile_orientation: str = "horizontal") -> np.ndarray:
    """
    Create tile grid with proper orientation for wall surfaces.
    Walls need horizontal and vertical grout lines relative to their plane, not the floor's plane.
    Tile orientation respects aspect ratio: vertical = long side vertical, horizontal = long side horizontal.
    """
    # Adjust tile dimensions based on orientation to respect aspect ratio
    if tile_orientation == "vertical":
        # For vertical orientation, ensure long side is vertical
        if tile_height < tile_width:
            # Swap dimensions to make long side vertical
            tile_width, tile_height = tile_height, tile_width
    elif tile_orientation == "horizontal":
        # For horizontal orientation, ensure long side is horizontal
        if tile_width < tile_height:
            # Swap dimensions to make long side horizontal
            tile_width, tile_height = tile_height, tile_width
    
    # Resize tile to corrected dimensions
    tile_resized = cv2.resize(tile_img, (tile_width, tile_height))
    
    # Create grout color background
    grout_color = 220
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * grout_color
    
    # Calculate spacing
    tile_spacing_w = tile_width + grout_width
    tile_spacing_h = tile_height + grout_width
    
    # For walls, ensure perfectly straight grout lines aligned with wall plane
    # Draw vertical grout lines first (perfectly straight)
    for x in range(tile_width, grid_width, tile_spacing_w):
        if x < grid_width:
            # Ensure perfectly straight vertical lines
            grid[:, max(0, x-grout_width):x] = grout_color
    
    # Draw horizontal grout lines (perfectly straight)
    for y in range(tile_height, grid_height, tile_spacing_h):
        if y < grid_height:
            # Ensure perfectly straight horizontal lines
            grid[max(0, y-grout_width):y, :] = grout_color
    
    # Place tiles in the grid cells, avoiding grout lines
    for y in range(0, grid_height, tile_spacing_h):
        for x in range(0, grid_width, tile_spacing_w):
            # Calculate tile placement area (excluding grout)
            tile_start_x = x
            tile_start_y = y
            tile_end_x = min(x + tile_width, grid_width)
            tile_end_y = min(y + tile_height, grid_height)
            
            # Ensure we don't overlap grout lines
            if tile_end_x > tile_start_x and tile_end_y > tile_start_y:
                # Add minimal variation for realism
                variation = 0.98 + 0.04 * np.random.rand()
                tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)
                
                # Place tile in the grid cell
                grid[tile_start_y:tile_end_y, tile_start_x:tile_end_x] = tile_var[:tile_end_y-tile_start_y, :tile_end_x-tile_start_x]
    
    return grid


def create_perfectly_straight_tile_grid(tile_img: np.ndarray, tile_width: int, tile_height: int, 
                                       grout_width: int, grid_width: int, grid_height: int) -> np.ndarray:
    """
    Create tile grid with perfectly straight grout lines using geometric approach.
    """
    # Resize tile to user dimensions
    tile_resized = cv2.resize(tile_img, (tile_width, tile_height))
    
    # Create grout color background
    grout_color = 220
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * grout_color
    
    # Calculate spacing
    tile_spacing_w = tile_width + grout_width
    tile_spacing_h = tile_height + grout_width
    
    # First, draw all grout lines perfectly straight
    # Draw vertical grout lines
    for x in range(tile_width, grid_width, tile_spacing_w):
        if x < grid_width:
            grid[:, max(0, x-grout_width):x] = grout_color
    
    # Draw horizontal grout lines
    for y in range(tile_height, grid_height, tile_spacing_h):
        if y < grid_height:
            grid[max(0, y-grout_width):y, :] = grout_color
    
    # Then place tiles in the grid cells (avoiding grout lines)
    for y in range(0, grid_height, tile_spacing_h):
        for x in range(0, grid_width, tile_spacing_w):
            # Calculate tile placement area (excluding grout)
            tile_start_x = x
            tile_start_y = y
            tile_end_x = min(x + tile_width, grid_width)
            tile_end_y = min(y + tile_height, grid_height)
            
            # Ensure we don't overlap grout lines
            if tile_end_x > tile_start_x and tile_end_y > tile_start_y:
                # Add minimal variation for realism
                variation = 0.98 + 0.04 * np.random.rand()
                tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)
                
                # Place tile in the grid cell
                grid[tile_start_y:tile_end_y, tile_start_x:tile_end_x] = tile_var[:tile_end_y-tile_start_y, :tile_end_x-tile_start_x]
    
    return grid


def create_geometric_tile_grid_with_straight_lines(tile_img: np.ndarray, tile_width: int, tile_height: int, 
                                              grout_width: int, corners: np.ndarray) -> np.ndarray:
    """
    Create geometric tile grid that maintains perfectly straight grout lines after perspective.
    """
    h, w = corners[2][1] + 100, corners[2][0] + 100  # Estimate image size
    
    # Calculate how many tiles needed based on corner geometry
    edge_lengths = [
        np.linalg.norm(corners[1] - corners[0]),  # top
        np.linalg.norm(corners[2] - corners[1]),  # right
        np.linalg.norm(corners[3] - corners[2]),  # bottom
        np.linalg.norm(corners[0] - corners[3]),  # left
    ]
    
    # Calculate tile count
    tile_spacing_w = tile_width + grout_width
    tile_spacing_h = tile_height + grout_width
    
    tiles_x = int(max(edge_lengths[0], edge_lengths[2]) / tile_spacing_w) + 2
    tiles_y = int(max(edge_lengths[1], edge_lengths[3]) / tile_spacing_h) + 2
    
    grid_w = tiles_x * tile_spacing_w
    grid_h = tiles_y * tile_spacing_h
    
    # Create perfectly straight grid
    grid = create_perfectly_straight_tile_grid(tile_img, tile_width, tile_height, grout_width, grid_w, grid_h)
    
    return grid


def apply_perfectly_straight_homography(plane_info: dict, tile_img: np.ndarray, 
                                    tile_width: int, tile_height: int, grout_width: int,
                                    room_bgr: np.ndarray, all_planes: list, tile_orientation: str = "horizontal") -> np.ndarray:
    """
    Apply homography that preserves perfectly straight grout lines with wall-specific fixes.
    """
    mask = plane_info['mask']
    corners = plane_info['corners']
    orientation = plane_info['orientation']
    h, w = room_bgr.shape[:2]
    
    # Create wall-oriented grid with proper orientation for vertical surfaces
    wall_grid = create_wall_oriented_tile_grid(
        tile_img, tile_width, tile_height, grout_width, 
        w * 2, h * 2, orientation, tile_orientation
    )
    
    # Get grid dimensions
    grid_h, grid_w = wall_grid.shape[:2]
    
    # Create source quadrilateral
    src_quad = np.float32([
        [0, 0],
        [grid_w, 0],
        [grid_w, grid_h],
        [0, grid_h]
    ])
    
    # Calculate wall-specific vanishing point for proper perspective
    vanishing_point = calculate_wall_vanishing_point(corners, orientation)
    print(f"  - Wall vanishing point for {orientation}: {vanishing_point}")
    
    # Enhanced homography computation using wall-specific perspective
    dst_quad = corners
    homography = cv2.getPerspectiveTransform(src_quad, dst_quad)
    
    # Apply high-quality perspective transformation
    warped = cv2.warpPerspective(wall_grid, homography, (w, h), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Clip grout lines strictly within surface mask to prevent bleeding
    warped = clip_grout_lines_to_mask(warped, mask)
    
    # Enhanced grout line bending at intersections (but keep lines straight within planes)
    intersection_mask = create_enhanced_intersection_mask(mask, all_planes, plane_info)
    
    if np.sum(intersection_mask) > 0:
        warped = apply_intersection_only_bending(warped, intersection_mask, mask)
    
    # Final mask clipping to ensure no bleeding
    warped = clip_grout_lines_to_mask(warped, mask)
    
    return warped


def apply_intersection_only_bending(warped_image: np.ndarray, intersection_mask: np.ndarray, 
                                plane_mask: np.ndarray) -> np.ndarray:
    """
    Apply bending only at intersections, keeping grout lines perfectly straight within planes.
    """
    # Create very smooth transition only at intersections
    bending_smooth = cv2.GaussianBlur(intersection_mask.astype(np.float32), (21, 21), 0)
    
    # Create alpha for intersection blending only
    alpha = bending_smooth / 255.0
    alpha = np.clip(alpha, 0, 0.6)  # Moderate bending at intersections
    
    result = warped_image.copy()
    
    # Apply subtle darkening only at intersection areas
    # This preserves straight grout lines within planes
    intersection_darkening = alpha[:, :, None] * 0.1
    result = result * (1 - intersection_darkening)
    
    return result


def process_wall_planes_with_perfectly_straight_lines(wall_mask: np.ndarray, tile_img: np.ndarray, 
                                                   tile_width: int, tile_height: int, grout_width: int, 
                                                   room_bgr: np.ndarray, tile_orientation: str = "horizontal") -> np.ndarray:
    """
    Process wall planes with perfectly straight grout lines within each plane.
    """
    h, w = room_bgr.shape[:2]
    
    print(f"Perfectly straight line processing - mask shape: {wall_mask.shape}, non-zero: {np.sum(wall_mask > 0)}")
    
    # 1. Enhanced wall plane detection
    wall_planes = detect_wall_planes_enhanced(wall_mask)
    
    if not wall_planes:
        print("No wall planes detected - using simple fallback")
        return create_simple_tile_grid(w, h, tile_img, tile_width, tile_height, grout_width, wall_mask)
    
    print(f"Processing {len(wall_planes)} wall planes with perfectly straight grout lines")
    
    # 2. Process each plane with perfectly straight grout lines
    full_result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, plane_info in enumerate(wall_planes):
        print(f"Processing plane {i+1}: {plane_info['orientation']}")
        print(f"  - Mask pixels: {np.sum(plane_info['mask'] > 0)}")
        print(f"  - Corners: {plane_info['corners']}")
        
        # 3. Apply perfectly straight homography
        plane_result = apply_perfectly_straight_homography(
            plane_info, tile_img, tile_width, tile_height, grout_width, room_bgr, wall_planes, tile_orientation
        )
        print(f"  - Perfectly straight result pixels: {np.sum(plane_result > 0)}")
        
        # 4. Enhanced compositing with smooth transitions
        full_result = enhanced_plane_compositing(
            full_result, plane_result, plane_info, wall_planes
        )
        
        print(f"  - Full result after plane {i+1}: {np.sum(full_result > 0)}")
    
    # 5. Add grout lines along wall separation lines
    if len(wall_planes) > 1:
        full_result = add_wall_separation_grout_lines(full_result, wall_planes, grout_width)
        print(f"  - Added wall separation grout lines")
    
    print(f"Perfectly straight final result pixels: {np.sum(full_result > 0)}")
    
    # 6. Enhanced lighting application
    full_result = apply_enhanced_lighting(room_bgr, full_result, wall_mask)
    
    return full_result


def get_accurate_wall_corners(contour, img_w: int, img_h: int) -> np.ndarray:
    """
    Get accurate quadrilateral corners for a wall plane using multiple methods.
    """
    # Method 1: Convex hull with adaptive approximation
    hull = cv2.convexHull(contour)
    
    # Adaptive epsilon based on contour size
    perimeter = cv2.arcLength(hull, True)
    epsilon = 0.02 * perimeter  # Start with 2% of perimeter
    
    # Try different epsilon values to get exactly 4 points
    for scale in [0.02, 0.03, 0.01, 0.04]:
        approx = cv2.approxPolyDP(hull, epsilon * scale, True)
        if len(approx) == 4:
            return np.float32(order_points_clockwise(approx.reshape(-1, 2)))
    
    # Method 2: Bounding rectangle if approximation fails
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.float32(order_points_clockwise(box))


def estimate_wall_orientation(corners: np.ndarray, img_w: int, img_h: int) -> str:
    """
    Estimate wall orientation based on corner geometry.
    Returns: 'front', 'left', 'right', or 'angled'
    """
    # Calculate wall normal vector
    top_edge = corners[1] - corners[0]
    left_edge = corners[3] - corners[0]
    
    # Calculate angles
    top_angle = np.arctan2(top_edge[1], top_edge[0]) * 180 / np.pi
    left_angle = np.arctan2(left_edge[1], left_edge[0]) * 180 / np.pi
    
    # Normalize angles
    top_angle = abs(top_angle)
    left_angle = abs(left_angle)
    
    # Determine orientation
    if abs(top_angle) < 15 and abs(left_angle - 90) < 15:
        return 'front'
    elif top_angle > 75:
        return 'left'
    elif top_angle < 15 and left_angle > 75:
        return 'right'
    else:
        return 'angled'


def compute_plane_homography(corners: np.ndarray, tile_size: int, grout_width: int) -> np.ndarray:
    """
    Compute homography matrix for a wall plane to maintain consistent tile size.
    """
    # Calculate real-world dimensions based on tile size
    tile_spacing = tile_size + grout_width
    
    # Estimate how many tiles fit in each dimension
    # Use the longer edge for more accurate estimation
    edge_lengths = [
        np.linalg.norm(corners[1] - corners[0]),  # top
        np.linalg.norm(corners[2] - corners[1]),  # right
        np.linalg.norm(corners[3] - corners[2]),  # bottom
        np.linalg.norm(corners[0] - corners[3]),  # left
    ]
    
    # Create source quadrilateral in tile-space
    # This ensures consistent tile size across all planes
    max_tiles_x = int(max(edge_lengths[0], edge_lengths[2]) / tile_spacing)
    max_tiles_y = int(max(edge_lengths[1], edge_lengths[3]) / tile_spacing)
    
    src_w = max_tiles_x * tile_spacing
    src_h = max_tiles_y * tile_spacing
    
    src_quad = np.float32([
        [0, 0],
        [src_w, 0],
        [src_w, src_h],
        [0, src_h]
    ])
    
    # Compute homography
    return cv2.getPerspectiveTransform(src_quad, corners)


def create_flat_tile_grid_for_plane(tile_img: np.ndarray, tile_width: int, tile_height: int, grout_width: int, 
                                   grid_width: int, grid_height: int) -> np.ndarray:
    """
    Generate flat tile + grout grid for a specific plane.
    """
    # Resize tile to user dimensions
    tile_resized = cv2.resize(tile_img, (tile_width, tile_height))
    
    # Create grout color background
    grout_color = 220
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * grout_color
    
    # Calculate spacing
    tile_spacing_w = tile_width + grout_width
    tile_spacing_h = tile_height + grout_width
    
    # Generate flat grid with consistent tile size
    for y in range(0, grid_height, tile_spacing_h):
        for x in range(0, grid_width, tile_spacing_w):
            # Add slight variation for realism
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)
            
            y_end = min(y + tile_height, grid_height)
            x_end = min(x + tile_width, grid_width)
            
            if y_end > y and x_end > x:
                grid[y:y_end, x:x_end] = tile_var[:y_end-y, :x_end-x]
    
    return grid


def apply_independent_plane_homography_with_bending(plane_info: dict, flat_grid: np.ndarray, room_bgr: np.ndarray, 
                                               all_planes: list) -> np.ndarray:
    """
    Apply homography with grout line bending at wall intersections.
    """
    mask = plane_info['mask']
    corners = plane_info['corners']
    h, w = room_bgr.shape[:2]
    
    # Get grid dimensions
    grid_h, grid_w = flat_grid.shape[:2]
    
    # Create source quadrilateral (flat grid)
    src_quad = np.float32([
        [0, 0],
        [grid_w, 0],
        [grid_w, grid_h],
        [0, grid_h]
    ])
    
    # Destination is the wall corners
    dst_quad = corners
    
    # Compute plane-specific homography
    homography = cv2.getPerspectiveTransform(src_quad, dst_quad)
    
    # Apply perspective transformation
    warped = cv2.warpPerspective(flat_grid, homography, (w, h), flags=cv2.INTER_LINEAR)
    
    # Create grout line bending at intersections
    # Find edges where this plane meets other planes
    edge_mask = create_intersection_bending_mask(mask, all_planes, plane_info)
    
    # Apply bending effect to grout lines at intersections
    if np.sum(edge_mask) > 0:
        # Create smooth transition for grout lines
        warped = apply_grout_bending_at_intersections(warped, edge_mask, mask)
    
    # Mask to plane boundaries
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result = np.where(mask[:, :, None] > 0, warped, result)
    
    return result


def create_intersection_bending_mask(current_mask: np.ndarray, all_planes: list, current_plane: dict) -> np.ndarray:
    """
    Create mask for areas where grout lines should bend at wall intersections.
    """
    h, w = current_mask.shape
    bending_mask = np.zeros_like(current_mask)
    
    # Find edges of current plane
    kernel = np.ones((5, 5), np.uint8)
    current_dilated = cv2.dilate(current_mask, kernel, iterations=1)
    current_edges = current_dilated - current_mask
    
    # Check for adjacent planes
    for other_plane in all_planes:
        if other_plane is current_plane:
            continue
            
        other_mask = other_plane['mask']
        other_dilated = cv2.dilate(other_mask, kernel, iterations=1)
        other_edges = other_dilated - other_mask
        
        # Find intersection areas
        intersection = cv2.bitwise_and(current_edges, other_edges)
        
        # Expand intersection area for smoother bending
        intersection_expanded = cv2.dilate(intersection, np.ones((7, 7), np.uint8), iterations=2)
        
        # Add to bending mask
        bending_mask = np.maximum(bending_mask, intersection_expanded)
    
    return bending_mask


def apply_grout_bending_at_intersections(warped_image: np.ndarray, bending_mask: np.ndarray, plane_mask: np.ndarray) -> np.ndarray:
    """
    Apply smooth grout line bending at wall intersections.
    """
    # Create smooth transition effect
    bending_smooth = cv2.GaussianBlur(bending_mask.astype(np.float32), (9, 9), 0)
    
    # Create alpha for blending
    alpha = bending_smooth / 255.0
    alpha = np.clip(alpha, 0, 0.5)  # Subtle bending effect
    
    # Apply perspective-aware bending
    # Darken grout lines slightly at intersections for depth
    result = warped_image.copy()
    
    # Find grout lines (darker areas)
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    grout_mask = gray < 230  # Grout lines are darker
    
    # Apply bending only to grout lines at intersections
    grout_bending = grout_mask.astype(np.float32) * alpha[:, :, None]
    
    # Slightly darken grout lines at intersections for depth
    darkening_factor = 0.95
    result = result * (1 - grout_bending * 0.05)  # Subtle darkening
    
    return result


def process_independent_wall_planes_enhanced(wall_mask: np.ndarray, tile_img: np.ndarray, 
                                           tile_width: int, tile_height: int, grout_width: int, 
                                           room_bgr: np.ndarray, tile_orientation: str = "horizontal") -> np.ndarray:
    """
    Enhanced processing with better plane separation and grout line bending.
    """
    h, w = room_bgr.shape[:2]
    
    print(f"Enhanced wall processing - mask shape: {wall_mask.shape}, non-zero: {np.sum(wall_mask > 0)}")
    
    # 1. Enhanced wall plane detection
    wall_planes = detect_wall_planes_enhanced(wall_mask)
    
    if not wall_planes:
        print("No wall planes detected - using simple fallback")
        return create_simple_tile_grid(w, h, tile_img, tile_width, tile_height, grout_width, wall_mask)
    
    print(f"Enhanced processing {len(wall_planes)} wall planes with improved grout bending")
    
    # 2. Process each plane with enhanced precision
    full_result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, plane_info in enumerate(wall_planes):
        print(f"Enhanced processing plane {i+1}: {plane_info['orientation']}")
        print(f"  - Mask pixels: {np.sum(plane_info['mask'] > 0)}")
        print(f"  - Corners: {plane_info['corners']}")
        
        # 3. Enhanced grid dimension calculation
        grid_w, grid_h = compute_plane_grid_dimensions(
            plane_info['corners'], tile_width, tile_height, grout_width
        )
        print(f"  - Enhanced grid: {grid_w}x{grid_h}")
        
        # 4. Create enhanced flat tile grid with straight grout lines
        flat_grid = create_enhanced_flat_tile_grid(
            tile_img, tile_width, tile_height, grout_width, grid_w, grid_h,
            plane_info['orientation'], tile_orientation
        )
        print(f"  - Enhanced flat grid created: {flat_grid.shape}")
        
        # 5. Apply enhanced homography with stronger grout bending
        plane_result = apply_enhanced_homography_with_bending(
            plane_info, flat_grid, room_bgr, wall_planes, tile_orientation
        )
        print(f"  - Enhanced result pixels: {np.sum(plane_result > 0)}")
        
        # 6. Enhanced compositing with smooth transitions
        full_result = enhanced_plane_compositing(
            full_result, plane_result, plane_info, wall_planes
        )
        
        print(f"  - Full result after plane {i+1}: {np.sum(full_result > 0)}")
    
    # 7. Add grout lines along wall separation lines
    if len(wall_planes) > 1:
        full_result = add_wall_separation_grout_lines(full_result, wall_planes, grout_width)
        print(f"  - Added wall separation grout lines")
    
    print(f"Enhanced final result pixels: {np.sum(full_result > 0)}")
    
    # 8. Enhanced lighting application
    full_result = apply_enhanced_lighting(room_bgr, full_result, wall_mask)
    
    return full_result


def apply_enhanced_homography_with_bending(plane_info: dict, flat_grid: np.ndarray, 
                                        room_bgr: np.ndarray, all_planes: list, tile_orientation: str = "horizontal") -> np.ndarray:
    """
    Apply enhanced homography with wall-specific fixes for proper grout line orientation.
    """
    mask = plane_info['mask']
    corners = plane_info['corners']
    orientation = plane_info['orientation']
    h, w = room_bgr.shape[:2]
    
    # Get grid dimensions
    grid_h, grid_w = flat_grid.shape[:2]
    
    # Create source quadrilateral
    src_quad = np.float32([
        [0, 0],
        [grid_w, 0],
        [grid_w, grid_h],
        [0, grid_h]
    ])
    
    # Calculate wall-specific vanishing point for proper perspective
    vanishing_point = calculate_wall_vanishing_point(corners, orientation)
    print(f"  - Enhanced wall vanishing point for {orientation}: {vanishing_point}")
    
    # Enhanced homography computation using wall-specific perspective
    dst_quad = corners
    homography = cv2.getPerspectiveTransform(src_quad, dst_quad)
    
    # Apply high-quality perspective transformation
    warped = cv2.warpPerspective(flat_grid, homography, (w, h), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Clip grout lines strictly within surface mask to prevent bleeding
    warped = clip_grout_lines_to_mask(warped, mask)
    
    # Enhanced grout line bending at intersections
    intersection_mask = create_enhanced_intersection_mask(mask, all_planes, plane_info)
    
    if np.sum(intersection_mask) > 0:
        warped = apply_enhanced_grout_bending(warped, intersection_mask, mask, corners)
    
    # Final mask clipping to ensure no bleeding
    warped = clip_grout_lines_to_mask(warped, mask)
    
    return warped


def create_enhanced_intersection_mask(current_mask: np.ndarray, all_planes: list, 
                                  current_plane: dict) -> np.ndarray:
    """
    Create enhanced intersection mask for stronger grout bending.
    """
    h, w = current_mask.shape
    intersection_mask = np.zeros_like(current_mask)
    
    # Find edges with enhanced precision
    kernel = np.ones((7, 7), np.uint8)
    current_dilated = cv2.dilate(current_mask, kernel, iterations=2)
    current_edges = current_dilated - current_mask
    
    # Check for adjacent planes with enhanced intersection detection
    for other_plane in all_planes:
        if other_plane is current_plane:
            continue
            
        other_mask = other_plane['mask']
        other_dilated = cv2.dilate(other_mask, kernel, iterations=2)
        other_edges = other_dilated - other_mask
        
        # Enhanced intersection detection
        intersection = cv2.bitwise_and(current_edges, other_edges)
        
        # Expand intersection area for stronger bending
        intersection_expanded = cv2.dilate(intersection, np.ones((11, 11), np.uint8), iterations=3)
        
        # Add to intersection mask
        intersection_mask = np.maximum(intersection_mask, intersection_expanded)
    
    return intersection_mask


def apply_enhanced_grout_bending(warped_image: np.ndarray, intersection_mask: np.ndarray, 
                               plane_mask: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Apply enhanced grout line bending with stronger effects.
    """
    # Create very smooth transition
    bending_smooth = cv2.GaussianBlur(intersection_mask.astype(np.float32), (15, 15), 0)
    
    # Create stronger alpha for more visible bending
    alpha = bending_smooth / 255.0
    alpha = np.clip(alpha, 0, 0.8)  # Stronger bending effect
    
    result = warped_image.copy()
    
    # Enhanced grout line detection
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    grout_mask = gray < 225  # More sensitive grout detection
    
    # Apply stronger bending to grout lines at intersections
    grout_bending = grout_mask.astype(np.float32) * alpha[:, :, None]
    
    # Enhanced darkening and perspective effects
    result = result * (1 - grout_bending * 0.15)  # Stronger darkening
    
    # Add perspective-aware distortion at intersections
    if np.sum(intersection_mask) > 0:
        # Create subtle distortion effect for natural bending
        distortion = cv2.GaussianBlur(intersection_mask.astype(np.float32), (21, 21), 0)
        distortion = (distortion / 255.0)[:, :, None] * 0.1
        
        # Apply distortion to create natural bending appearance
        result = result * (1 - distortion) + (result * 0.95) * distortion
    
    return result


def enhanced_plane_compositing(full_result: np.ndarray, plane_result: np.ndarray, 
                            plane_info: dict, all_planes: list) -> np.ndarray:
    """
    Enhanced compositing with smoother transitions between planes.
    """
    plane_mask = plane_info['mask']
    
    # Create enhanced blending at intersections
    blend_mask = create_enhanced_blend_mask(plane_mask, all_planes, plane_info)
    
    if np.sum(blend_mask) > 0:
        # Smooth alpha blending
        alpha = cv2.GaussianBlur(blend_mask.astype(np.float32), (9, 9), 0) / 255.0
        alpha = alpha[:, :, None]
        
        # Enhanced blending
        existing = full_result * (1 - alpha)
        new_pixels = plane_result * alpha
        full_result = existing + new_pixels
    else:
        # Direct placement
        full_result = np.where(plane_mask[:, :, None] > 0, plane_result, full_result)
    
    return full_result


def create_enhanced_blend_mask(current_mask: np.ndarray, all_planes: list, 
                           current_plane: dict) -> np.ndarray:
    """
    Create enhanced blend mask for smoother transitions.
    """
    h, w = current_mask.shape
    blend_mask = np.zeros_like(current_mask)
    
    # Find edges with larger kernel for smoother blending
    kernel = np.ones((9, 9), np.uint8)
    current_dilated = cv2.dilate(current_mask, kernel, iterations=2)
    current_edges = current_dilated - current_mask
    
    # Check for adjacent planes
    for other_plane in all_planes:
        if other_plane is current_plane:
            continue
            
        other_mask = other_plane['mask']
        other_dilated = cv2.dilate(other_mask, kernel, iterations=2)
        other_edges = other_dilated - other_mask
        
        # Enhanced blend area
        blend_area = cv2.bitwise_and(current_edges, other_edges)
        blend_expanded = cv2.dilate(blend_area, np.ones((13, 13), np.uint8), iterations=2)
        
        blend_mask = np.maximum(blend_mask, blend_expanded)
    
    return blend_mask


def apply_enhanced_lighting(room_bgr: np.ndarray, tile_surface: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """
    Apply enhanced realistic lighting with better depth preservation.
    """
    # Extract enhanced lighting information
    gray_room = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    
    # Create enhanced lighting map
    lighting_map = gray_room.astype(np.float32) / 255.0
    
    # Apply lighting with enhanced depth
    enhanced_surface = tile_surface.astype(np.float32)
    enhanced_surface = enhanced_surface * lighting_map[:, :, None]
    
    # Add enhanced depth effects
    depth_map = cv2.GaussianBlur(wall_mask.astype(np.float32), (15, 15), 0) / 255.0
    depth_effect = 0.9 + 0.1 * (1 - depth_map)
    enhanced_surface = enhanced_surface * depth_effect[:, :, None]
    
    return np.clip(enhanced_surface, 0, 255).astype(np.uint8)


def create_enhanced_flat_tile_grid(tile_img: np.ndarray, tile_width: int, tile_height: int, 
                                grout_width: int, grid_width: int, grid_height: int,
                                wall_orientation: str = "vertical", tile_orientation: str = "horizontal") -> np.ndarray:
    """
    Generate enhanced flat tile + grout grid for a specific plane with wall-specific orientation.
    """
    # Use wall-oriented grid creation for proper vertical surface alignment
    return create_wall_oriented_tile_grid(
        tile_img, tile_width, tile_height, grout_width, 
        grid_width, grid_height, wall_orientation, tile_orientation
    )


def compute_plane_grid_dimensions(corners: np.ndarray, tile_width: int, tile_height: int, grout_width: int) -> tuple:
    """
    Compute appropriate grid dimensions for a plane to maintain consistent tile size.
    """
    # Calculate edge lengths
    edge_lengths = [
        np.linalg.norm(corners[1] - corners[0]),  # top
        np.linalg.norm(corners[2] - corners[1]),  # right
        np.linalg.norm(corners[3] - corners[2]),  # bottom
        np.linalg.norm(corners[0] - corners[3]),  # left
    ]
    
    # Calculate how many tiles fit
    tile_spacing_w = tile_width + grout_width
    tile_spacing_h = tile_height + grout_width
    
    # Use average of horizontal and vertical edges
    avg_width = (edge_lengths[0] + edge_lengths[2]) / 2
    avg_height = (edge_lengths[1] + edge_lengths[3]) / 2
    
    # Calculate grid size (add extra tiles for coverage)
    grid_width = int(avg_width / tile_spacing_w * 1.5) * tile_spacing_w
    grid_height = int(avg_height / tile_spacing_h * 1.5) * tile_spacing_h
    
    # Ensure minimum size
    grid_width = max(grid_width, tile_spacing_w * 3)
    grid_height = max(grid_height, tile_spacing_h * 3)
    
    return grid_width, grid_height


def detect_wall_planes_advanced(wall_mask: np.ndarray) -> list:
    """
    Advanced wall plane detection - wrapper for enhanced detection.
    """
    return detect_wall_planes_enhanced(wall_mask)


def process_independent_wall_planes(wall_mask: np.ndarray, tile_img: np.ndarray, 
                                  tile_width: int, tile_height: int, grout_width: int, 
                                  room_bgr: np.ndarray) -> np.ndarray:
    """
    Process each wall plane independently with natural grout bending at intersections.
    """
    h, w = room_bgr.shape[:2]
    
    print(f"Input wall mask shape: {wall_mask.shape}, non-zero pixels: {np.sum(wall_mask > 0)}")
    
    # 1. Detect and segment separate wall planes
    wall_planes = detect_wall_planes_advanced(wall_mask)
    
    if not wall_planes:
        print("No wall planes detected - using simple fallback")
        # Fallback: simple tiling without plane detection
        return create_simple_tile_grid(w, h, tile_img, tile_width, tile_height, grout_width, wall_mask)
    
    print(f"Processing {len(wall_planes)} independent wall planes")
    
    # 2. Process each plane independently
    full_result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, plane_info in enumerate(wall_planes):
        print(f"Processing plane {i+1}: {plane_info['orientation']}")
        print(f"  - Mask non-zero pixels: {np.sum(plane_info['mask'] > 0)}")
        print(f"  - Corners: {plane_info['corners']}")
        
        # 3. Compute grid dimensions for this plane
        grid_w, grid_h = compute_plane_grid_dimensions(
            plane_info['corners'], tile_width, tile_height, grout_width
        )
        print(f"  - Grid dimensions: {grid_w}x{grid_h}")
        
        # 4. Generate flat tile + grout grid for this plane
        flat_grid = create_flat_tile_grid_for_plane(
            tile_img, tile_width, tile_height, grout_width, grid_w, grid_h
        )
        print(f"  - Flat grid created, shape: {flat_grid.shape}")
        
        # 5. Apply plane-specific homography
        plane_result = apply_independent_plane_homography(
            plane_info, flat_grid, room_bgr
        )
        print(f"  - Plane result non-zero pixels: {np.sum(plane_result > 0)}")
        
        # 6. Composite with natural blending at intersections
        plane_mask = plane_info['mask']
        
        # Direct placement for debugging
        full_result = np.where(plane_mask[:, :, None] > 0, plane_result, full_result)
        
        print(f"  - Full result non-zero pixels after plane {i+1}: {np.sum(full_result > 0)}")
    
    print(f"Final result non-zero pixels: {np.sum(full_result > 0)}")
    
    # 7. Apply realistic lighting
    full_result = apply_realistic_lighting(room_bgr, full_result, wall_mask)
    
    return full_result


def split_wall_by_perspective(wall_mask: np.ndarray, main_contour):
    """
    Advanced split of a single wall mask into separate 3D planes using geometric analysis.
    """
    h, w = wall_mask.shape
    binary = (wall_mask > 0).astype(np.uint8) * 255
    
    # 1. Analyze the contour to find natural wall planes
    hull = cv2.convexHull(main_contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    pts = approx.reshape(-1, 2)
    
    print(f"Contour approximation found {len(pts)} points")
    
    # 2. For typical room layouts, try to create 3 wall planes
    wall_planes = []
    
    # Strategy 1: If we have 6+ points, likely 3 walls meeting at corners
    if len(pts) >= 6:
        # Sort points by x-coordinate to find natural wall divisions
        sorted_pts = pts[pts[:, 0].argsort()]
        
        # Find natural split points (corners)
        split_points = []
        for i in range(1, len(sorted_pts)):
            # Look for significant x-coordinate changes (corners)
            if abs(sorted_pts[i][0] - sorted_pts[i-1][0]) > w // 8:
                split_points.append(sorted_pts[i][0])
        
        # Ensure we have reasonable splits
        if len(split_points) >= 1:
            all_x = [0] + split_points + [w]
        else:
            # Fallback: divide into 3 equal sections
            section_w = w // 3
            all_x = [0, section_w, 2*section_w, w]
    else:
        # Strategy 2: Simple 3-way split
        section_w = w // 3
        all_x = [0, section_w, 2*section_w, w]
    
    # 3. Create wall planes from splits
    for i in range(len(all_x) - 1):
        x_start, x_end = int(all_x[i]), int(all_x[i+1])
        if x_end - x_start < w // 20:
            continue
            
        plane_mask = np.zeros_like(wall_mask)
        plane_mask[:, x_start:x_end] = wall_mask[:, x_start:x_end]
        
        # Only keep if there's significant wall area
        if np.sum(plane_mask > 0) > (wall_mask.size * 0.02):
            wall_planes.append(plane_mask)
            print(f"Created 3D wall plane {len(wall_planes)} at x={x_start}-{x_end}")
    
    return wall_planes if wall_planes else [wall_mask]


def create_3d_depth_map(room_bgr: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """
    Create a 3D depth map from room geometry for realistic perspective.
    """
    h, w = room_bgr.shape[:2]
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Estimate depth from vertical position (walls get darker/higher)
    depth_map = np.zeros((h, w), dtype=np.float32)
    
    # Create vertical gradient (higher = darker = farther)
    for y in range(h):
        depth_factor = 1.0 - (y / h) * 0.3  # 30% depth variation
        depth_map[y, :] = depth_factor
    
    # 2. Add horizontal perspective (center = closer)
    center_x = w // 2
    for x in range(w):
        horizontal_factor = 1.0 - abs(x - center_x) / center_x * 0.2
        depth_map[:, x] *= horizontal_factor
    
    # 3. Detect and enhance corners/edges for depth
    edges = cv2.Canny(wall_mask, 50, 150)
    kernel = np.ones((9, 9), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create depth enhancement at corners
    corner_depth = cv2.GaussianBlur(dilated_edges.astype(np.float32), (31, 31), 0) / 255.0
    corner_depth = np.power(corner_depth, 2.0) * 0.4  # Enhance corners
    
    # 4. Combine depth maps
    combined_depth = depth_map * (1.0 - corner_depth * 0.6)
    
    # 5. Smooth the depth map
    depth_smooth = cv2.GaussianBlur(combined_depth, (15, 15), 0)
    
    return depth_smooth


def create_3d_perspective_tile_grid(room_bgr: np.ndarray, wall_mask: np.ndarray, tile_bgr: np.ndarray,
                                   tile_width: int = None, tile_height: int = None,
                                   tile_size: int = DEFAULT_TILE_SIZE, grout: int = DEFAULT_GROUT_WIDTH) -> np.ndarray:
    """
    Create tiles with proper 3D perspective where grout lines demonstrate depth and wall orientation.
    """
    h, w = room_bgr.shape[:2]
    
    # Determine tile dimensions
    if tile_width and tile_height:
        tile_w, tile_h = tile_width, tile_height
    else:
        tile_w, tile_h = tile_size, tile_size
    
    # Resize tile image to user dimensions
    tile_resized = cv2.resize(tile_bgr, (tile_w, tile_h))
    
    # Get wall corners for perspective transformation
    corners = get_surface_corners(wall_mask)
    if corners is None:
        # Fallback: create simple grid without perspective
        return create_simple_tile_grid(w, h, tile_resized, tile_w, tile_h, grout, wall_mask)
    
    print(f"Applying 3D perspective with corners: {corners}")
    
    # Create destination quadrilateral from wall corners
    dst_quad = corners
    
    # Create source quadrilateral (flat grid)
    src_quad = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])
    
    # Get perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    
    # Create high-resolution grid for smooth perspective
    grid_h, grid_w = h * 2, w * 2
    
    # Generate grout grid in high resolution
    grout_color = 220
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * grout_color
    
    # Create tile pattern with proper spacing
    tile_spacing_w = (tile_w + grout) * 2
    tile_spacing_h = (tile_h + grout) * 2
    
    for y in range(0, grid_h, tile_spacing_h):
        for x in range(0, grid_w, tile_spacing_w):
            # Apply tile with variation
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)
            
            # Scale tile for high resolution
            scaled_tile = cv2.resize(tile_var, (tile_w * 2, tile_h * 2))
            
            y_end = min(y + tile_h * 2, grid_h)
            x_end = min(x + tile_w * 2, grid_w)
            
            if y_end > y and x_end > x:
                grid[y:y_end, x:x_end] = scaled_tile[:y_end-y, :x_end-x]
    
    # Apply perspective transformation
    transformed = cv2.warpPerspective(grid, M, (w, h), flags=cv2.INTER_LINEAR)
    
    # Mask to wall boundaries
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result = np.where(wall_mask[:, :, None] > 0, transformed, result)
    
    return result


def create_simple_tile_grid(w: int, h: int, tile_img: np.ndarray, tile_w: int, tile_h: int, grout: int, mask: np.ndarray) -> np.ndarray:
    """
    Create simple tile grid without perspective as fallback.
    """
    print(f"Creating simple tile grid: {w}x{h}, tiles: {tile_w}x{tile_h}, grout: {grout}")
    
    grout_color = 220
    result = np.ones((h, w, 3), dtype=np.uint8) * grout_color
    
    # Resize tile to user dimensions
    tile_resized = cv2.resize(tile_img, (tile_w, tile_h))
    
    tile_count = 0
    for y in range(0, h, tile_h + grout):
        for x in range(0, w, tile_w + grout):
            # Check if within mask
            if y >= h or x >= w:
                continue
                
            tile_region_mask = mask[y:min(y+tile_h+grout, h), x:min(x+tile_w+grout, w)]
            if np.sum(tile_region_mask) == 0:
                continue
            
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)
            
            y_end = min(y + tile_h, h)
            x_end = min(x + tile_w, w)
            
            if y_end > y and x_end > x:
                result[y:y_end, x:x_end] = tile_var[:y_end-y, :x_end-x]
                tile_count += 1
    
    print(f"Applied {tile_count} tiles in simple grid")
    return result


def apply_realistic_lighting(room_bgr: np.ndarray, tile_surface: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """
    Apply realistic lighting, shadows, and depth effects to tiled surface.
    """
    # Extract lighting information from original room image
    gray_room = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    
    # Create ambient occlusion map for corners and edges
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(wall_mask, kernel, iterations=2)
    edge_mask = cv2.subtract(dilated_mask, wall_mask)
    
    # Gaussian blur edge mask for soft shadows
    shadow_map = cv2.GaussianBlur(edge_mask.astype(np.float32), (21, 21), 0) / 255.0
    
    # Extract highlights and shadows from original room
    blurred_room = cv2.GaussianBlur(gray_room, (31, 31), 0)
    lighting_map = blurred_room.astype(np.float32) / 255.0
    
    # Enhance contrast for more dramatic shadows/highlights
    lighting_map = np.power(lighting_map, 1.3)
    lighting_map = 0.3 + 0.7 * lighting_map  # Keep base brightness
    
    # Combine lighting with shadow effects
    combined_lighting = lighting_map * (1.0 - shadow_map * 0.6)  # Darken edges
    
    # Apply lighting to tiles
    lit_tiles = tile_surface.astype(np.float32) * combined_lighting[:, :, None]
    
    # Add subtle color variation from original room
    lab_room = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2LAB)
    lab_tiles = cv2.cvtColor(lit_tiles.astype(np.uint8), cv2.COLOR_BGR2LAB)
    
    # Blend color temperature slightly
    lab_tiles[:, :, 0] = lab_tiles[:, :, 0] * 0.9 + lab_room[:, :, 0] * 0.1
    
    # Convert back to BGR
    result = cv2.cvtColor(lab_tiles, cv2.COLOR_LAB2BGR)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def build_full_tile_grid(room_bgr: np.ndarray,
                         surface_mask: np.ndarray,
                         tile_bgr: np.ndarray,
                         rotation_angle: float = 10.0,
                         tile_size: int = DEFAULT_TILE_SIZE,
                         tile_width: int = None,
                         tile_height: int = None,
                         grout: int = DEFAULT_GROUT_WIDTH,
                         orientation: str = "horizontal",
                         scale: int = TILE_GRID_SCALE,
                         surface_name: str = "floor") -> np.ndarray:
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
    
    # Handle multiple wall planes with perfectly straight grout lines
    if surface_name == "wall":
        # Process with perfectly straight grout lines within each plane
        return process_wall_planes_with_perfectly_straight_lines(
            surface_mask, tile_bgr, 
            tile_width or tile_size, tile_height or tile_size, grout, room_bgr, orientation
        )

    # 1. Oversized rotated tile pattern
    big_w, big_h = w * scale, h * scale
    tile_pattern = create_tile_pattern(big_w, big_h, tile_bgr,
                                       tile_size=tile_size, tile_width=tile_width, 
                                       tile_height=tile_height, grout=grout, orientation=orientation)

    center_rot = (big_w // 2, big_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center_rot, rotation_angle, 1.0)
    tile_pattern = cv2.warpAffine(tile_pattern, rot_mat, (big_w, big_h),
                                  borderMode=cv2.BORDER_REFLECT)

    # 2. Perspective mapping via surface corners (disabled for walls to keep tiles straight)
    dst_pts = get_surface_corners(surface_mask)
    use_perspective = False

    # Only use perspective transformation for floors, not walls
    if dst_pts is not None and surface_name != "wall":
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
                              feather_radius: int = 15) -> np.ndarray:
    """
    Enhanced compositing with better blending for realistic integration.
    """
    # Create enhanced feathered mask for smooth blending
    non_surface = 255 - surface_mask
    non_surface_smooth = cv2.GaussianBlur(non_surface, (feather_radius, feather_radius), 0)
    
    # Create edge-aware blending
    edge_mask = cv2.Canny(surface_mask, 50, 150)
    edge_dilated = cv2.dilate(edge_mask, np.ones((5,5), np.uint8), iterations=1)
    edge_smooth = cv2.GaussianBlur(edge_dilated.astype(np.float32), (11, 11), 0) / 255.0
    
    # Combine masks for sophisticated blending
    alpha = non_surface_smooth.astype(np.float32) / 255.0
    alpha_3 = np.stack([alpha] * 3, axis=-1)
    
    # Add edge-aware blending to preserve details
    edge_blend = edge_smooth[:, :, None] * 0.3
    
    # Enhanced compositing with edge preservation
    result = full_tile_image * (1.0 - alpha_3) + room_bgr * alpha_3
    
    # Add subtle original room texture through edges
    result = result * (1.0 - edge_blend) + room_bgr * edge_blend
    
    return np.clip(result, 0, 255).astype(np.uint8)
