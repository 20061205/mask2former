"""
Per-step tile application with independent homographies for treads and risers.

Each step gets its own perspective-correct tile mapping with proper 3D geometry.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple

from config import DEFAULT_TILE_SIZE, DEFAULT_GROUT_WIDTH
from tile_engine import create_tile_pattern


def create_tile_texture_with_grout(tile_img: np.ndarray,
                                 tile_size: int = DEFAULT_TILE_SIZE,
                                 grout_width: int = DEFAULT_GROUT_WIDTH,
                                 pattern_width: int = 800,
                                 pattern_height: int = 800) -> np.ndarray:
    """
    Create a tile texture with baked-in grout lines.
    
    Parameters
    ----------
    tile_img : np.ndarray
        Base tile image
    tile_size : int
        Size of individual tiles
    grout_width : int
        Width of grout lines
    pattern_width, pattern_height : int
        Dimensions of the pattern texture
        
    Returns
    -------
    tile_pattern : np.ndarray
        Tile texture with grout lines
    """
    # Create base pattern
    pattern = np.ones((pattern_height, pattern_width, 3), dtype=np.uint8) * 220  # Grout color
    
    # Resize tile image
    tile_resized = cv2.resize(tile_img, (tile_size - grout_width, tile_size - grout_width))
    
    # Fill pattern with tiles
    for y in range(0, pattern_height, tile_size):
        for x in range(0, pattern_width, tile_size):
            # Add slight variation for realism
            variation = 0.95 + 0.1 * np.random.rand()
            tile_var = np.clip(tile_resized * variation, 0, 255).astype(np.uint8)
            
            y_end = min(y + tile_size - grout_width, pattern_height)
            x_end = min(x + tile_size - grout_width, pattern_width)
            
            if y_end > y and x_end > x:
                pattern[y:y_end, x:x_end] = tile_var[:y_end-y, :x_end-x]
    
    return pattern


def order_polygon_points_clockwise(polygon: np.ndarray) -> np.ndarray:
    """
    Order polygon points in clockwise order starting from top-left.
    
    Parameters
    ----------
    polygon : np.ndarray
        Polygon points array
        
    Returns
    -------
    ordered_points : np.ndarray
        Points ordered clockwise
    """
    # Calculate centroid
    centroid = np.mean(polygon, axis=0)
    
    # Calculate angles from centroid
    angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    ordered_points = polygon[sorted_indices]
    
    # Ensure we start from top-left (approximate)
    min_sum_idx = np.argmin(ordered_points[:, 0] + ordered_points[:, 1])
    reordered = np.roll(ordered_points, -min_sum_idx, axis=0)
    
    return reordered.astype(np.float32)


def compute_step_homography(polygon: np.ndarray,
                           target_width: int = 200,
                           target_height: int = 200) -> np.ndarray:
    """
    Compute homography for a step polygon to a rectangular target.
    
    Parameters
    ----------
    polygon : np.ndarray
        4+ polygon points in image coordinates
    target_width, target_height : int
        Dimensions of the target rectangle
        
    Returns
    -------
    H : np.ndarray
        3x3 homography matrix
    """
    # Ensure we have exactly 4 points
    if len(polygon) > 4:
        # Take the 4 corner points using convex hull
        hull = cv2.convexHull(polygon)
        if len(hull) > 4:
            # Use min area rectangle
            rect = cv2.minAreaRect(hull)
            polygon = cv2.boxPoints(rect)
        else:
            polygon = hull.reshape(-1, 2)
    elif len(polygon) < 4:
        # Pad with additional points if needed
        centroid = np.mean(polygon, axis=0)
        while len(polygon) < 4:
            # Add points around centroid
            angle = len(polygon) * np.pi / 2
            new_point = centroid + 50 * np.array([np.cos(angle), np.sin(angle)])
            polygon = np.vstack([polygon, new_point])
    
    # Order points clockwise
    ordered_polygon = order_polygon_points_clockwise(polygon[:4])
    
    # Define destination rectangle points
    dst_points = np.float32([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ])
    
    # Compute homography
    H = cv2.getPerspectiveTransform(ordered_polygon, dst_points)
    
    return H


def apply_tile_to_step(room_bgr: np.ndarray,
                       step_mask: np.ndarray,
                       step_polygon: np.ndarray,
                       tile_texture: np.ndarray,
                       is_vertical: bool = False,
                       tile_size: int = DEFAULT_TILE_SIZE,
                       grout_width: int = DEFAULT_GROUT_WIDTH) -> np.ndarray:
    """
    Apply tiles to a single step with perspective correction.
    
    Parameters
    ----------
    room_bgr : np.ndarray
        Original room image
    step_mask : np.ndarray
        Binary mask for this step
    step_polygon : np.ndarray
        Polygon points for this step
    tile_texture : np.ndarray
        Tile texture with grout lines
    is_vertical : bool
        Whether this is a vertical surface (riser) or horizontal (tread)
    tile_size : int
        Size of individual tiles
    grout_width : int
        Width of grout lines
        
    Returns
    -------
    step_result : np.ndarray
        Room image with tiles applied to this step
    """
    h, w = room_bgr.shape[:2]
    result = room_bgr.copy()
    
    # Compute homography for this step
    target_size = max(tile_size * 3, 300)  # Ensure enough resolution
    H = compute_step_homography(step_polygon, target_size, target_size)
    
    # Create a larger tile pattern for this step
    if is_vertical:
        # For vertical surfaces, use less tile variation
        step_tile_pattern = create_tile_texture_with_grout(
            tile_texture, tile_size, grout_width, target_size * 2, target_size * 2
        )
        # Apply slight rotation for vertical surfaces
        center = (target_size, target_size)
        rot_mat = cv2.getRotationMatrix2D(center, 5, 1.0)  # Small rotation
        step_tile_pattern = cv2.warpAffine(step_tile_pattern, rot_mat, 
                                          (target_size * 2, target_size * 2))
    else:
        # For horizontal surfaces, use normal pattern
        step_tile_pattern = create_tile_texture_with_grout(
            tile_texture, tile_size, grout_width, target_size * 2, target_size * 2
        )
        # Apply more rotation for horizontal surfaces
        center = (target_size, target_size)
        rot_mat = cv2.getRotationMatrix2D(center, 15, 1.0)  # More rotation
        step_tile_pattern = cv2.warpAffine(step_tile_pattern, rot_mat,
                                          (target_size * 2, target_size * 2))
    
    # Warp tile pattern to step polygon
    warped_tiles = cv2.warpPerspective(step_tile_pattern, H, (w, h))
    
    # Create mask for this step
    step_mask_3d = np.stack([step_mask > 0] * 3, axis=-1)
    
    # Extract original lighting from the step region
    step_region = room_bgr.copy()
    step_region[step_mask == 0] = 0
    
    # Calculate average lighting in step region
    gray_step = cv2.cvtColor(step_region, cv2.COLOR_BGR2GRAY)
    mask_3ch = np.stack([step_mask > 0] * 3, axis=-1)
    
    # Apply Gaussian blur to get smooth lighting
    lighting = cv2.GaussianBlur(gray_step, (31, 31), 0)
    lighting[step_mask == 0] = 128  # Neutral value for non-step areas
    lighting = lighting / 128.0  # Normalize
    
    # Apply lighting to warped tiles
    lighting_3d = np.stack([lighting] * 3, axis=-1)
    lit_tiles = np.clip(warped_tiles.astype(np.float32) * lighting_3d, 0, 255).astype(np.uint8)
    
    # Composite tiles onto result
    result = result.astype(np.float32)
    tiles_masked = lit_tiles.astype(np.float32) * (step_mask_3d.astype(np.float32) / 255.0)
    result = result * (1 - step_mask_3d.astype(np.float32) / 255.0) + tiles_masked
    
    return result.astype(np.uint8)


def apply_tiles_to_all_steps(room_bgr: np.ndarray,
                           steps: List[Dict],
                           tile_img: np.ndarray,
                           tile_size: int = DEFAULT_TILE_SIZE,
                           grout_width: int = DEFAULT_GROUT_WIDTH) -> np.ndarray:
    """
    Apply tiles to all staircase steps with independent homographies.
    
    Parameters
    ----------
    room_bgr : np.ndarray
        Original room image
    steps : list[dict]
        List of step dictionaries with masks and polygons
    tile_img : np.ndarray
        Base tile image
    tile_size : int
        Size of individual tiles
    grout_width : int
        Width of grout lines
        
    Returns
    -------
    result : np.ndarray
        Room image with tiles applied to all steps
    """
    result = room_bgr.copy()
    
    for i, step in enumerate(steps):
        # Apply tiles to tread
        if 'tread_mask' in step and 'tread_polygon' in step:
            if np.sum(step['tread_mask'] > 0) > 0:
                result = apply_tile_to_step(
                    result, step['tread_mask'], step['tread_polygon'],
                    tile_img, is_vertical=False, tile_size=tile_size, grout_width=grout_width
                )
        
        # Apply tiles to riser
        if 'riser_mask' in step and 'riser_polygon' in step:
            if np.sum(step['riser_mask'] > 0) > 0:
                result = apply_tile_to_step(
                    result, step['riser_mask'], step['riser_polygon'],
                    tile_img, is_vertical=True, tile_size=tile_size, grout_width=grout_width
                )
    
    return result


def create_debug_visualization(room_bgr: np.ndarray,
                              steps: List[Dict],
                              output_path: str):
    """
    Create debug visualization showing detected steps.
    
    Parameters
    ----------
    room_bgr : np.ndarray
        Original room image
    steps : list[dict]
        List of step dictionaries
    output_path : str
        Path to save debug visualization
    """
    debug_img = room_bgr.copy()
    
    for i, step in enumerate(steps):
        # Draw tread polygon in green
        if 'tread_polygon' in step and step['tread_polygon'] is not None:
            cv2.polylines(debug_img, [step['tread_polygon'].astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(debug_img, f'T{i}', tuple(step['tread_polygon'][0].astype(np.int32)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw riser polygon in red
        if 'riser_polygon' in step and step['riser_polygon'] is not None:
            cv2.polylines(debug_img, [step['riser_polygon'].astype(np.int32)], True, (0, 0, 255), 2)
            cv2.putText(debug_img, f'R{i}', tuple(step['riser_polygon'][0].astype(np.int32)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, debug_img)
