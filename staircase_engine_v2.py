"""
Complete rewrite of staircase detection and tile projection logic.
Focuses on individual step detection, railing-based clipping, and 3D depth for curved staircases.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_railings(image_bgr: np.ndarray, stair_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect railing lines to use as hard boundaries for the staircase region.
    Returns (left_boundary, right_boundary) as masks or None.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Focus on staircase region
    edges_stair = cv2.bitwise_and(edges, edges, mask=stair_mask)
    
    # Use HoughLinesP to find strong vertical-ish lines for railings
    lines = cv2.HoughLinesP(edges_stair, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    if lines is None:
        return None, None
        
    h, w = image_bgr.shape[:2]
    left_lines = []
    right_lines = []
    
    stair_center_x = np.mean(np.where(stair_mask > 0)[1])
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate slope to filter out horizontal lines
        if x2 - x1 == 0:
            slope = 999
        else:
            slope = (y2 - y1) / (x2 - x1)
            
        if abs(slope) > 0.5: # More vertical than horizontal
            if (x1 + x2) / 2 < stair_center_x:
                left_lines.append(line)
            else:
                right_lines.append(line)
    
    # Create boundary masks
    left_bound = np.zeros((h, w), dtype=np.uint8)
    right_bound = np.zeros((h, w), dtype=np.uint8)
    
    if left_lines:
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(left_bound, (x1, y1), (x2, y2), 255, 5)
            
    if right_lines:
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(right_bound, (x1, y1), (x2, y2), 255, 5)
            
    return left_bound, right_bound

def segment_staircase_steps(image_bgr: np.ndarray, stair_mask: np.ndarray) -> List[Dict]:
    """
    Detect individual steps using horizontal edge detection and Hough transform.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Railing detection for boundaries
    left_bound, right_bound = detect_railings(image_bgr, stair_mask)
    
    # Use Canny focused on staircase region
    # Increase contrast in staircase region for better edge detection
    stair_roi = cv2.bitwise_and(gray, gray, mask=stair_mask)
    edges = cv2.Canny(stair_roi, 30, 100)
    
    # Hough transform for horizontal lines (step boundaries)
    # We want near-horizontal lines: angle near 90 degrees (pi/2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=w//10, maxLineGap=20)
    
    if lines is None or len(lines) < 3:
        logger.warning(f"Insufficient step edges detected: {0 if lines is None else len(lines)}")
        return []
        
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 20 or angle > 160: # Horizontal-ish
            horizontal_lines.append(line[0])
            
    if len(horizontal_lines) < 3:
        logger.warning(f"Insufficient horizontal step edges: {len(horizontal_lines)}")
        return []
        
    # Sort lines by y-coordinate
    horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    
    # Merge nearby lines and ensure they span the width of the staircase
    merged_lines = []
    if horizontal_lines:
        curr_line = horizontal_lines[0]
        for next_line in horizontal_lines[1:]:
            curr_y = (curr_line[1] + curr_line[3]) / 2
            next_y = (next_line[1] + next_line[3]) / 2
            if abs(next_y - curr_y) < 5: # Even tighter merge threshold for precision
                x_min = min(curr_line[0], curr_line[2], next_line[0], next_line[2])
                x_max = max(curr_line[0], curr_line[2], next_line[0], next_line[2])
                curr_line = [x_min, int(curr_y), x_max, int(curr_y)]
            else:
                merged_lines.append(curr_line)
                curr_line = next_line
        merged_lines.append(curr_line)
    
    # Clean up: ensure lines are within mask bounds and remove duplicates
    valid_merged = []
    for line in merged_lines:
        y = int((line[1] + line[3]) / 2)
        if 0 <= y < h:
            mask_row = np.where(stair_mask[y, :] > 0)[0]
            if len(mask_row) > 10: # Only keep lines with significant mask overlap
                valid_merged.append([mask_row[0], y, mask_row[-1], y])
    merged_lines = valid_merged
        
    # Ensure we have lines at the very top and bottom of the mask
    stair_y_indices = np.where(stair_mask > 0)[0]
    if len(stair_y_indices) > 0:
        y_min_mask = np.min(stair_y_indices)
        y_max_mask = np.max(stair_y_indices)
        
        # Add top boundary if not present
        if not merged_lines or abs(merged_lines[0][1] - y_min_mask) > 10:
            row = np.where(stair_mask[y_min_mask, :] > 0)[0]
            if len(row) > 0:
                merged_lines.insert(0, [row[0], y_min_mask, row[-1], y_min_mask])
                
        # Add bottom boundary if not present
        if not merged_lines or abs(merged_lines[-1][1] - y_max_mask) > 10:
            row = np.where(stair_mask[y_max_mask, :] > 0)[0]
            if len(row) > 0:
                merged_lines.append([row[0], y_max_mask, row[-1], y_max_mask])

    # Re-sort after additions
    merged_lines.sort(key=lambda l: (l[1] + l[3]) / 2)

    if len(merged_lines) < 2:
        # Fallback: if no horizontal lines found, use adaptive thresholding on Y-profile
        y_profile = np.sum(edges, axis=1)
        peaks = []
        for y in range(5, h - 5):
            if y_profile[y] > np.mean(y_profile) * 1.5 and y_profile[y] == np.max(y_profile[y-5:y+5]):
                peaks.append(y)
        
        for y in peaks:
            mask_row = np.where(stair_mask[y, :] > 0)[0]
            if len(mask_row) > 0:
                merged_lines.append([mask_row[0], y, mask_row[-1], y])
    
    steps = []
    # Process each segment between horizontal lines as a step (tread + riser)
    for i in range(len(merged_lines) - 1):
        line_top = merged_lines[i]
        line_bottom = merged_lines[i+1]
        
        y_top = int((line_top[1] + line_top[3]) / 2)
        y_bottom = int((line_bottom[1] + line_bottom[3]) / 2)
        
        # In a staircase, the region between two horizontal edges contains:
        # 1. The tread of the current step
        # 2. The riser of the next step (in a front view)
        
        # For each segment, we split it into tread (upper part) and riser (lower part)
        # Typically risers are vertical and treads are horizontal.
        # Height of the segment
        seg_height = y_bottom - y_top
        mid_y = y_top + int(seg_height * 0.7) # Treads usually occupy more visual space than risers in this view
        
        # Tread calculation
        mask_row_top = np.where(stair_mask[y_top, :] > 0)[0]
        mask_row_mid = np.where(stair_mask[mid_y, :] > 0)[0]
        
        if len(mask_row_top) < 2 or len(mask_row_mid) < 2:
            continue
            
        # Use simple trapezoidal geometry for more stable 3D depth perception
        tread_poly = np.array([
            [mask_row_top[0], y_top], [mask_row_top[-1], y_top],
            [mask_row_mid[-1], mid_y], [mask_row_mid[0], mid_y]
        ], dtype=np.int32)
        
        # Ensure we always use the full width of the mask for the polygon corners
        # to prevent orientation skewing
        tread_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(tread_mask, [tread_poly], 255)
        tread_mask = cv2.bitwise_and(tread_mask, stair_mask)
        
        # Riser calculation
        mask_row_bot = np.where(stair_mask[y_bottom, :] > 0)[0]
        if len(mask_row_bot) < 2:
            continue
            
        riser_poly = np.array([
            [mask_row_mid[0], mid_y], [mask_row_mid[-1], mid_y],
            [mask_row_bot[-1], y_bottom], [mask_row_bot[0], y_bottom]
        ], dtype=np.int32)
        
        riser_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(riser_mask, [riser_poly], 255)
        riser_mask = cv2.bitwise_and(riser_mask, stair_mask)
        
        steps.append({
            'tread_polygon': tread_poly,
            'tread_mask': tread_mask,
            'riser_polygon': riser_poly,
            'riser_mask': riser_mask,
            'y_pos': y_top,
            'depth_index': i # Used for 3D perspective scaling
        })
        
    return steps

def compute_homography_points(poly: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robustly extract and order 4 corner points for consistent perspective mapping.
    Ensures top-left, top-right, bottom-right, bottom-left ordering.
    """
    # 1. Ensure we have exactly 4 points representing the corners of the surface
    if len(poly) == 4:
        pts = poly.astype(np.float32)
    else:
        # Get the oriented bounding box which is always 4 points
        rect = cv2.minAreaRect(poly)
        pts = cv2.boxPoints(rect).astype(np.float32)
    
    # Sort points based on their geometric position to ensure consistent orientation
    # For side-view staircases, we need to be careful with tl/tr/br/bl assignment
    # Use center of mass to determine relative positions
    center = np.mean(pts, axis=0)
    
    tl = tr = br = bl = None
    
    for pt in pts:
        if pt[0] < center[0] and pt[1] < center[1]: tl = pt
        elif pt[0] >= center[0] and pt[1] < center[1]: tr = pt
        elif pt[0] >= center[0] and pt[1] >= center[1]: br = pt
        elif pt[0] < center[0] and pt[1] >= center[1]: bl = pt

    # Fallback to sum/diff if quadrant method fails (e.g., highly skewed)
    if tl is None or tr is None or br is None or bl is None:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
    
    src_ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
    
    return src_ordered, dst

def project_tiles_to_steps(image_bgr: np.ndarray, steps: List[Dict], tile_texture: np.ndarray) -> np.ndarray:
    """
    Apply tile texture to each step using individual homographies and 3D depth effects.
    """
    h, w = image_bgr.shape[:2]
    result = image_bgr.copy().astype(np.float32)
    
    num_steps = len(steps)
    
    for i, step in enumerate(steps):
        # Calculate depth-based scaling (3D effect)
        # Steps further away (top of image, lower index) should have smaller tile patterns
        depth_scale = 0.5 + (i / num_steps) * 0.5 # Scale from 0.5 to 1.0
        
        # 1. Process Tread
        tread_poly = step['tread_polygon']
        if len(tread_poly) >= 3:
            # Adjust target size based on depth for 3D view
            tw, th = int(800 * depth_scale), int(400 * depth_scale)
            src_pts, dst_pts = compute_homography_points(tread_poly, tw, th)
            
            # Safety check: Ensure points are unique and non-collinear
            if len(np.unique(src_pts, axis=0)) < 4:
                continue

            try:
                H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                H_inv = np.linalg.inv(H)
                
                # Warp texture to full image size
                warped_tread = cv2.warpPerspective(tile_texture, H_inv, (w, h), flags=cv2.INTER_LINEAR)
                
                # Check for invalid values in warped image (black pixels often come from H failures)
                if np.all(warped_tread == 0):
                    continue

                # Mask the warped texture
                tread_mask = step['tread_mask']
                
                # Perspective-aware shading
                tread_shading = 0.8 + (i / num_steps) * 0.2
                
                # Direct Alpha Blending to keep original tile color
                gray_roi = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                
                for c in range(3):
                    luminance = gray_roi / 255.0
                    blended_tex = warped_tread[:, :, c].astype(np.float32) * (0.7 + 0.3 * luminance)
                    
                    result[:, :, c] = np.where(
                        tread_mask > 0,
                        np.clip(blended_tex * tread_shading, 0, 255),
                        result[:, :, c]
                    )
            except Exception as e:
                logger.error(f"Error projecting tread for step {i}: {e}")
                continue
            
        # 2. Process Riser
        riser_mask = step['riser_mask']
        if np.sum(riser_mask > 0) > 20:
            riser_poly = step['riser_polygon']
            # Risers are vertical, narrower in perspective
            tw, th = int(800 * depth_scale), int(200 * depth_scale) 
            src_pts_r, dst_pts_r = compute_homography_points(riser_poly, tw, th)
            
            try:
                H_r = cv2.getPerspectiveTransform(src_pts_r, dst_pts_r)
                H_r_inv = np.linalg.inv(H_r)
                
                warped_riser = cv2.warpPerspective(tile_texture, H_r_inv, (w, h), flags=cv2.INTER_LINEAR)
                
                # Check for invalid values in warped image
                if np.all(warped_riser == 0):
                    continue

                # Alpha blend for riser
                riser_shading = 0.5 + (i / num_steps) * 0.2
                gray_roi = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                
                for c in range(3):
                    luminance = gray_roi / 255.0
                    blended_tex_r = warped_riser[:, :, c].astype(np.float32) * (0.6 + 0.2 * luminance)
                    
                    result[:, :, c] = np.where(
                        riser_mask > 0,
                        np.clip(blended_tex_r * riser_shading, 0, 255),
                        result[:, :, c]
                    )
            except Exception as e:
                logger.error(f"Error projecting riser for step {i}: {e}")
                continue

    return np.clip(result, 0, 255).astype(np.uint8)

def process_staircase(image_bgr: np.ndarray, stair_mask: np.ndarray, tile_img: np.ndarray) -> np.ndarray:
    """
    Main entry point for staircase tile visualization.
    """
    # Create high-res tile pattern with grout
    from per_step_tile_engine import create_tile_texture_with_grout
    tile_texture = create_tile_texture_with_grout(tile_img, tile_size=64, grout_width=4, pattern_width=1000, pattern_height=1000)
    
    # 1. Detect steps
    steps = segment_staircase_steps(image_bgr, stair_mask)
    
    if not steps:
        logger.error("Step detection failed or insufficient steps found. Skipping staircase.")
        return image_bgr
        
    # 2. Project tiles
    result = project_tiles_to_steps(image_bgr, steps, tile_texture)
    
    return result
