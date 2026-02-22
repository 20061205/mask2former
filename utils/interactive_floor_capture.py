"""
Interactive Floor Capture Module
==================================
Allows users to draw rough polygons or scribbles over the floor area,
then refines them using region growing, edge snapping, and superpixel segmentation.

Key Features:
- Polygon drawing tool
- Scribble/freehand drawing
- Automatic refinement using computer vision
- Support for arbitrary polygon shapes (not limited to 4 corners)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage.segmentation import slic, felzenszwalb
from skimage.segmentation import mark_boundaries


class InteractiveFloorCapture:
    """
    Interactive tool for capturing floor regions with geometric refinement
    """
    
    def __init__(self, image: np.ndarray):
        """
        Initialize interactive floor capture
        
        Args:
            image: Input room image (BGR format)
        """
        self.image = image.copy()
        self.original_image = image.copy()
        self.height, self.width = image.shape[:2]
        
        # Drawing state - UPDATED: Support multiple polygons
        self.polygon_points: List[Tuple[int, int]] = []  # Current polygon being drawn
        self.completed_polygons: List[List[Tuple[int, int]]] = []  # All completed polygons
        self.scribble_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.is_drawing = False
        self.last_point: Optional[Tuple[int, int]] = None
        
        # Capture mode
        self.mode = 'polygon'  # 'polygon' or 'scribble'
        self.brush_size = 15
        
        # Pre-compute segmentation for refinement
        print("🔍 Computing superpixel segmentation...")
        self._compute_segmentation()
        
        # Pre-compute edge map for snapping
        print("🔍 Computing edge map...")
        self._compute_edge_map()
        
        print("✅ Interactive floor capture ready!")
    
    def _compute_segmentation(self):
        """Pre-compute superpixel segmentation for region growing"""
        # SLIC superpixels
        self.superpixels = slic(
            self.image, 
            n_segments=300, 
            compactness=10, 
            sigma=1,
            start_label=1
        )
        
        # Felzenszwalb segmentation (graph-based)
        self.graph_segments = felzenszwalb(
            self.image, 
            scale=100, 
            sigma=0.5, 
            min_size=50
        )
        
        print(f"   ✓ Generated {self.superpixels.max()} superpixels")
    
    def _compute_edge_map(self):
        """Compute edge map for intelligent snapping"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Multi-scale Canny edge detection
        edges1 = cv2.Canny(gray_filtered, 30, 100)
        edges2 = cv2.Canny(gray_filtered, 50, 150)
        self.edge_map = cv2.bitwise_or(edges1, edges2)
        
        # Dilate edges slightly for better snapping
        kernel = np.ones((3, 3), np.uint8)
        self.edge_map = cv2.dilate(self.edge_map, kernel, iterations=1)
    
    def set_mode(self, mode: str):
        """
        Set capture mode
        
        Args:
            mode: 'polygon' or 'scribble'
        """
        if mode in ['polygon', 'scribble']:
            self.mode = mode
            print(f"📍 Mode set to: {mode}")
        else:
            print(f"⚠️ Invalid mode: {mode}. Use 'polygon' or 'scribble'")
    
    def add_polygon_point(self, x: int, y: int):
        """Add a point to the current polygon"""
        self.polygon_points.append((x, y))
        print(f"   Point {len(self.polygon_points)}: ({x}, {y})")
    
    def finish_current_polygon(self):
        """Finish the current polygon and add it to completed polygons"""
        if len(self.polygon_points) >= 3:
            self.completed_polygons.append(self.polygon_points.copy())
            print(f"✅ Polygon {len(self.completed_polygons)} completed with {len(self.polygon_points)} points")
            self.polygon_points = []
            return True
        else:
            print("⚠️ Need at least 3 points to complete a polygon")
            return False
    
    def delete_last_polygon(self):
        """Delete the last completed polygon"""
        if len(self.completed_polygons) > 0:
            removed = self.completed_polygons.pop()
            print(f"🗑️ Removed polygon with {len(removed)} points ({len(self.completed_polygons)} polygons remaining)")
            return True
        else:
            print("⚠️ No completed polygons to delete")
            return False
    
    def get_polygon_count(self) -> int:
        """Get the number of completed polygons"""
        return len(self.completed_polygons)
    
    def start_scribble(self, x: int, y: int):
        """Start scribble drawing"""
        self.is_drawing = True
        self.last_point = (x, y)
        # Draw initial point
        cv2.circle(self.scribble_mask, (x, y), self.brush_size, 255, -1)
    
    def continue_scribble(self, x: int, y: int):
        """Continue scribble drawing"""
        if self.is_drawing and self.last_point is not None:
            # Draw line from last point to current point
            cv2.line(
                self.scribble_mask,
                self.last_point,
                (x, y),
                255,
                thickness=self.brush_size * 2
            )
            self.last_point = (x, y)
    
    def stop_scribble(self):
        """Stop scribble drawing"""
        self.is_drawing = False
        self.last_point = None
    
    def get_rough_mask(self) -> np.ndarray:
        """
        Get rough mask from user input (polygon or scribble)
        Combines all completed polygons plus current polygon being drawn
        
        Returns:
            Rough mask (0 or 255)
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        if self.mode == 'polygon':
            # Draw all completed polygons
            for poly_points in self.completed_polygons:
                if len(poly_points) >= 3:
                    points = np.array(poly_points, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)
            
            # Draw current polygon being drawn (if it has at least 3 points)
            if len(self.polygon_points) >= 3:
                points = np.array(self.polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
            
            return mask
        elif self.mode == 'scribble':
            # Return scribble mask
            return self.scribble_mask.copy()
        else:
            return mask
    
    def refine_mask_region_growing(self, rough_mask: np.ndarray, 
                                   tolerance: int = 20) -> np.ndarray:
        """
        Refine mask using region growing from rough mask
        
        Args:
            rough_mask: Input rough mask
            tolerance: Color tolerance for region growing
            
        Returns:
            Refined mask
        """
        print("🌱 Refining mask using region growing...")
        
        # Find seed points from rough mask
        seed_points = self._get_seed_points(rough_mask)
        
        if len(seed_points) == 0:
            print("   ⚠️ No seed points found")
            return rough_mask
        
        # Initialize refined mask
        refined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # For each seed point, perform flood fill
        temp_image = self.image.copy()
        for seed in seed_points:
            x, y = seed
            
            # Create flood fill mask
            fill_mask = np.zeros((self.height + 2, self.width + 2), dtype=np.uint8)
            flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
            
            cv2.floodFill(
                temp_image,
                fill_mask,
                (x, y),
                255,
                loDiff=(tolerance, tolerance, tolerance),
                upDiff=(tolerance, tolerance, tolerance),
                flags=flags
            )
            
            # Add to refined mask
            refined_mask = cv2.bitwise_or(refined_mask, fill_mask[1:-1, 1:-1])
        
        print("   ✓ Region growing complete")
        return refined_mask
    
    def refine_mask_superpixel(self, rough_mask: np.ndarray) -> np.ndarray:
        """
        Refine mask using superpixel segmentation
        
        Args:
            rough_mask: Input rough mask
            
        Returns:
            Refined mask using superpixels
        """
        print("🔲 Refining mask using superpixels...")
        
        # Find which superpixels are covered by rough mask
        threshold = 0.3  # At least 30% covered
        
        refined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for sp_id in range(1, self.superpixels.max() + 1):
            # Get superpixel region
            sp_region = (self.superpixels == sp_id)
            
            # Calculate overlap with rough mask
            overlap = np.sum((rough_mask > 0) & sp_region)
            total = np.sum(sp_region)
            
            if total > 0 and (overlap / total) >= threshold:
                # Include this superpixel
                refined_mask[sp_region] = 255
        
        print("   ✓ Superpixel refinement complete")
        return refined_mask
    
    def refine_mask_edge_snapping(self, rough_mask: np.ndarray, 
                                  snap_distance: int = 20) -> np.ndarray:
        """
        Refine mask by snapping to detected edges
        
        Args:
            rough_mask: Input rough mask
            snap_distance: Maximum distance to snap to edges
            
        Returns:
            Edge-snapped mask
        """
        print("🧲 Snapping mask to edges...")
        
        # Find contours of rough mask
        contours, _ = cv2.findContours(
            rough_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return rough_mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Snap each point to nearest edge
        snapped_points = []
        for point in largest_contour:
            x, y = point[0]
            
            # Find nearest edge within snap_distance
            snapped = self._snap_to_edge(x, y, snap_distance)
            snapped_points.append(snapped)
        
        # Create new mask from snapped contour
        snapped_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if len(snapped_points) > 0:
            snapped_contour = np.array(snapped_points, dtype=np.int32)
            cv2.fillPoly(snapped_mask, [snapped_contour], 255)
        
        print("   ✓ Edge snapping complete")
        return snapped_mask
    
    def _snap_to_edge(self, x: int, y: int, max_distance: int) -> Tuple[int, int]:
        """Snap a point to the nearest edge"""
        # Define search region
        x1 = max(0, x - max_distance)
        x2 = min(self.width, x + max_distance)
        y1 = max(0, y - max_distance)
        y2 = min(self.height, y + max_distance)
        
        # Search for edges in region
        region = self.edge_map[y1:y2, x1:x2]
        
        if np.any(region > 0):
            # Find closest edge pixel
            edge_points = np.argwhere(region > 0)
            
            if len(edge_points) > 0:
                # Convert to absolute coordinates
                edge_points[:, 0] += y1
                edge_points[:, 1] += x1
                
                # Find closest
                distances = np.sum((edge_points - np.array([y, x])) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                closest_point = edge_points[closest_idx]
                
                return (int(closest_point[1]), int(closest_point[0]))
        
        # No edge found, return original point
        return (x, y)
    
    def _get_seed_points(self, mask: np.ndarray, 
                        num_points: int = 20) -> List[Tuple[int, int]]:
        """Get seed points from mask for region growing"""
        # Find all non-zero points
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return []
        
        # Sample uniformly
        num_points = min(num_points, len(x_coords))
        indices = np.linspace(0, len(x_coords) - 1, num_points, dtype=int)
        
        seed_points = [(int(x_coords[i]), int(y_coords[i])) for i in indices]
        return seed_points
    
    def clear(self):
        """Clear all captured data"""
        self.polygon_points = []
        self.completed_polygons = []
        self.scribble_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.is_drawing = False
        self.last_point = None
        print("🗑️ Cleared all input")
