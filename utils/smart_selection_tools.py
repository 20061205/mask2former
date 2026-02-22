"""
Advanced Smart Selection Tools
===============================
Photoshop-like intelligent selection tools for floor correction:

1. Magic Wand - Click to select similar colored regions
2. Quick Selection - Paint to select with intelligent edge detection  
3. Intelligent Scissors - Click points to create edge-following paths
4. Object Select - Click inside/outside to intelligently select
5. Color Range Select - Select all similar colors in image
6. Grow/Shrink - Expand or contract selection intelligently

All tools use AI and edge detection for smart, accurate selections.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


class SmartSelectionTools:
    """
    Advanced selection tools for intelligent floor area correction
    Similar to Adobe Photoshop's selection toolkit
    """
    
    def __init__(self, image: np.ndarray, edge_map: Optional[np.ndarray] = None):
        """
        Initialize smart selection tools
        
        Args:
            image: Input image (BGR)
            edge_map: Pre-computed edge map (optional, will compute if None)
        """
        self.image = image.copy()
        self.height, self.width = image.shape[:2]
        
        # Compute edge map if not provided
        if edge_map is None:
            self.edge_map = self._compute_edge_map()
        else:
            self.edge_map = edge_map
        
        # Convert to different color spaces for better selection
        self.image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        self.image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        print("✅ Smart selection tools initialized")
    
    def _compute_edge_map(self) -> np.ndarray:
        """Compute edge map for intelligent selection"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Multi-scale Canny
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        return edges
    
    def magic_wand_select(self, seed_point: Tuple[int, int], 
                         tolerance: int = 30,
                         use_contiguous: bool = True) -> np.ndarray:
        """
        Magic Wand tool - Select similar colored pixels
        
        Args:
            seed_point: (x, y) point to start selection
            tolerance: Color tolerance (0-255)
            use_contiguous: If True, only select connected pixels
        
        Returns:
            Selection mask (0-255)
        """
        x, y = seed_point
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        if use_contiguous:
            # Contiguous selection (flood fill)
            mask = np.zeros((self.height + 2, self.width + 2), dtype=np.uint8)
            
            flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
            
            cv2.floodFill(
                self.image.copy(),
                mask,
                (x, y),
                255,
                loDiff=(tolerance, tolerance, tolerance),
                upDiff=(tolerance, tolerance, tolerance),
                flags=flags
            )
            
            # Remove padding
            result = mask[1:-1, 1:-1]
        else:
            # Global selection (all similar pixels)
            seed_color = self.image[y, x].astype(np.int32)
            
            # Calculate color distance
            color_diff = np.abs(self.image.astype(np.int32) - seed_color)
            color_distance = np.sqrt(np.sum(color_diff**2, axis=2))
            
            # Create mask
            result = (color_distance <= tolerance).astype(np.uint8) * 255
        
        # Refine using edge information
        result = self._refine_with_edges(result)
        
        return result
    
    def quick_selection_brush(self, scribble_mask: np.ndarray,
                             is_foreground: bool = True,
                             refinement_iterations: int = 3) -> np.ndarray:
        """
        Quick Selection tool - Paint to select with intelligent edge detection
        
        Args:
            scribble_mask: Mask showing where user painted (255 = painted)
            is_foreground: True if painting foreground, False for background
            refinement_iterations: Number of GrabCut iterations
        
        Returns:
            Selection mask (0-255)
        """
        print("🖌️ Quick selection with intelligent edge detection...")
        
        # Initialize GrabCut mask
        gc_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        gc_mask[:, :] = cv2.GC_PR_BGD  # Start with probable background
        
        # Set scribbled areas
        if is_foreground:
            gc_mask[scribble_mask > 0] = cv2.GC_FGD  # Definite foreground
        else:
            gc_mask[scribble_mask > 0] = cv2.GC_BGD  # Definite background
        
        # Set some probable foreground (expand scribbles slightly)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        expanded = cv2.dilate(scribble_mask, kernel, iterations=1)
        gc_mask[expanded > 0] = cv2.GC_PR_FGD
        
        # Initialize models
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        
        try:
            # Apply GrabCut
            cv2.grabCut(
                self.image,
                gc_mask,
                None,
                bg_model,
                fg_model,
                refinement_iterations,
                cv2.GC_INIT_WITH_MASK
            )
            
            # Extract foreground
            result = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0
            ).astype(np.uint8)
            
            # Refine with edges
            result = self._refine_with_edges(result)
            
            print("✅ Quick selection complete")
            return result
            
        except Exception as e:
            print(f"⚠️ Quick selection failed: {e}")
            return scribble_mask
    
    def intelligent_scissors(self, control_points: List[Tuple[int, int]],
                           closed_path: bool = True) -> np.ndarray:
        """
        Intelligent Scissors tool - Click points, path follows edges
        
        Args:
            control_points: List of (x, y) control points
            closed_path: If True, connect last point to first
        
        Returns:
            Selection mask (0-255)
        """
        print(f"✂️ Intelligent scissors with {len(control_points)} points...")
        
        if len(control_points) < 2:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Find edge-following paths between each pair of points
        all_path_points = []
        
        for i in range(len(control_points) - 1):
            path_segment = self._find_edge_path_dijkstra(
                control_points[i],
                control_points[i + 1]
            )
            all_path_points.extend(path_segment[:-1])  # Avoid duplicates
        
        # Add last point
        all_path_points.append(control_points[-1])
        
        # Close path if requested
        if closed_path and len(control_points) > 2:
            closing_segment = self._find_edge_path_dijkstra(
                control_points[-1],
                control_points[0]
            )
            all_path_points.extend(closing_segment)
        
        # Create mask from path
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Convert points to numpy array
        path_array = np.array(all_path_points, dtype=np.int32)
        
        if closed_path:
            # Fill polygon
            cv2.fillPoly(mask, [path_array], 255)
        else:
            # Draw path line
            cv2.polylines(mask, [path_array], False, 255, 2)
        
        print("✅ Intelligent scissors complete")
        return mask
    
    def _find_edge_path_dijkstra(self, start: Tuple[int, int], 
                                 end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find path that follows edges using Dijkstra's algorithm
        """
        from heapq import heappush, heappop
        
        # Convert to (row, col)
        start_node = (start[1], start[0])
        end_node = (end[1], end[0])
        
        # Priority queue
        queue = [(0, start_node)]
        visited = set()
        came_from = {}
        cost_so_far = {start_node: 0}
        
        # 8-connected neighbors
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        max_iterations = 5000
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            _, current = heappop(queue)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end_node:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append((current[1], current[0]))  # Convert to (x, y)
                    current = came_from[current]
                path.append((start_node[1], start_node[0]))
                path.reverse()
                return path
            
            # Explore neighbors
            for dy, dx in directions:
                neighbor = (current[0] + dy, current[1] + dx)
                
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width):
                    continue
                
                if neighbor in visited:
                    continue
                
                # Cost based on edge strength (prefer edges)
                edge_val = self.edge_map[neighbor[0], neighbor[1]]
                cost = 10.0 - (edge_val / 255.0) * 9.0  # Range: 1.0 to 10.0
                
                # Add Euclidean distance heuristic
                dist = np.sqrt((neighbor[0] - end_node[0])**2 + (neighbor[1] - end_node[1])**2)
                heuristic = dist * 0.5
                
                new_cost = cost_so_far[current] + cost + heuristic
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heappush(queue, (new_cost, neighbor))
        
        # Fallback to straight line if pathfinding fails
        return self._bresenham_line(start, end)
    
    def _bresenham_line(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm"""
        x0, y0 = start
        x1, y1 = end
        
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def object_select_tool(self, positive_points: List[Tuple[int, int]],
                          negative_points: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        Object Select tool - Click inside object to select, outside to exclude
        Uses advanced segmentation algorithms
        
        Args:
            positive_points: Points inside object to select
            negative_points: Points outside object (background)
        
        Returns:
            Selection mask (0-255)
        """
        print("🎯 Object select with AI assistance...")
        
        if negative_points is None:
            negative_points = []
        
        # Create scribble masks
        fg_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        bg_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Draw points with radius
        radius = 10
        for point in positive_points:
            cv2.circle(fg_mask, point, radius, 255, -1)
        
        for point in negative_points:
            cv2.circle(bg_mask, point, radius, 255, -1)
        
        # Use quick selection
        combined_mask = fg_mask.copy()
        result = self.quick_selection_brush(combined_mask, is_foreground=True)
        
        # Remove background areas
        result[bg_mask > 0] = 0
        
        print("✅ Object select complete")
        return result
    
    def color_range_select(self, reference_point: Tuple[int, int],
                          fuzziness: int = 50,
                          use_lab_space: bool = True) -> np.ndarray:
        """
        Color Range selection - Select all similar colors in image
        
        Args:
            reference_point: Reference point for color
            fuzziness: Color tolerance (0-100)
            use_lab_space: Use LAB color space for perceptual similarity
        
        Returns:
            Selection mask (0-255)
        """
        x, y = reference_point
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        if use_lab_space:
            # Use LAB color space for perceptual similarity
            ref_color = self.image_lab[y, x].astype(np.float32)
            
            # Calculate Delta E (perceptual color difference)
            diff = self.image_lab.astype(np.float32) - ref_color
            distance = np.sqrt(np.sum(diff**2, axis=2))
        else:
            # Use RGB space
            ref_color = self.image[y, x].astype(np.float32)
            diff = self.image.astype(np.float32) - ref_color
            distance = np.sqrt(np.sum(diff**2, axis=2))
        
        # Create mask based on fuzziness
        threshold = fuzziness * 2.55  # Convert 0-100 to 0-255
        mask = (distance <= threshold).astype(np.uint8) * 255
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def grow_selection(self, current_mask: np.ndarray, 
                      iterations: int = 5) -> np.ndarray:
        """
        Grow selection to similar neighboring regions
        
        Args:
            current_mask: Current selection mask
            iterations: Number of growth iterations
        
        Returns:
            Grown selection mask
        """
        print(f"📈 Growing selection ({iterations} iterations)...")
        
        # Use watershed algorithm for intelligent growth
        markers = np.zeros_like(current_mask, dtype=np.int32)
        markers[current_mask > 127] = 1  # Foreground
        markers[current_mask == 0] = 2   # Background
        
        # Mark uncertain areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (iterations * 4 + 1, iterations * 4 + 1))
        dilated = cv2.dilate(current_mask, kernel, iterations=1)
        uncertain = dilated - current_mask
        markers[uncertain > 0] = 0  # Unknown region
        
        try:
            # Apply watershed
            markers_result = cv2.watershed(self.image, markers)
            
            # Extract grown mask
            result = np.where(markers_result == 1, 255, 0).astype(np.uint8)
            
            print("✅ Selection grown")
            return result
        except:
            # Fallback to simple dilation
            result = cv2.dilate(current_mask, kernel, iterations=1)
            print("✅ Selection grown (simple method)")
            return result
    
    def shrink_selection(self, current_mask: np.ndarray,
                        pixels: int = 5) -> np.ndarray:
        """
        Shrink selection by specified pixels
        
        Args:
            current_mask: Current selection mask
            pixels: Number of pixels to shrink
        
        Returns:
            Shrunk selection mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
        result = cv2.erode(current_mask, kernel, iterations=1)
        return result
    
    def _refine_with_edges(self, mask: np.ndarray, 
                          edge_threshold: int = 100) -> np.ndarray:
        """
        Refine mask boundaries using edge information
        Snaps mask boundaries to strong edges
        """
        # Find contours of current mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Refine contours to align with edges
        refined_contours = []
        
        for contour in contours:
            refined = []
            for point in contour:
                x, y = point[0]
                
                # Search for nearest strong edge
                snap_radius = 8
                x_min = max(0, x - snap_radius)
                x_max = min(self.width, x + snap_radius + 1)
                y_min = max(0, y - snap_radius)
                y_max = min(self.height, y + snap_radius + 1)
                
                edge_patch = self.edge_map[y_min:y_max, x_min:x_max]
                
                if edge_patch.size > 0:
                    # Find strongest edge in patch
                    edge_points = np.argwhere(edge_patch > edge_threshold)
                    
                    if len(edge_points) > 0:
                        # Find closest edge
                        distances = np.sqrt(
                            (edge_points[:, 1] - (x - x_min))**2 +
                            (edge_points[:, 0] - (y - y_min))**2
                        )
                        closest_idx = np.argmin(distances)
                        
                        if distances[closest_idx] < snap_radius:
                            new_y = edge_points[closest_idx, 0] + y_min
                            new_x = edge_points[closest_idx, 1] + x_min
                            refined.append([[new_x, new_y]])
                            continue
                
                refined.append([[x, y]])
            
            refined_contours.append(np.array(refined, dtype=np.int32))
        
        # Create refined mask
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, refined_contours, -1, 255, -1)
        
        return refined_mask
    
    def smooth_selection(self, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """Smooth selection edges"""
        return cv2.medianBlur(mask, radius * 2 + 1)
    
    def feather_selection(self, mask: np.ndarray, radius: int = 10) -> np.ndarray:
        """Feather (blur) selection edges"""
        blurred = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return result


# Demonstration
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎨 SMART SELECTION TOOLS - PHOTOSHOP-STYLE")
    print("="*70)
    print("\nAvailable tools:")
    print("  1. Magic Wand - Click to select similar colors")
    print("  2. Quick Selection - Paint to select intelligently")
    print("  3. Intelligent Scissors - Click points for edge-following paths")
    print("  4. Object Select - Click inside/outside objects")
    print("  5. Color Range - Select all similar colors")
    print("  6. Grow/Shrink - Expand or contract selections")
    print("\n" + "="*70 + "\n")
