"""
Intelligent Edge Snapping System
=================================
Automatically corrects user-drawn lines to snap to actual edges in the image.
Similar to Photoshop's edge detection and snapping features.

Features:
- Automatic edge detection and analysis
- Smart line correction (straightens and snaps imperfect lines)
- Gradient-based edge following
- Multi-scale edge pyramids for robust detection
- Live edge snapping during drawing
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from scipy.spatial import distance


class IntelligentEdgeSnapper:
    """
    Intelligent edge detection and snapping system
    Automatically corrects user input to align with real edges
    """
    
    def __init__(self, image: np.ndarray, edge_strength_threshold: int = 50):
        """
        Initialize edge snapper
        
        Args:
            image: Input image (BGR)
            edge_strength_threshold: Minimum edge strength to consider (0-255)
        """
        self.image = image.copy()
        self.height, self.width = image.shape[:2]
        self.edge_threshold = edge_strength_threshold
        
        # Compute multi-scale edge maps
        print("🔍 Computing intelligent edge maps...")
        self.edge_map = self._compute_comprehensive_edge_map()
        self.gradient_x, self.gradient_y = self._compute_image_gradients()
        
        # Distance transform for quick nearest edge lookup
        self.edge_distance = cv2.distanceTransform(
            255 - self.edge_map, 
            cv2.DIST_L2, 
            5
        )
        
        print("✅ Edge detection complete")
    
    def _compute_comprehensive_edge_map(self) -> np.ndarray:
        """
        Compute comprehensive edge map using multiple methods
        Combines Canny, Sobel, and Laplacian edge detection
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Denoise while preserving edges
        gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Method 1: Multi-scale Canny
        canny1 = cv2.Canny(gray_filtered, 30, 100)
        canny2 = cv2.Canny(gray_filtered, 50, 150)
        canny3 = cv2.Canny(gray_filtered, 70, 200)
        canny_combined = cv2.bitwise_or(cv2.bitwise_or(canny1, canny2), canny3)
        
        # Method 2: Sobel edge detection
        sobelx = cv2.Sobel(gray_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_filtered, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_normalized = np.uint8(255 * sobel_magnitude / sobel_magnitude.max())
        _, sobel_edges = cv2.threshold(sobel_normalized, self.edge_threshold, 255, cv2.THRESH_BINARY)
        
        # Method 3: Laplacian edge detection
        laplacian = cv2.Laplacian(gray_filtered, cv2.CV_64F, ksize=3)
        laplacian_abs = np.abs(laplacian)
        laplacian_normalized = np.uint8(255 * laplacian_abs / laplacian_abs.max())
        _, laplacian_edges = cv2.threshold(laplacian_normalized, self.edge_threshold, 255, cv2.THRESH_BINARY)
        
        # Method 4: Color edges (for floors with texture/patterns)
        color_edges = self._detect_color_edges()
        
        # Combine all edge detection methods
        combined = cv2.bitwise_or(canny_combined, sobel_edges)
        combined = cv2.bitwise_or(combined, laplacian_edges)
        combined = cv2.bitwise_or(combined, color_edges)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Thin edges to single pixel width
        combined = cv2.ximgproc.thinning(combined) if hasattr(cv2, 'ximgproc') else combined
        
        return combined
    
    def _detect_color_edges(self) -> np.ndarray:
        """Detect edges based on color changes (useful for textured floors)"""
        # Convert to LAB color space for better color difference detection
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        edges_list = []
        for channel in range(3):
            # Compute gradients for each channel
            sobelx = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(lab[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            edges_list.append(magnitude)
        
        # Combine channel edges
        combined = np.max(edges_list, axis=0)
        combined_normalized = np.uint8(255 * combined / combined.max())
        
        _, edges = cv2.threshold(combined_normalized, self.edge_threshold, 255, cv2.THRESH_BINARY)
        
        return edges
    
    def _compute_image_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute image gradients for direction-aware snapping"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        return gradient_x, gradient_y
    
    def snap_point_to_nearest_edge(self, point: Tuple[int, int], 
                                   max_distance: int = 20) -> Tuple[int, int]:
        """
        Snap a point to the nearest strong edge
        
        Args:
            point: (x, y) coordinate
            max_distance: Maximum distance to search for edge
        
        Returns:
            Snapped (x, y) coordinate
        """
        x, y = point
        
        # Check if already on an edge
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.edge_map[y, x] > 0:
                return (x, y)
        
        # Search in local window
        x_min = max(0, x - max_distance)
        x_max = min(self.width, x + max_distance + 1)
        y_min = max(0, y - max_distance)
        y_max = min(self.height, y + max_distance + 1)
        
        # Extract local edge patch
        edge_patch = self.edge_map[y_min:y_max, x_min:x_max]
        
        if edge_patch.size == 0:
            return (x, y)
        
        # Find all edge points in patch
        edge_points = np.argwhere(edge_patch > 0)
        
        if len(edge_points) == 0:
            return (x, y)
        
        # Convert to absolute coordinates
        edge_points_abs = edge_points + np.array([y_min, x_min])
        
        # Compute distances to all edge points
        distances = distance.cdist([[y, x]], edge_points_abs, 'euclidean')[0]
        
        # Find nearest edge point
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] <= max_distance:
            nearest_point = edge_points_abs[nearest_idx]
            return (int(nearest_point[1]), int(nearest_point[0]))  # Return as (x, y)
        
        return (x, y)
    
    def correct_drawn_line(self, start: Tuple[int, int], end: Tuple[int, int],
                          num_samples: int = 50) -> List[Tuple[int, int]]:
        """
        Correct a user-drawn line to follow actual edges
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            num_samples: Number of points to sample along line
        
        Returns:
            List of corrected points forming the line
        """
        # First, snap endpoints to nearest edges
        start_snapped = self.snap_point_to_nearest_edge(start, max_distance=15)
        end_snapped = self.snap_point_to_nearest_edge(end, max_distance=15)
        
        # Generate path using intelligent edge following
        corrected_path = self._follow_edge_path(start_snapped, end_snapped, num_samples)
        
        return corrected_path
    
    def _follow_edge_path(self, start: Tuple[int, int], end: Tuple[int, int],
                          num_points: int = 50) -> List[Tuple[int, int]]:
        """
        Find path that follows strong edges between two points
        Uses A* pathfinding with edge strength as cost
        """
        # Use Dijkstra's algorithm with edge-based cost
        path = self._dijkstra_edge_path(start, end)
        
        if path is None or len(path) < 2:
            # Fallback to straight line
            return self._interpolate_line(start, end, num_points)
        
        # Smooth the path
        path = self._smooth_path(path)
        
        return path
    
    def _dijkstra_edge_path(self, start: Tuple[int, int], end: Tuple[int, int],
                           max_iterations: int = 10000) -> Optional[List[Tuple[int, int]]]:
        """
        Find optimal path using Dijkstra's algorithm
        Cost is based on edge strength (prefer following strong edges)
        """
        from heapq import heappush, heappop
        
        start_node = (start[1], start[0])  # Convert to (row, col)
        end_node = (end[1], end[0])
        
        # Priority queue: (cost, node)
        queue = [(0, start_node)]
        visited = set()
        came_from = {}
        cost_so_far = {start_node: 0}
        
        # Directions: 8-connected
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        iterations = 0
        while queue and iterations < max_iterations:
            iterations += 1
            current_cost, current = heappop(queue)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check if reached goal
            if current == end_node:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append((node[1], node[0]))  # Convert back to (x, y)
                    node = came_from[node]
                path.append((start_node[1], start_node[0]))
                path.reverse()
                return path
            
            # Early termination if too far from goal
            dist_to_goal = abs(current[0] - end_node[0]) + abs(current[1] - end_node[1])
            if dist_to_goal > 200:  # Too far, use simpler pathbreak
                break
            
            # Explore neighbors
            for dy, dx in directions:
                neighbor = (current[0] + dy, current[1] + dx)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width):
                    continue
                
                if neighbor in visited:
                    continue
                
                # Calculate cost (prefer edges)
                edge_strength = self.edge_map[neighbor[0], neighbor[1]]
                
                # Cost function: prefer following edges, penalize non-edges
                if edge_strength > 100:
                    step_cost = 1.0  # Low cost for strong edges
                elif edge_strength > 50:
                    step_cost = 2.0  # Medium cost
                else:
                    step_cost = 5.0  # High cost for non-edges
                
                # Add distance-to-goal heuristic
                heuristic = abs(neighbor[0] - end_node[0]) + abs(neighbor[1] - end_node[1])
                step_cost += heuristic * 0.1
                
                # Diagonal movement costs more
                if abs(dy) + abs(dx) == 2:
                    step_cost *= 1.4
                
                new_cost = cost_so_far[current] + step_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heappush(queue, (new_cost, neighbor))
        
        # If pathfinding failed, return None
        return None
    
    def _interpolate_line(self, start: Tuple[int, int], end: Tuple[int, int],
                         num_points: int) -> List[Tuple[int, int]]:
        """Interpolate straight line between two points"""
        x1, y1 = start
        x2, y2 = end
        
        points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            points.append((x, y))
        
        return points
    
    def _smooth_path(self, path: List[Tuple[int, int]], 
                     window_size: int = 5) -> List[Tuple[int, int]]:
        """Smooth path using moving average"""
        if len(path) < window_size:
            return path
        
        path_array = np.array(path, dtype=np.float32)
        
        # Apply Gaussian smoothing
        smoothed_x = ndimage.gaussian_filter1d(path_array[:, 0], sigma=2)
        smoothed_y = ndimage.gaussian_filter1d(path_array[:, 1], sigma=2)
        
        smoothed_path = [(int(x), int(y)) for x, y in zip(smoothed_x, smoothed_y)]
        
        return smoothed_path
    
    def auto_correct_polygon(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Auto-correct a polygon drawn by user to snap to edges
        
        Args:
            points: List of polygon vertices (x, y)
        
        Returns:
            Corrected polygon vertices
        """
        if len(points) < 2:
            return points
        
        corrected_points = []
        
        # Correct each edge of the polygon
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            
            # Correct the line segment
            corrected_segment = self.correct_drawn_line(start, end, num_samples=30)
            
            # Add points (avoid duplicates at vertices)
            if i == 0:
                corrected_points.extend(corrected_segment)
            else:
                corrected_points.extend(corrected_segment[1:])
        
        return corrected_points
    
    def get_edge_visualization(self, show_distance_map: bool = False) -> np.ndarray:
        """
        Get visualization of detected edges
        
        Args:
            show_distance_map: Show distance transform instead of edge map
        
        Returns:
            Visualization image
        """
        if show_distance_map:
            # Normalize distance transform for visualization
            dist_normalized = (self.edge_distance / self.edge_distance.max() * 255).astype(np.uint8)
            dist_colored = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_JET)
            return dist_colored
        else:
            # Show edge map overlaid on original image
            result = self.image.copy()
            result[self.edge_map > 0] = [0, 255, 255]  # Yellow edges
            return result


def demonstrate_edge_snapping(image_path: str):
    """
    Demonstrate edge snapping capabilities
    
    Args:
        image_path: Path to test image
    """
    print("\n" + "="*70)
    print("🧲 INTELLIGENT EDGE SNAPPING DEMONSTRATION")
    print("="*70 + "\n")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Initialize edge snapper
    snapper = IntelligentEdgeSnapper(image)
    
    # Show edge visualization
    edge_viz = snapper.get_edge_visualization()
    cv2.imwrite("edge_detection.png", edge_viz)
    print("✅ Edge visualization saved as: edge_detection.png")
    
    print("\n✨ Edge snapping system ready!")
    print("   - Multi-scale edge detection complete")
    print("   - Gradient maps computed")
    print("   - Distance transforms calculated")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Demo
    demonstrate_edge_snapping("room.jpg")
