"""
Plane Approximation Module
===========================
Computes best-fit floor plane from refined mask and estimates
perspective transformation for realistic tile projection.

Key Features:
- Compute convex hull and minimum area quadrilateral
- Estimate perspective transformation matrix
- Handle vanishing point detection
- Support arbitrary polygon shapes for masking
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial import ConvexHull


class PlaneApproximation:
    """
    Geometric plane approximation for floor projection
    """
    
    def __init__(self, image: np.ndarray, mask: np.ndarray):
        """
        Initialize plane approximation
        
        Args:
            image: Original room image (BGR)
            mask: Refined floor mask (binary, 0 or 255)
        """
        self.image = image
        self.mask = mask
        self.height, self.width = image.shape[:2]
        
        # Results
        self.convex_hull_points: Optional[np.ndarray] = None
        self.quad_points: Optional[np.ndarray] = None
        self.perspective_matrix: Optional[np.ndarray] = None
        self.vanishing_point: Optional[Tuple[float, float]] = None
        
        print("✅ Plane approximation initialized")
    
    def compute_convex_hull(self) -> np.ndarray:
        """
        Compute convex hull of the floor mask
        
        Returns:
            Convex hull points (N x 2 array)
        """
        print("🔺 Computing convex hull...")
        
        # Get all mask points
        y_coords, x_coords = np.where(self.mask > 0)
        
        if len(x_coords) < 3:
            print("   ⚠️ Not enough points for convex hull")
            return np.array([])
        
        # Compute convex hull
        points = np.column_stack((x_coords, y_coords))
        
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            self.convex_hull_points = hull_points
            print(f"   ✓ Convex hull with {len(hull_points)} vertices")
            
            return hull_points
        except Exception as e:
            print(f"   ⚠️ Error computing convex hull: {e}")
            return np.array([])
    
    def compute_minimum_area_quad(self) -> np.ndarray:
        """
        Compute approximating quadrilateral that respects perspective.
        Instead of a simple 2D minAreaRect (which fails for trapezoidal floor masks),
        this finds the tightest convex quadrilateral that fits the floor mask,
        prioritizing corners that align with the room's perspective.
        
        Returns:
            4 corner points (4 x 2 array) in order: top-left, top-right, bottom-right, bottom-left
        """
        print("📐 Computing perspective-aware quadrilateral...")
        
        # First compute convex hull
        if self.convex_hull_points is None:
            self.compute_convex_hull()
            
        # Fallback if hull failed
        if self.convex_hull_points is None or len(self.convex_hull_points) < 4:
            return self._get_bounding_box()
            
        # ── ROBUST QUADRILATERAL FITTING ──────────────────────────────────────
        # We need to simplify the convex hull down to exactly 4 points.
        # cv2.approxPolyDP with increasing epsilon is the standard robust way.
        
        epsilon = 0.005 * cv2.arcLength(self.convex_hull_points, True)
        max_epsilon = 0.1 * cv2.arcLength(self.convex_hull_points, True)
        
        approx = None
        
        # Iteratively increase epsilon until we get 4 points
        # Start small to preserve detail, increase to force simplification
        current_epsilon = epsilon
        while current_epsilon < max_epsilon:
            approx = cv2.approxPolyDP(self.convex_hull_points, current_epsilon, True)
            
            if len(approx) == 4:
                # Found exactly 4 corners!
                quad_reshaped = approx.reshape(4, 2).astype(np.float32)
                # Ensure convex
                if cv2.isContourConvex(approx):
                    self.quad_points = self._order_quadrilateral_points(quad_reshaped)
                    print("   ✓ Found exact 4-corner perspective quad via simplification")
                    return self.quad_points
                break # If not convex, break and use fallback
            
            elif len(approx) < 4:
                # Oversimplified (triangle) - backtrack and force 4 points manually
                break
            
            current_epsilon += 0.005 * cv2.arcLength(self.convex_hull_points, True)
            
        # Strategy 2: Geometric Extremes on Convex Hull (Robust Fallback)
        # If simplification failed, we find the "corners" by looking for 
        # points that are most extreme in the 4 diagonal directions.
        # This works for any convex-ish shape (trapezoid, rotated rect, etc).
        
        print("   ⚠️ approxPolyDP failed, using geometric extremes fallback")
        hull_pts = self.convex_hull_points.reshape(-1, 2)
        
        # Calculate sums and differences
        # Top-Left: minimize (x + y)
        # Bottom-Right: maximize (x + y)
        sum_pts = hull_pts.sum(axis=1)
        tl_idx = np.argmin(sum_pts)
        br_idx = np.argmax(sum_pts)
        
        # Top-Right: maximize (x - y) <=> minimize (y-x)
        # Bottom-Left: minimize (x - y) <=> maximize (y-x)
        diff_pts = np.diff(hull_pts, axis=1).flatten() # y - x
        
        tr_idx = np.argmin(diff_pts) # min(y-x) is max(x-y) -> TR
        bl_idx = np.argmax(diff_pts) # max(y-x) is min(x-y) -> BL
        
        corners = hull_pts[[tl_idx, tr_idx, br_idx, bl_idx]]
        
        # Ensure we have 4 distinct points (for very small/degenerate masks)
        if len(np.unique(corners, axis=0)) < 4:
             return self._get_bounding_box()

        self.quad_points = self._order_quadrilateral_points(corners.astype(np.float32))
        print("   ✓ Computed robust quad from hull extremes")
        return self.quad_points
        
        # (Old fallback code removed - replaced by geometric extremes above)

    
    def _order_quadrilateral_points(self, points: np.ndarray) -> np.ndarray:
        """
        Order quadrilateral points in consistent order:
        top-left, top-right, bottom-right, bottom-left
        
        Args:
            points: 4 x 2 array of points
            
        Returns:
            Ordered 4 x 2 array
        """
        # Sort by y-coordinate
        sorted_by_y = points[np.argsort(points[:, 1])]
        
        # Top two points
        top_points = sorted_by_y[:2]
        top_points = top_points[np.argsort(top_points[:, 0])]  # Sort by x
        top_left, top_right = top_points
        
        # Bottom two points
        bottom_points = sorted_by_y[2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # Sort by x
        bottom_left, bottom_right = bottom_points
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def _get_bounding_box(self) -> np.ndarray:
        """Get axis-aligned bounding box as fallback"""
        y_coords, x_coords = np.where(self.mask > 0)
        
        if len(x_coords) == 0:
            # Return full image
            return np.array([
                [0, 0],
                [self.width - 1, 0],
                [self.width - 1, self.height - 1],
                [0, self.height - 1]
            ], dtype=np.float32)
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        return np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype=np.float32)
    
    def estimate_perspective_transform(self, 
                                      target_width: int = 1000,
                                      target_height: int = 1000) -> np.ndarray:
        """
        Estimate perspective transformation matrix to unwarp floor to top-down view
        
        Args:
            target_width: Width of target unwarped view
            target_height: Height of target unwarped view
            
        Returns:
            3x3 perspective transformation matrix
        """
        print("🎯 Estimating perspective transformation...")
        
        # Get quadrilateral if not computed
        if self.quad_points is None:
            self.compute_minimum_area_quad()
        
        # Source points (quadrilateral corners in image)
        src_points = self.quad_points.astype(np.float32)
        
        # Destination points (rectangle in top-down view)
        dst_points = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        print("   ✓ Perspective matrix computed")
        return self.perspective_matrix
    
    def get_inverse_perspective_transform(self) -> Optional[np.ndarray]:
        """
        Get inverse perspective transformation (top-down → image perspective)
        
        Returns:
            Inverse 3x3 transformation matrix
        """
        if self.perspective_matrix is None:
            self.estimate_perspective_transform()
        
        if self.perspective_matrix is not None:
            return np.linalg.inv(self.perspective_matrix)
        return None
    
    def detect_vanishing_point(self) -> Optional[Tuple[float, float]]:
        """
        Detect vanishing point from floor edges
        Uses Hough line detection and line intersection
        
        Returns:
            (x, y) vanishing point, or None if not detected
        """
        print("🎯 Detecting vanishing point...")
        
        # Apply Canny edge detection on masked region
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=self.mask)
        edges = cv2.Canny(masked_gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) < 2:
            print("   ⚠️ Not enough lines detected for vanishing point")
            return None
        
        # Find intersection of lines
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                
                # Convert to Cartesian form
                a1 = np.cos(theta1)
                b1 = np.sin(theta1)
                a2 = np.cos(theta2)
                b2 = np.sin(theta2)
                
                # Solve for intersection
                det = a1 * b2 - a2 * b1
                
                if abs(det) > 1e-6:  # Lines are not parallel
                    x = (b2 * rho1 - b1 * rho2) / det
                    y = (a1 * rho2 - a2 * rho1) / det
                    intersections.append((x, y))
        
        if len(intersections) == 0:
            print("   ⚠️ No line intersections found")
            return None
        
        # Use median intersection as vanishing point
        intersections = np.array(intersections)
        vp_x = np.median(intersections[:, 0])
        vp_y = np.median(intersections[:, 1])
        
        self.vanishing_point = (float(vp_x), float(vp_y))
        print(f"   ✓ Vanishing point: ({vp_x:.1f}, {vp_y:.1f})")
        
        return self.vanishing_point
    
    def get_projection_quadrilateral(self) -> np.ndarray:
        """
        Get the quadrilateral to use for tile projection
        
        Returns:
            4 x 2 array of corner points
        """
        if self.quad_points is None:
            self.compute_minimum_area_quad()
        
        return self.quad_points
    
    def visualize_plane_approximation(self) -> np.ndarray:
        """
        Create visualization of plane approximation
        
        Returns:
            Visualization image
        """
        vis = self.image.copy()
        
        # Draw convex hull
        if self.convex_hull_points is not None:
            cv2.polylines(
                vis, 
                [self.convex_hull_points.astype(np.int32)], 
                True, 
                (255, 255, 0), 
                2
            )
        
        # Draw quadrilateral
        if self.quad_points is not None:
            cv2.polylines(
                vis,
                [self.quad_points.astype(np.int32)],
                True,
                (0, 255, 0),
                3
            )
            
            # Draw corner points
            for i, point in enumerate(self.quad_points):
                cv2.circle(vis, tuple(point.astype(int)), 8, (0, 0, 255), -1)
                cv2.putText(
                    vis, 
                    str(i), 
                    tuple(point.astype(int) + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        # Draw vanishing point
        if self.vanishing_point is not None:
            vp = (int(self.vanishing_point[0]), int(self.vanishing_point[1]))
            cv2.circle(vis, vp, 10, (255, 0, 255), -1)
            cv2.putText(
                vis,
                "VP",
                (vp[0] + 15, vp[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )
        
        return vis
