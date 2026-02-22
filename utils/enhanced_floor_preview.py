"""
Enhanced Floor Detection Preview & Visualization
=================================================
Shows detected floor area with multiple visualization modes:
- Overlay view with adjustable transparency
- Side-by-side comparison
- Edge highlighting
- Confidence heatmap
- Interactive preview with zoom

This gives users a clear view of what was detected before manual correction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FloorPreviewVisualizer:
    """
    Advanced visualization for floor detection results
    Shows users the detected area clearly before manual correction
    """
    
    def __init__(self, image: np.ndarray, mask: np.ndarray):
        """
        Initialize preview visualizer
        
        Args:
            image: Original image (BGR)
            mask: Detected floor mask (binary or grayscale)
        """
        self.image = image.copy()
        self.mask = mask.copy()
        self.height, self.width = image.shape[:2]
        
        # Normalize mask to 0-255
        if self.mask.max() <= 1:
            self.mask = (self.mask * 255).astype(np.uint8)
        
        # Compute boundaries for better visualization
        self.boundaries = self._compute_boundaries()
        
        # Visualization settings
        self.overlay_alpha = 0.5
        self.boundary_color = (0, 255, 255)  # Yellow
        self.floor_color = (0, 255, 0)  # Green
        
    def _compute_boundaries(self) -> np.ndarray:
        """Compute boundary edges of detected floor area"""
        # Find contours
        contours, _ = cv2.findContours(
            self.mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        # Draw boundaries
        boundaries = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(boundaries, contours, -1, 255, 2)
        
        return boundaries
    
    def show_preview_dialog(self) -> bool:
        """
        Show interactive preview dialog
        User can accept or reject the detection
        
        Returns:
            True if user accepts, False if rejects
        """
        window_name = "Floor Detection Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Create trackbars for interactive adjustment
        cv2.createTrackbar('View Mode', window_name, 0, 4, lambda x: None)
        cv2.createTrackbar('Alpha', window_name, 50, 100, lambda x: None)
        cv2.createTrackbar('Show Boundary', window_name, 1, 1, lambda x: None)
        
        print("\n" + "="*70)
        print("📊 FLOOR DETECTION PREVIEW")
        print("="*70)
        print("\n🎨 Preview Controls:")
        print("  View Mode Slider:")
        print("    0 = Overlay")
        print("    1 = Side-by-side")
        print("    2 = Mask only")
        print("    3 = Edges highlighted")
        print("    4 = Heatmap")
        print("\n  Alpha Slider: Adjust overlay transparency")
        print("  Show Boundary: Toggle boundary edges")
        print("\n  Keyboard:")
        print("    SPACE/ENTER = Accept and continue to correction")
        print("    R = Reject and restart detection")
        print("    S = Save preview image")
        print("    ESC = Exit")
        print("\n" + "="*70 + "\n")
        
        while True:
            # Get trackbar values
            view_mode = cv2.getTrackbarPos('View Mode', window_name)
            alpha = cv2.getTrackbarPos('Alpha', window_name) / 100.0
            show_boundary = cv2.getTrackbarPos('Show Boundary', window_name) == 1
            
            # Generate visualization based on mode
            if view_mode == 0:
                display = self.create_overlay_view(alpha, show_boundary)
            elif view_mode == 1:
                display = self.create_sidebyside_view(show_boundary)
            elif view_mode == 2:
                display = self.create_mask_view()
            elif view_mode == 3:
                display = self.create_edge_highlight_view()
            elif view_mode == 4:
                display = self.create_heatmap_view()
            
            # Add statistics overlay
            display = self._add_statistics_overlay(display)
            
            # Add instructions
            display = self._add_instructions_overlay(display, view_mode)
            
            cv2.imshow(window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord(' ') or key == 13:  # SPACE or ENTER
                print("✅ Detection accepted - Proceeding to manual correction")
                cv2.destroyWindow(window_name)
                return True
            
            elif key == ord('r') or key == ord('R'):
                print("❌ Detection rejected - Will restart detection")
                cv2.destroyWindow(window_name)
                return False
            
            elif key == ord('s') or key == ord('S'):
                filename = "floor_preview.png"
                cv2.imwrite(filename, display)
                print(f"💾 Preview saved as: {filename}")
            
            elif key == 27:  # ESC
                print("❌ Preview cancelled")
                cv2.destroyWindow(window_name)
                return False
        
        cv2.destroyWindow(window_name)
        return True
    
    def create_overlay_view(self, alpha: float = 0.5, show_boundary: bool = True) -> np.ndarray:
        """
        Create overlay visualization with detected floor highlighted
        
        Args:
            alpha: Overlay transparency (0-1)
            show_boundary: Whether to show boundary edges
        
        Returns:
            Visualization image
        """
        result = self.image.copy()
        
        # Create colored overlay
        overlay = self.image.copy()
        overlay[self.mask > 0] = self.floor_color
        
        # Blend
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        # Add boundary
        if show_boundary:
            result[self.boundaries > 0] = self.boundary_color
        
        return result
    
    def create_sidebyside_view(self, show_boundary: bool = True) -> np.ndarray:
        """Create side-by-side comparison view"""
        # Original image on left
        left = self.image.copy()
        
        # Detection visualization on right
        right = self.create_overlay_view(0.5, show_boundary)
        
        # Combine
        result = np.hstack([left, right])
        
        # Add dividing line
        cv2.line(result, (self.width, 0), (self.width, self.height), (255, 255, 255), 3)
        
        # Add labels
        cv2.putText(result, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "Detected Floor", (self.width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result
    
    def create_mask_view(self) -> np.ndarray:
        """Create mask-only view"""
        # Convert mask to BGR
        mask_bgr = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        
        # Make it green
        mask_colored = np.zeros_like(self.image)
        mask_colored[self.mask > 0] = self.floor_color
        
        return mask_colored
    
    def create_edge_highlight_view(self) -> np.ndarray:
        """Create view with edge boundaries highlighted"""
        result = self.image.copy()
        
        # Dim the non-floor areas
        mask_inv = cv2.bitwise_not(self.mask)
        result[mask_inv > 0] = (result[mask_inv > 0] * 0.4).astype(np.uint8)
        
        # Highlight boundaries in bright yellow
        result[self.boundaries > 0] = self.boundary_color
        
        # Thicken boundary for better visibility
        kernel = np.ones((3, 3), np.uint8)
        boundaries_thick = cv2.dilate(self.boundaries, kernel, iterations=1)
        result[boundaries_thick > 0] = self.boundary_color
        
        return result
    
    def create_heatmap_view(self) -> np.ndarray:
        """Create confidence heatmap view"""
        # Normalize mask to 0-1
        mask_normalized = self.mask.astype(np.float32) / 255.0
        
        # Apply colormap (hot = high confidence, cold = low confidence)
        heatmap = cv2.applyColorMap((mask_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        result = cv2.addWeighted(self.image, 0.4, heatmap, 0.6, 0)
        
        # Add colorbar legend
        result = self._add_colorbar(result)
        
        return result
    
    def _add_colorbar(self, image: np.ndarray) -> np.ndarray:
        """Add colorbar legend to heatmap"""
        h, w = image.shape[:2]
        
        # Create colorbar
        colorbar_width = 30
        colorbar_height = 200
        colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
        
        # Fill with gradient
        for i in range(colorbar_height):
            value = int((1 - i / colorbar_height) * 255)
            color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            colorbar[i, :] = color
        
        # Position colorbar
        x = w - colorbar_width - 20
        y = h - colorbar_height - 50
        
        # Add to image
        image[y:y+colorbar_height, x:x+colorbar_width] = colorbar
        
        # Add labels
        cv2.putText(image, "High", (x + 35, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, "Low", (x + 35, y + colorbar_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image
    
    def _add_statistics_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add statistics overlay showing detection quality metrics"""
        h, w = image.shape[:2]
        
        # Calculate statistics
        total_pixels = self.mask.size
        floor_pixels = np.sum(self.mask > 0)
        coverage = (floor_pixels / total_pixels) * 100
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_regions = len(contours)
        
        # Largest region size
        if contours:
            largest_area = max([cv2.contourArea(cnt) for cnt in contours])
            largest_coverage = (largest_area / total_pixels) * 100
        else:
            largest_coverage = 0
        
        # Create semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, h - 120), (300, h - 10), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Add statistics text
        stats_text = [
            "DETECTION STATISTICS:",
            f"Coverage: {coverage:.1f}%",
            f"Regions: {num_regions}",
            f"Largest: {largest_coverage:.1f}%"
        ]
        
        y_offset = h - 100
        for i, text in enumerate(stats_text):
            if i == 0:
                color = (0, 255, 255)
                thickness = 2
            else:
                color = (255, 255, 255)
                thickness = 1
            
            cv2.putText(image, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            y_offset += 25
        
        return image
    
    def _add_instructions_overlay(self, image: np.ndarray, view_mode: int) -> np.ndarray:
        """Add instructions overlay"""
        h, w = image.shape[:2]
        
        # Mode name
        mode_names = ["Overlay", "Side-by-Side", "Mask Only", "Edge Highlight", "Heatmap"]
        mode_name = mode_names[view_mode]
        
        # Create semi-transparent background for header
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # Add header text
        cv2.putText(image, f"Mode: {mode_name}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, "SPACE=Accept | R=Reject | S=Save | ESC=Exit", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image
    
    def export_visualizations(self, output_dir: str = "outputs"):
        """
        Export all visualization modes to files
        
        Args:
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📸 Exporting visualizations to {output_dir}/...")
        
        # Save all views
        views = {
            "1_overlay.png": self.create_overlay_view(),
            "2_sidebyside.png": self.create_sidebyside_view(),
            "3_mask.png": self.create_mask_view(),
            "4_edges.png": self.create_edge_highlight_view(),
            "5_heatmap.png": self.create_heatmap_view()
        }
        
        for filename, view in views.items():
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, view)
            print(f"  ✅ Saved: {filename}")
        
        print("✅ All visualizations exported!\n")


def preview_floor_detection(image: np.ndarray, mask: np.ndarray, 
                           auto_export: bool = False) -> bool:
    """
    Show interactive preview of floor detection
    
    Args:
        image: Original image (BGR)
        mask: Detected floor mask
        auto_export: Whether to automatically export all visualizations
    
    Returns:
        True if user accepts detection, False otherwise
    """
    visualizer = FloorPreviewVisualizer(image, mask)
    
    # Export visualizations if requested
    if auto_export:
        visualizer.export_visualizations()
    
    # Show interactive preview
    return visualizer.show_preview_dialog()
