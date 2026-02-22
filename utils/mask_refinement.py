"""
Mask Refinement Tools
======================
Tools for refining floor masks using morphological operations
and optional correction tools (add/remove brush strokes).

Key Features:
- Smooth edges with morphological operations
- Fill holes and remove islands
- Add/remove regions with brush tools
- Gaussian smoothing for natural boundaries
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


class MaskRefinement:
    """
    Tools for refining and smoothing floor masks
    """
    
    def __init__(self):
        """Initialize mask refinement tools"""
        pass
    
    def smooth_edges(self, mask: np.ndarray, 
                    method: str = 'morphological',
                    kernel_size: int = 5) -> np.ndarray:
        """
        Smooth mask edges using various methods
        
        Args:
            mask: Input binary mask (0 or 255)
            method: 'morphological', 'gaussian', or 'bilateral'
            kernel_size: Size of smoothing kernel (odd number)
            
        Returns:
            Smoothed mask
        """
        if method == 'morphological':
            return self._smooth_morphological(mask, kernel_size)
        elif method == 'gaussian':
            return self._smooth_gaussian(mask, kernel_size)
        elif method == 'bilateral':
            return self._smooth_bilateral(mask, kernel_size)
        else:
            print(f"⚠️ Unknown smoothing method: {method}")
            return mask
    
    def _smooth_morphological(self, mask: np.ndarray, 
                             kernel_size: int = 5) -> np.ndarray:
        """
        Smooth using morphological operations (opening + closing)
        
        This removes small protrusions and fills small holes
        """
        print("🔄 Applying morphological smoothing...")
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        # Closing: fills small holes
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Opening: removes small protrusions
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        
        print("   ✓ Morphological smoothing complete")
        return smoothed
    
    def _smooth_gaussian(self, mask: np.ndarray, 
                        kernel_size: int = 5) -> np.ndarray:
        """
        Smooth using Gaussian blur
        
        Creates softer, more natural boundaries
        """
        print("🔄 Applying Gaussian smoothing...")
        
        # Apply Gaussian blur
        smoothed = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Re-threshold to binary
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        
        print("   ✓ Gaussian smoothing complete")
        return smoothed
    
    def _smooth_bilateral(self, mask: np.ndarray, 
                         kernel_size: int = 5) -> np.ndarray:
        """
        Smooth using bilateral filter
        
        Preserves edges while smoothing
        """
        print("🔄 Applying bilateral smoothing...")
        
        # Convert to float for bilateral filter
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply bilateral filter
        smoothed = cv2.bilateralFilter(mask_float, kernel_size, 75, 75)
        
        # Convert back to binary
        smoothed = (smoothed * 255).astype(np.uint8)
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        
        print("   ✓ Bilateral smoothing complete")
        return smoothed
    
    def fill_holes(self, mask: np.ndarray, 
                   min_hole_size: int = 100) -> np.ndarray:
        """
        Fill holes in the mask
        
        Args:
            mask: Input binary mask
            min_hole_size: Minimum hole size to fill (in pixels)
            
        Returns:
            Mask with holes filled
        """
        print("🕳️ Filling holes in mask...")
        
        # Invert mask to find holes
        inverted = cv2.bitwise_not(mask)
        
        # Find connected components (holes)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        
        # Create output mask
        filled = mask.copy()
        
        # Fill holes (skip background label 0)
        holes_filled = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            
            if area < min_hole_size:
                # This is a hole to fill
                filled[labels == label] = 255
                holes_filled += 1
        
        print(f"   ✓ Filled {holes_filled} holes")
        return filled
    
    def remove_islands(self, mask: np.ndarray, 
                      min_island_size: int = 200) -> np.ndarray:
        """
        Remove small disconnected regions (islands)
        
        Args:
            mask: Input binary mask
            min_island_size: Minimum island size to keep (in pixels)
            
        Returns:
            Mask with small islands removed
        """
        print("🏝️ Removing small islands...")
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Create output mask
        cleaned = np.zeros_like(mask)
        
        # Keep large components (skip background label 0)
        islands_removed = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            
            if area >= min_island_size:
                cleaned[labels == label] = 255
            else:
                islands_removed += 1
        
        print(f"   ✓ Removed {islands_removed} small islands")
        return cleaned
    
    def erode_mask(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Erode mask (shrink boundaries)
        
        Args:
            mask: Input binary mask
            iterations: Number of erosion iterations
            
        Returns:
            Eroded mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask, kernel, iterations=iterations)
        print(f"   ↔️ Eroded mask by {iterations} iterations")
        return eroded
    
    def dilate_mask(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Dilate mask (expand boundaries)
        
        Args:
            mask: Input binary mask
            iterations: Number of dilation iterations
            
        Returns:
            Dilated mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        print(f"   ↔️ Dilated mask by {iterations} iterations")
        return dilated
    
    def refine_mask_complete(self, mask: np.ndarray,
                            smooth: bool = True,
                            fill_holes: bool = True,
                            remove_islands: bool = True) -> np.ndarray:
        """
        Complete mask refinement pipeline
        
        Args:
            mask: Input binary mask
            smooth: Apply smoothing
            fill_holes: Fill holes in mask
            remove_islands: Remove small disconnected regions
            
        Returns:
            Fully refined mask
        """
        print("🔧 Starting complete mask refinement...")
        
        refined = mask.copy()
        
        if fill_holes:
            refined = self.fill_holes(refined, min_hole_size=100)
        
        if remove_islands:
            refined = self.remove_islands(refined, min_island_size=200)
        
        if smooth:
            refined = self.smooth_edges(refined, method='morphological', kernel_size=5)
        
        print("✅ Mask refinement complete!")
        return refined
    
    def add_region_brush(self, mask: np.ndarray, 
                        center: Tuple[int, int],
                        radius: int = 20) -> np.ndarray:
        """
        Add a circular region to the mask (brush tool)
        
        Args:
            mask: Input binary mask
            center: (x, y) center of brush
            radius: Brush radius
            
        Returns:
            Modified mask
        """
        modified = mask.copy()
        cv2.circle(modified, center, radius, 255, -1)
        return modified
    
    def remove_region_brush(self, mask: np.ndarray,
                           center: Tuple[int, int],
                           radius: int = 20) -> np.ndarray:
        """
        Remove a circular region from the mask (eraser tool)
        
        Args:
            mask: Input binary mask
            center: (x, y) center of brush
            radius: Brush radius
            
        Returns:
            Modified mask
        """
        modified = mask.copy()
        cv2.circle(modified, center, radius, 0, -1)
        return modified
    
    def smooth_contour_approximation(self, mask: np.ndarray, 
                                    epsilon_factor: float = 0.001) -> np.ndarray:
        """
        Smooth mask by approximating contours with fewer points
        
        Args:
            mask: Input binary mask
            epsilon_factor: Approximation accuracy (smaller = more accurate)
            
        Returns:
            Mask with smoothed contours
        """
        print("📐 Smoothing contours...")
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return mask
        
        # Approximate contours
        smoothed_contours = []
        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            smoothed_contours.append(approx)
        
        # Create new mask from smoothed contours
        smoothed_mask = np.zeros_like(mask)
        cv2.drawContours(smoothed_mask, smoothed_contours, -1, 255, -1)
        
        print("   ✓ Contours smoothed")
        return smoothed_mask
