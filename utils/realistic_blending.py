"""
Realistic Blending Module
==========================
Blends projected tiles with the original image using color matching,
brightness adjustment, and alpha blending for realistic results.

Key Features:
- Color tone matching
- Brightness/contrast adjustment
- Alpha blending with smooth transitions
- Shadow preservation
- Lighting adaptation
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class RealisticBlending:
    """
    Tools for realistic blending of tiles with original image
    """
    
    def __init__(self, original_image: np.ndarray, mask: np.ndarray):
        """
        Initialize realistic blending
        
        Args:
            original_image: Original room image (BGR)
            mask: Floor mask (binary, 0 or 255)
        """
        self.original_image = original_image.copy()
        self.mask = mask
        self.height, self.width = original_image.shape[:2]
        
        # Compute statistics of original floor region
        self._compute_floor_statistics()
        
        print("✅ Realistic blending initialized")
    
    def _compute_floor_statistics(self):
        """Compute color and brightness statistics of original floor"""
        # Extract floor region
        floor_region = cv2.bitwise_and(
            self.original_image,
            self.original_image,
            mask=self.mask
        )
        
        # Get non-zero pixels
        floor_pixels = floor_region[self.mask > 0]
        
        if len(floor_pixels) > 0:
            self.floor_mean_color = np.mean(floor_pixels, axis=0)
            self.floor_std_color = np.std(floor_pixels, axis=0)
            self.floor_mean_brightness = np.mean(cv2.cvtColor(
                floor_region, cv2.COLOR_BGR2GRAY
            )[self.mask > 0])
        else:
            self.floor_mean_color = np.array([128, 128, 128])
            self.floor_std_color = np.array([30, 30, 30])
            self.floor_mean_brightness = 128
        
        print(f"   📊 Floor mean color: {self.floor_mean_color}")
        print(f"   📊 Floor mean brightness: {self.floor_mean_brightness:.1f}")
    
    def match_brightness(self, 
                        tile_image: np.ndarray,
                        target_brightness: Optional[float] = None,
                        strength: float = 0.2) -> np.ndarray:
        """
        Match tile image brightness to original floor (with adjustable strength)
        
        Args:
            tile_image: Input tile image (BGR)
            target_brightness: Target brightness (if None, use floor brightness)
            strength: Strength of matching (0.0-1.0), default 0.2 to preserve tile colors
            
        Returns:
            Brightness-adjusted tile image
        """
        print(f"💡 Matching brightness (strength: {strength:.1%})...")
        
        if target_brightness is None:
            target_brightness = self.floor_mean_brightness
        
        # Convert to grayscale to compute current brightness
        tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        tile_brightness = np.mean(tile_gray[tile_gray > 0])
        
        if tile_brightness == 0:
            return tile_image
        
        # Compute brightness adjustment factor with strength control
        brightness_factor = 1.0 + (target_brightness / tile_brightness - 1.0) * strength
        
        # Apply adjustment
        adjusted = tile_image.astype(np.float32) * brightness_factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        print(f"   ✓ Brightness adjusted: {tile_brightness:.1f} → {tile_brightness * brightness_factor:.1f}")
        return adjusted
    
    def match_color_tone(self, 
                        tile_image: np.ndarray,
                        method: str = 'none',
                        strength: float = 0.3) -> np.ndarray:
        """
        Match tile color tone to original floor (OPTIONAL - use 'none' to preserve original colors)
        
        Args:
            tile_image: Input tile image (BGR)
            method: 'histogram', 'mean_std', or 'none' (default: 'none' to preserve colors)
            strength: Strength of color matching (0.0-1.0)
            
        Returns:
            Color-matched tile image
        """
        if method == 'none':
            print("🎨 Preserving original tile colors (no color matching)")
            return tile_image
            
        print(f"🎨 Matching color tone (method: {method}, strength: {strength:.1%})...")
        
        if method == 'histogram':
            matched = self._match_color_histogram(tile_image)
        elif method == 'mean_std':
            matched = self._match_color_mean_std(tile_image)
        else:
            print(f"   ⚠️ Unknown method: {method}")
            return tile_image
        
        # Blend matched with original based on strength
        result = cv2.addWeighted(tile_image, 1.0 - strength, matched, strength, 0)
        return result
    
    def _match_color_histogram(self, tile_image: np.ndarray) -> np.ndarray:
        """Match color using histogram matching"""
        # Extract floor region from original
        floor_region = cv2.bitwise_and(
            self.original_image,
            self.original_image,
            mask=self.mask
        )
        
        # Match histograms for each channel
        matched = tile_image.copy()
        
        for c in range(3):
            # Compute histograms
            source_hist, _ = np.histogram(
                tile_image[:, :, c].flatten(),
                bins=256,
                range=(0, 256)
            )
            template_hist, _ = np.histogram(
                floor_region[:, :, c][self.mask > 0],
                bins=256,
                range=(0, 256)
            )
            
            # Compute cumulative histograms
            source_cdf = source_hist.cumsum()
            source_cdf = source_cdf / source_cdf[-1]
            
            template_cdf = template_hist.cumsum()
            template_cdf = template_cdf / template_cdf[-1]
            
            # Create mapping
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # Find closest value in template CDF
                diff = np.abs(template_cdf - source_cdf[i])
                mapping[i] = np.argmin(diff)
            
            # Apply mapping
            matched[:, :, c] = mapping[tile_image[:, :, c]]
        
        print("   ✓ Histogram matching complete")
        return matched
    
    def _match_color_mean_std(self, tile_image: np.ndarray) -> np.ndarray:
        """Match color using mean and standard deviation"""
        # Compute tile statistics
        tile_pixels = tile_image[tile_image.sum(axis=2) > 0]
        
        if len(tile_pixels) == 0:
            return tile_image
        
        tile_mean = np.mean(tile_pixels, axis=0)
        tile_std = np.std(tile_pixels, axis=0)
        
        # Avoid division by zero
        tile_std = np.maximum(tile_std, 1.0)
        
        # Normalize and scale
        matched = tile_image.astype(np.float32)
        
        for c in range(3):
            matched[:, :, c] = (matched[:, :, c] - tile_mean[c]) / tile_std[c]
            matched[:, :, c] = matched[:, :, c] * self.floor_std_color[c] + self.floor_mean_color[c]
        
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        
        print("   ✓ Mean/std matching complete")
        return matched
    
    def extract_lighting_map(self) -> np.ndarray:
        """
        Extract lighting map from original floor region
        
        Returns:
            Lighting intensity map (0-1 float)
        """
        print("💡 Extracting lighting map...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to get smooth lighting
        lighting = cv2.GaussianBlur(gray, (51, 51), 0)
        
        # Normalize to 0-1
        lighting = lighting.astype(np.float32) / 255.0
        
        print("   ✓ Lighting map extracted")
        return lighting
    
    def apply_lighting(self, 
                      tile_image: np.ndarray,
                      lighting_map: np.ndarray,
                      strength: float = 0.2) -> np.ndarray:
        """
        Apply original lighting to tile image (reduced default strength)
        
        Args:
            tile_image: Input tile image (BGR)
            lighting_map: Lighting intensity map (0-1)
            strength: Strength of lighting effect (0-1), reduced to 0.2
            
        Returns:
            Tile image with lighting applied
        """
        print(f"💡 Applying lighting (strength: {strength:.1%})...")
        
        # Convert to float
        tile_float = tile_image.astype(np.float32) / 255.0
        
        # Apply lighting to each channel
        lighting_3d = np.stack([lighting_map] * 3, axis=2)
        
        # Blend with lighting
        lit_tile = tile_float * (1.0 - strength) + tile_float * lighting_3d * strength
        
        # Convert back to uint8
        lit_tile = np.clip(lit_tile * 255, 0, 255).astype(np.uint8)
        
        print("   ✓ Lighting applied")
        return lit_tile
    
    def alpha_blend(self,
                   tile_image: np.ndarray,
                   alpha: float = 0.8,
                   feather_size: int = 10) -> np.ndarray:
        """
        Blend tiles with original image using:
          - Multiplicative shadow transfer (room lighting preserved on tiles)
          - Unsharp mask to boost tile pattern visibility
          - Lossless non-floor pixel copy
        
        Args:
            tile_image: Tile image (BGR or BGRA — alpha channel used when present)
            alpha: Global alpha value (0-1)
            feather_size: Size of edge feathering in pixels
        
        Returns:
            Blended result (BGR)
        """
        print(f"🎨 Alpha blending with shadow+sharpening (alpha: {alpha:.2f})...")

        # ── 1. Separate tile BGR and alpha ────────────────────────────────────
        if tile_image.ndim == 3 and tile_image.shape[2] == 4:
            raw_alpha = tile_image[:, :, 3].astype(np.float64) / 255.0
            tile_bgr  = tile_image[:, :, :3].astype(np.float64)
        else:
            raw_alpha = self.mask.astype(np.float64) / 255.0
            tile_bgr  = tile_image.astype(np.float64)

        # ── 2. Feather edges ──────────────────────────────────────────────────
        if feather_size > 0:
            raw_alpha = self._feather_mask(
                raw_alpha.astype(np.float32), feather_size
            ).astype(np.float64)
        alpha_map = np.clip(raw_alpha * alpha, 0.0, 1.0)   # (H, W)

        original_f64 = self.original_image.astype(np.float64)
        floor_bool   = self.mask > 0                        # (H, W) bool

        # ── 3. Multiplicative shadow transfer (AMBIENT OCCLUSION) ────────────────
        # Extract greyscale brightness of original floor and use it as a
        # per-pixel multiplier so room shadows/highlights appear on the tile.
        orig_gray = cv2.cvtColor(
            self.original_image, cv2.COLOR_BGR2GRAY
        ).astype(np.float64)

        if np.any(floor_bool):
            avg_bright = float(orig_gray[floor_bool].mean())
        else:
            avg_bright = 128.0
        if avg_bright < 1.0:
            avg_bright = 128.0

        # Enhance shadow contrast (make darks darker)
        shadow_map = orig_gray / avg_bright
        # Use power function to deepen shadows slightly (gamma correction-like)
        shadow_map = np.power(shadow_map, 1.2) 
        
        shadow_map = np.clip(shadow_map, 0.2, 1.8)
        shadow_map = cv2.GaussianBlur(
            shadow_map.astype(np.float32), (21, 21), 0
        ).astype(np.float64)
        shadow_3d  = shadow_map[:, :, np.newaxis]           # broadcast over 3 ch

        # ── 4. Specular Reflection (GLOSS) ──────────────────────────────────────
        # Find bright spots in original image (specular highlights)
        # Threshold high brightness values
        specular_mask = np.maximum(0, orig_gray - 200) / 55.0  # Normalize 200-255 range to 0-1
        specular_mask = np.clip(specular_mask, 0, 1)
        specular_mask = cv2.GaussianBlur(specular_mask.astype(np.float32), (15, 15), 0)
        specular_3d = specular_mask[:, :, np.newaxis]

        # Apply shadow + Add reflection
        # Tile receives shadows *multiplicatively*
        tile_lit_f64 = tile_bgr * shadow_3d
        
        # Tile receives highlights *additively* (gloss)
        # INCREASED intensity from 0.15 to 0.5 to match the goal image's shine
        tile_lit_f64 = tile_lit_f64 + (230.0 * specular_3d * 0.5)
        
        tile_lit_f64 = np.clip(tile_lit_f64, 0.0, 255.0)

        # ── 5. Unsharp-mask on tile to pop the pattern ────────────────────────
        tile_lit_u8 = tile_lit_f64.astype(np.uint8)
        blur = cv2.GaussianBlur(tile_lit_u8, (0, 0), sigmaX=1.5)
        # Slightly softer sharpening to avoid halo artifacts
        tile_sharp_u8 = cv2.addWeighted(tile_lit_u8, 1.5, blur, -0.5, 0)

        # ── 6. Alpha-composite ─────────────────────────────────────────────────
        alpha_3d   = np.stack([alpha_map] * 3, axis=2)
        tile_f64   = tile_sharp_u8.astype(np.float64)
        blended_f64 = tile_f64 * alpha_3d + original_f64 * (1.0 - alpha_3d)
        blended_u8  = np.clip(blended_f64, 0.0, 255.0).astype(np.uint8)

        # ── 7. Lossless non-floor pixels (exact original bytes) ───────────────
        floor_3d = np.stack([floor_bool] * 3, axis=2)
        result   = np.where(floor_3d, blended_u8, self.original_image)

        print("   ✓ Blending complete (AO shadows + Glossy reflection + Sharpening applied)")
        return result
    
    def _feather_mask(self, mask: np.ndarray, feather_size: int) -> np.ndarray:
        """
        Apply feathering (soft edges) to mask
        
        Args:
            mask: Input mask (0-1 float)
            feather_size: Feathering size in pixels
            
        Returns:
            Feathered mask
        """
        # Compute distance transform
        mask_binary = (mask > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        
        # Normalize by feather size
        feathered = np.clip(dist_transform / feather_size, 0, 1)
        
        return feathered.astype(np.float32)
    
    def blend_complete(self,
                      tile_image: np.ndarray,
                      match_brightness: bool = False,  # Changed default to False
                      match_color: bool = False,       # Changed default to False
                      apply_lighting: bool = False,    # Changed default to False
                      alpha: float = 0.99,             # Changed default to 0.99
                      feather_size: int = 5) -> np.ndarray:  # Reduced default feather
        """
        Complete blending pipeline (MINIMAL adjustments for maximum tile quality)
        
        Args:
            tile_image: Input tile image (BGR or BGRA)
            match_brightness: Apply brightness matching (default: False for quality)
            match_color: Apply color matching (default: False for quality)
            apply_lighting: Apply lighting map (default: False for quality)
            alpha: Global alpha value (default: 0.99 for maximum tile visibility)
            feather_size: Edge feathering size (default: 5 for subtle edges)
            
        Returns:
            Final blended result
        """
        print("🎬 Starting QUALITY-PRESERVING blending pipeline...")

        # ── Separate channels so adjustments work on BGR only ─────────────────
        has_alpha = (tile_image.ndim == 3 and tile_image.shape[2] == 4)
        if has_alpha:
            tile_bgr   = tile_image[:, :, :3].copy()
            tile_alpha = tile_image[:, :, 3:4]   # keep for re-packing
        else:
            tile_bgr   = tile_image.copy()
            tile_alpha = None

        # Step 1: Match brightness (skip if disabled for quality)
        if match_brightness:
            tile_bgr = self.match_brightness(tile_bgr)
        else:
            print("⏭️ Skipping brightness matching - preserving original tile")

        # Step 2: Match color tone (skip if disabled for quality)
        if match_color:
            tile_bgr = self.match_color_tone(tile_bgr, method='mean_std')
        else:
            print("⏭️ Skipping color matching - preserving original tile")

        # Step 3: Apply lighting (skip if disabled for quality)
        if apply_lighting:
            lighting_map = self.extract_lighting_map()
            tile_bgr = self.apply_lighting(tile_bgr, lighting_map, strength=0.2)
        else:
            print("⏭️ Skipping lighting effects - preserving original tile")

        # Step 4: Re-pack alpha so alpha_blend gets the proper BGRA mask ──────
        if has_alpha:
            tile_for_blend = np.concatenate([tile_bgr, tile_alpha], axis=2)
        else:
            tile_for_blend = tile_bgr

        # Step 5: Alpha blend with shadow transfer + sharpening
        result = self.alpha_blend(tile_for_blend, alpha=alpha, feather_size=feather_size)

        print("✅ QUALITY-PRESERVING blending complete! Original tile appearance maintained.")
        return result
