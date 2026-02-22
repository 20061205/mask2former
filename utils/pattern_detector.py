import cv2
import numpy as np
from typing import Tuple, List, Optional

class PatternDetector:
    """
    Intelligently detects the optimal tiling pattern for a given tile image.
    Analyzes edge continuity to determine if the tile is a "quadrant" of a larger
    pattern (e.g. circles) or a standalone standard tile.
    """
    
    def __init__(self, tile_image: np.ndarray):
        self.original_tile = tile_image
        self.height, self.width = tile_image.shape[:2]
        
    def analyze_optimal_pattern(self) -> Tuple[np.ndarray, str]:
        """
        Test different tiling arrangements (Standard, Rotated, Mirrored)
        and return the seamless 2x2 macro-tile that forms the best pattern.
        """
        print("🔍 Analysing tile pattern logic...")
        
        # Define candidate patterns (methods to arrange 2x2 grid)
        candidates = {
            'standard': self._create_2x2_standard,
            'rotated_quadrant': self._create_2x2_rotated,   # 0, 90, 270, 180 (Circular formation)
            'mirrored': self._create_2x2_mirrored,          # Mirror horizontal/vertical
            'diamond': self._create_2x2_diamond             # 45-degree illusion (simple check)
        }
        
        best_score = float('inf')
        best_pattern_name = 'standard'
        best_composite = None
        
        # Test each pattern
        for name, method in candidates.items():
            composite = method()
            score = self._calculate_seam_error(composite)
            
            print(f"   - Pattern '{name}': Continuity Error = {score:.2f}")
            
            if score < best_score:
                best_score = score
                best_pattern_name = name
                best_composite = composite
        
        # Bias towards standard if improvement is marginal (simpler is better)
        # If rotated/mirrored isn't significantly better (e.g. < 20% better), stick to standard
        standard_score = self._calculate_seam_error(self._create_2x2_standard())
        if best_pattern_name != 'standard' and best_score > standard_score * 0.8:
            print(f"   ℹ️  '{best_pattern_name}' wasn't significantly better than standard. Reverting.")
            best_pattern_name = 'standard'
            best_composite = self._create_2x2_standard()
            
        print(f"✅ Detected optimal pattern: {best_pattern_name.upper()}")
        
        if best_pattern_name != 'standard':
            print(f"   ✨ Constructing seamless macro-tile ({best_composite.shape[1]}x{best_composite.shape[0]})")
            return best_composite, best_pattern_name
        else:
            return self.original_tile, 'standard' # Return original single tile
    
    def _create_2x2_standard(self):
        """Standard grid (abab)"""
        top_row = np.hstack([self.original_tile, self.original_tile])
        return np.vstack([top_row, top_row])
        
    def _create_2x2_rotated(self):
        """
        Pinwheel / Circle former (Radial symmetry):
        Try to form a continuous pattern at the CENTER.
        
        Assumes the 'interesting' part of the tile is in one corner.
        Tests ALL 4 rotations to find which one forms the center match.
        """
        # We don't know which corner the pattern is in (TL, TR, BL, BR).
        # So we try 4 variations of "radial symmetry" and pick the best one.
        
        candidates = []
        
        # Variation A: Simple rotation sequence 0, 90, 180, 270
        tl = self.original_tile
        tr = cv2.rotate(self.original_tile, cv2.ROTATE_90_CLOCKWISE)
        br = cv2.rotate(self.original_tile, cv2.ROTATE_180)
        bl = cv2.rotate(self.original_tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
        candidates.append(np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])]))
        
        # Variation B: Inward facing (mirror-like rotation)
        # If the pattern is in Top-Left, we need:
        # TL: Rot 180 (puts TL->BR)
        # TR: Rot 270 (puts TL->BL)
        # BL: Rot 90  (puts TL->TR)
        # BR: Rot 0   (puts TL->TL) -- wait, BR needs TL at TL? No, BR needs pattern at TL. 
        # So if source has pattern at Top-Left:
        # TL pos needs pattern at BR -> Rot 180
        # TR pos needs pattern at BL -> Rot 270
        # BL pos needs pattern at TR -> Rot 90
        # BR pos needs pattern at TL -> Rot 0
        
        r0 = self.original_tile
        r90 = cv2.rotate(self.original_tile, cv2.ROTATE_90_CLOCKWISE)
        r180 = cv2.rotate(self.original_tile, cv2.ROTATE_180)
        r270 = cv2.rotate(self.original_tile, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Case 1: Pattern is Top-Left
        c1_tl = r180
        c1_tr = r270
        c1_bl = r90
        c1_br = r0
        candidates.append(np.vstack([np.hstack([c1_tl, c1_tr]), np.hstack([c1_bl, c1_br])]))
        
        # Case 2: Pattern is Top-Right
        # TL needs pattern at BR -> Rot 270
        # TR needs pattern at BL -> Rot 180
        # BL needs pattern at TR -> Rot 0
        # BR needs pattern at TL -> Rot 90
        c2_tl = r270
        c2_tr = r180
        c2_bl = r0
        c2_br = r90
        candidates.append(np.vstack([np.hstack([c2_tl, c2_tr]), np.hstack([c2_bl, c2_br])]))
        
        # Case 3: Pattern is Bottom-Right (Standard Quarter Circle tile usually)
        # TL needs pattern at BR -> Rot 0
        # TR needs pattern at BL -> Rot 90
        # BL needs pattern at TR -> Rot 270
        # BR needs pattern at TL -> Rot 180
        c3_tl = r0
        c3_tr = r90
        c3_bl = r270
        c3_br = r180
        candidates.append(np.vstack([np.hstack([c3_tl, c3_tr]), np.hstack([c3_bl, c3_br])]))

         # Case 4: Pattern is Bottom-Left
        # TL needs pattern at BR -> Rot 90
        # TR needs pattern at BL -> Rot 0
        # BL needs pattern at TR -> Rot 180
        # BR needs pattern at TL -> Rot 270
        c4_tl = r90
        c4_tr = r0
        c4_bl = r180
        c4_br = r270
        candidates.append(np.vstack([np.hstack([c4_tl, c4_tr]), np.hstack([c4_bl, c4_br])]))

        # Return the candidate with best seam score
        best_cand = candidates[0]
        best_score = float('inf')
        
        for cand in candidates:
            score = self._calculate_seam_error(cand)
            if score < best_score:
                best_score = score
                best_cand = cand
                
        return best_cand
    
    def _create_2x2_mirrored(self):
        """
        Kaleidoscope / Bookmatch:
        TL(0)   TR(FlipH)
        BL(FlipV) BR(FlipHV)
        """
        tl = self.original_tile
        tr = cv2.flip(self.original_tile, 1) # Horizontal
        bl = cv2.flip(self.original_tile, 0) # Vertical
        br = cv2.flip(tr, 0)                 # Both
        
        top = np.hstack([tl, tr])
        bottom = np.hstack([bl, br])
        return np.vstack([top, bottom])
        
    def _create_2x2_diamond(self):
        # Placeholder - Diamond is hard to simulate with just 2x2 pixel grid without rotation/crop
        # Returns standard for now
        return self._create_2x2_standard()

    def _calculate_seam_error(self, composite: np.ndarray) -> float:
        """
        Calculate pixel discontinuity across the internal seams of the 2x2 composite.
        Lower is better.
        """
        h, w = composite.shape[:2]
        cy, cx = h // 2, w // 2
        
        # Extract the pixels at the seam boundaries
        # Vertical Seam (middle column)
        # Compare column just left of center vs column at center
        col_left = composite[:, cx-1, :]
        col_right = composite[:, cx, :]
        
        # Horizontal Seam (middle row)
        # Compare row just above center vs row at center
        row_top = composite[cy-1, :, :]
        row_bottom = composite[cy, :, :]
        
        # Calculate mean absolute difference
        v_diff = np.mean(np.abs(col_left.astype(float) - col_right.astype(float)))
        h_diff = np.mean(np.abs(row_top.astype(float) - row_bottom.astype(float)))
        
        # WEIGHTING: The center point (where all 4 meet) is critical for patterns like circles.
        # Errors near the center should be penalized more than errors at the outer edges.
        
        # Crop center region (central 20%)
        center_margin = int(min(h, w) * 0.1)
        if center_margin > 0:
            center_v_diff = np.mean(np.abs(
                col_left[cy-center_margin:cy+center_margin, :].astype(float) - 
                col_right[cy-center_margin:cy+center_margin, :].astype(float)
            ))
            center_h_diff = np.mean(np.abs(
                row_top[:, cx-center_margin:cx+center_margin].astype(float) - 
                row_bottom[:, cx-center_margin:cx+center_margin].astype(float)
            ))
            
            # Total score = Average diff + 2 * Center diff (Penalize center breaks)
            return (v_diff + h_diff) + 2.0 * (center_v_diff + center_h_diff)
            
        return v_diff + h_diff
