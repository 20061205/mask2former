"""
Tile Projection Engine
=======================
Projects tile textures onto the detected floor plane using 
perspective transformation and geometric warping.

Key Features:
- Load and prepare tile textures
- Generate repeating tile grids
- Warp tiles into floor plane perspective
- Clip warped tiles using irregular masks
- Support for tile scale adjustment
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class TileProjectionEngine:
    """
    Engine for projecting tile textures onto floor plane
    """
    
    def __init__(self, 
                 tile_texture: np.ndarray,
                 mask: np.ndarray,
                 quad_points: np.ndarray):
        """
        Initialize tile projection engine
        
        Args:
            tile_texture: Tile texture image (BGR)
            mask: Floor mask for clipping (binary, 0 or 255)
            quad_points: 4 corner points for perspective (4 x 2 array)
        """
        self.tile_texture = tile_texture
        self.mask = mask
        self.quad_points = quad_points.astype(np.float32)
        
        self.height, self.width = mask.shape[:2]
        
        # PROFESSIONAL APPROACH: Calculate real floor dimensions
        floor_area_pixels = np.count_nonzero(mask)
        floor_width_px = np.max(quad_points[:, 0]) - np.min(quad_points[:, 0])
        floor_height_px = np.max(quad_points[:, 1]) - np.min(quad_points[:, 1])
        
        # Assume standard room: 1 pixel = ~0.5 cm in real world
        # This gives us realistic floor dimensions
        self.pixels_per_cm = 2.0  # 2 pixels = 1 cm
        self.floor_width_cm = floor_width_px / self.pixels_per_cm
        self.floor_height_cm = floor_height_px / self.pixels_per_cm
        
        # Real-world tile sizes in cm (common sizes)
        # 30x30cm, 40x40cm, 45x45cm, 60x60cm are standard
        self.tile_size_cm = 45.0  # Default: 45cm x 45cm tiles (18 inches)
        
        # Grout/spacing between tiles (typically 2-5mm in real world)
        self.grout_width_cm = 0.3  # 3mm grout line
        self.grout_color = (220, 220, 220)  # Light gray grout
        
        # Tile installation parameters
        self.start_corner = 'bottom-left'  # Start installation from here
        self.cut_threshold = 0.3  # Cut tiles smaller than 30% are replaced
        
        print("✅ Professional tile installation engine initialized")
        print(f"   Floor dimensions: {self.floor_width_cm:.1f}cm x {self.floor_height_cm:.1f}cm")
        print(f"   Floor area: {(self.floor_width_cm * self.floor_height_cm / 10000):.2f} square meters")
        print(f"   Tile size: {self.tile_size_cm:.0f}cm x {self.tile_size_cm:.0f}cm")
        print(f"   Grout width: {self.grout_width_cm * 10:.1f}mm")
    
    def set_tile_size(self, size_cm: float):
        """
        Set real-world tile size in centimeters (PROFESSIONAL CONTROL)
        
        Args:
            size_cm: Tile size in cm (common: 30, 40, 45, 60, 80)
        """
        self.tile_size_cm = max(20.0, min(120.0, size_cm))
        tiles_across = self.floor_width_cm / (self.tile_size_cm + self.grout_width_cm)
        tiles_down = self.floor_height_cm / (self.tile_size_cm + self.grout_width_cm)
        total_tiles = int(tiles_across) * int(tiles_down)
        print(f"🔲 Tile size set to: {self.tile_size_cm:.0f}cm x {self.tile_size_cm:.0f}cm")
        print(f"   Estimated tiles needed: ~{total_tiles} tiles ({tiles_across:.1f} across x {tiles_down:.1f} down)")
    
    def set_grout_width(self, width_mm: float):
        """
        Set grout line width in millimeters
        
        Args:
            width_mm: Grout width in mm (typical: 2-5mm)
        """
        self.grout_width_cm = width_mm / 10.0
        print(f"🎨 Grout width set to: {width_mm:.1f}mm")
    
    def get_tile_count(self) -> Tuple[int, int]:
        """
        Calculate how many tiles fit on the floor
        
        Returns:
            Tuple of (tiles_across, tiles_down)
        """
        tile_plus_grout = self.tile_size_cm + self.grout_width_cm
        tiles_across = int(np.ceil(self.floor_width_cm / tile_plus_grout))
        tiles_down = int(np.ceil(self.floor_height_cm / tile_plus_grout))
        return tiles_across, tiles_down
    
    def set_grout_color(self, color: Tuple[int, int, int]):
        """
        Set grout color
        
        Args:
            color: Grout color (B, G, R)
        """
        self.grout_color = color
        print(f"🎨 Grout color set to: RGB{color}")
    
    def install_tiles_professionally(self) -> np.ndarray:
        """
        Generate a repeating tile grid in top-down view
        
        Args:
            grid_width: Width of grid
            grid_height: Height of grid
            tile_size: Size of each tile in pixels (already scaled)
            
        Returns:
            Tiled image (BGR)
        """
        print("🔲 Generating tile grid...")
        print(f"   Grid size: {grid_width}x{grid_height}, Tile size: {tile_size}px")
        
        # Ensure tile size is large enough to show pattern detail
        # Professional minimum: 200px for simple patterns, 400+ for detailed patterns
        min_tile_size = 400
        if tile_size < min_tile_size:
            print(f"   ⚠️ Tile size {tile_size}px too small, using minimum {min_tile_size}px for pattern clarity")
            tile_size = min_tile_size
        
        # Resize tile texture with HIGHEST QUALITY interpolation
        # For detailed patterns, preserve as much detail as possible
        tile_resized = cv2.resize(
            self.tile_texture,
            (tile_size - self.grout_width, tile_size - self.grout_width),
            interpolation=cv2.INTER_LANCZOS4  # Highest quality interpolation
        )
        
        print(f"   ✅ Tile resized to: {tile_resized.shape[1]}x{tile_resized.shape[0]} (preserving detail)")
        
        # Create grid
        num_tiles_x = (grid_width + tile_size - 1) // tile_size + 1
        num_tiles_y = (grid_height + tile_size - 1) // tile_size + 1
        
        # Create canvas with grout color
        grid = np.full(
            (num_tiles_y * tile_size, num_tiles_x * tile_size, 3),
            self.grout_color,
            dtype=np.uint8
        )
        
        # Place tiles
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                y_start = i * tile_size
                x_start = j * tile_size
                
                y_end = y_start + tile_size - self.grout_width
                x_end = x_start + tile_size - self.grout_width
                
                if y_end <= grid.shape[0] and x_end <= grid.shape[1]:
                    grid[y_start:y_end, x_start:x_end] = tile_resized
        
        # Crop to exact size
        grid = grid[:grid_height, :grid_width]
        
        print(f"   ✓ Grid generated: {grid.shape[1]}x{grid.shape[0]}, {num_tiles_x * num_tiles_y} tiles")
        return grid
    
    def warp_tiles_to_perspective(self, 
                                  tile_grid: np.ndarray,
                                  inverse_matrix: np.ndarray) -> np.ndarray:
        """
        Warp tile grid from top-down to image perspective
        
        Args:
            tile_grid: Tile grid in top-down view
            inverse_matrix: Inverse perspective transformation matrix
            
        Returns:
            Warped tile image
        """
        print("🔄 Warping tiles to perspective...")
        
        # Warp tile grid to image perspective with HIGHEST QUALITY interpolation
        warped = cv2.warpPerspective(
            tile_grid,
            inverse_matrix,
            (self.width, self.height),
            flags=cv2.INTER_LANCZOS4,  # Highest quality interpolation for best results
            borderMode=cv2.BORDER_CONSTANT
        )
        
        print("   ✓ Tiles warped")
        return warped
    
    def clip_with_mask(self, warped_tiles: np.ndarray) -> np.ndarray:
        """
        Clip warped tiles using the floor mask
        
        Args:
            warped_tiles: Warped tile image
            
        Returns:
            Clipped tile image (with transparency)
        """
        print("✂️ Clipping tiles with mask...")
        
        # Convert to BGRA (add alpha channel)
        if warped_tiles.shape[2] == 3:
            warped_tiles_alpha = cv2.cvtColor(warped_tiles, cv2.COLOR_BGR2BGRA)
        else:
            warped_tiles_alpha = warped_tiles.copy()
        
        # Set alpha channel based on mask
        warped_tiles_alpha[:, :, 3] = self.mask
        
        print("   ✓ Tiles clipped")
        return warped_tiles_alpha
    
    def project_tiles_full(self, 
                          grid_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete tile projection pipeline
        
        Args:
            grid_size: Size of tile grid to generate
            
        Returns:
            Tuple of (warped_tiles, warped_tiles_clipped)
        """
        print("🎬 Starting full tile projection...")
        
        # Step 1: Calculate optimal tile size (PROFESSIONAL APPROACH)
        # Formula: grid_size / optimal_tile_count = tile size in grid space
        # For detailed floral patterns, we need 6-10 tiles across the floor
        base_tile_size = grid_size // self.optimal_tile_count
        
        # Apply user scaling
        tile_size = int(base_tile_size * self.tile_scale)
        
        # Ensure minimum size for pattern visibility
        tile_size = max(tile_size, 500)  # Professional minimum: 500px for detailed patterns
        
        print(f"   📊 Professional tile sizing:")
        print(f"      Grid: {grid_size}x{grid_size}px")
        print(f"      Target tiles across: {self.optimal_tile_count}")
        print(f"      Calculated tile size: {tile_size}px")
        print(f"      Scale factor: {self.tile_scale:.2f}")
        
        tile_grid = self.generate_tile_grid(grid_size, grid_size, tile_size)
        
        # Step 2: Compute inverse perspective transform
        dst_points = np.array([
            [0, 0],
            [grid_size - 1, 0],
            [grid_size - 1, grid_size - 1],
            [0, grid_size - 1]
        ], dtype=np.float32)
        
        forward_matrix = cv2.getPerspectiveTransform(self.quad_points, dst_points)
        inverse_matrix = np.linalg.inv(forward_matrix)
        
        # Step 3: Warp to perspective
        warped_tiles = self.warp_tiles_to_perspective(tile_grid, inverse_matrix)
        
        # Step 4: Clip with mask
        warped_tiles_clipped = self.clip_with_mask(warped_tiles)
        
        print("✅ Tile projection complete!")
        return warped_tiles, warped_tiles_clipped
    
    def add_tile_variation(self, tile_grid: np.ndarray, 
                          variation_amount: float = 0.1) -> np.ndarray:
        """
        Add subtle brightness variation to tiles for realism
        
        Args:
            tile_grid: Input tile grid
            variation_amount: Amount of variation (0-1)
            
        Returns:
            Tile grid with variation
        """
        print("✨ Adding tile variation...")
        
        # Convert to float
        tile_grid_float = tile_grid.astype(np.float32)
        
        # Generate random variation pattern
        variation = np.random.randn(tile_grid.shape[0], tile_grid.shape[1]) * variation_amount
        variation = cv2.GaussianBlur(variation, (51, 51), 0)
        
        # Apply variation
        for c in range(3):
            tile_grid_float[:, :, c] *= (1.0 + variation)
        
        # Clip and convert back
        tile_grid_varied = np.clip(tile_grid_float, 0, 255).astype(np.uint8)
        
        print("   ✓ Variation added")
        return tile_grid_varied
    
    def add_perspective_distortion(self, 
                                   tile_grid: np.ndarray,
                                   strength: float = 0.1) -> np.ndarray:
        """
        Add subtle perspective distortion to tile grid for more realism
        
        Args:
            tile_grid: Input tile grid
            strength: Strength of distortion (0-1)
            
        Returns:
            Distorted tile grid
        """
        h, w = tile_grid.shape[:2]
        
        # Create subtle distortion map
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                # Perspective scaling factor
                scale = 1.0 + strength * (i / h)
                
                map_x[i, j] = j * scale
                map_y[i, j] = i
        
        # Apply distortion
        distorted = cv2.remap(
            tile_grid,
            map_x,
            map_y,
            cv2.INTER_LINEAR
        )
        
        return distorted
    
    def visualize_tile_grid(self, tile_grid: np.ndarray) -> np.ndarray:
        """
        Create visualization of tile grid
        
        Args:
            tile_grid: Tile grid to visualize
            
        Returns:
            Visualization image
        """
        # Resize for display if too large
        max_display_size = 800
        h, w = tile_grid.shape[:2]
        
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_grid = cv2.resize(tile_grid, (new_w, new_h))
        else:
            display_grid = tile_grid.copy()
        
        return display_grid
