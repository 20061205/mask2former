"""
Surface mask extraction and cleaning for each tileable surface type.
"""

import cv2
import numpy as np

from config import SURFACE_IDS, MASK_KERNEL_SIZE, MASK_CLOSE_ITERATIONS
from staircase_detector import separate_staircase_components


def extract_surface_mask(segmentation_map: np.ndarray, surface_name: str) -> np.ndarray:
    """
    Extract a binary mask (0/255) for a named surface.

    Parameters
    ----------
    segmentation_map : (H, W) integer class IDs
    surface_name     : key from SURFACE_IDS  (e.g. "floor", "wall", "stairway")

    Returns
    -------
    mask : np.ndarray (H, W) uint8, 0 or 255
    """
    if surface_name not in SURFACE_IDS:
        raise ValueError(
            f"Unknown surface '{surface_name}'. "
            f"Available: {list(SURFACE_IDS.keys())}"
        )
    label_id = SURFACE_IDS[surface_name]
    return (segmentation_map == label_id).astype(np.uint8) * 255


def clean_mask(mask: np.ndarray,
               kernel_size: int = MASK_KERNEL_SIZE,
               close_iterations: int = MASK_CLOSE_ITERATIONS) -> np.ndarray:
    """
    Morphological cleaning: close small holes, then dilate slightly
    to connect nearby regions.
    """
    m = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Only close small holes — do NOT dilate, to avoid bleeding onto objects
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return (m * 255).astype(np.uint8)


def extract_all_masks(segmentation_map: np.ndarray,
                      surface_names: list[str],
                      clean: bool = True) -> dict[str, np.ndarray]:
    """
    Extract and optionally clean masks for multiple surfaces.

    Returns
    -------
    dict  { surface_name: mask_uint8 }
    """
    masks = {}
    for name in surface_names:
        m = extract_surface_mask(segmentation_map, name)
        if clean:
            m = clean_mask(m)
        masks[name] = m
    return masks


def combine_masks(masks: dict[str, np.ndarray]) -> np.ndarray:
    """
    Merge multiple surface masks into one combined mask (OR).
    Useful when you want to tile multiple surfaces at once
    with the same tile pattern.
    """
    combined = None
    for m in masks.values():
        if combined is None:
            combined = m.copy()
        else:
            combined = np.maximum(combined, m)
    return combined


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Return the non-surface mask (everything that is NOT the surface)."""
    return 255 - mask


def process_staircase_surface(segmentation_map: np.ndarray) -> dict[str, np.ndarray]:
    """
    Process staircase (screen) surface to separate treads and risers.
    
    Parameters
    ----------
    segmentation_map : np.ndarray
        Segmentation map with class IDs
        
    Returns
    -------
    staircase_surfaces : dict
        Dictionary with individual tread and riser masks
    """
    # Extract screen mask (which actually represents stairways)
    screen_mask = extract_surface_mask(segmentation_map, "screen")
    
    if np.sum(screen_mask > 0) == 0:
        return {}
    
    # Clean the mask
    screen_mask = clean_mask(screen_mask)
    
    # Separate into individual steps
    stair_steps = separate_staircase_components(screen_mask)
    
    if not stair_steps:
        return {}
    
    # Create surface dictionary
    staircase_surfaces = {}
    for i, step in enumerate(stair_steps):
        if 'tread' in step and np.sum(step['tread'] > 0) > 0:
            staircase_surfaces[f'stair_step_{i}_tread'] = step['tread']
        if 'riser' in step and np.sum(step['riser'] > 0) > 0:
            staircase_surfaces[f'stair_step_{i}_riser'] = step['riser']
    
    return staircase_surfaces


def extract_all_masks_with_staircase(segmentation_map: np.ndarray,
                                   surface_names: list[str],
                                   clean: bool = True) -> dict[str, np.ndarray]:
    """
    Extract masks for multiple surfaces, including special staircase processing.
    
    Parameters
    ----------
    segmentation_map : np.ndarray
        Segmentation map with class IDs
    surface_names : list[str]
        List of surface names to extract
    clean : bool
        Whether to apply morphological cleaning
        
    Returns
    -------
    masks : dict
        Dictionary mapping surface names to masks
    """
    masks = {}
    
    for name in surface_names:
        if name == 'screen':
            # Special processing for staircase
            staircase_masks = process_staircase_surface(segmentation_map)
            masks.update(staircase_masks)
        else:
            # Regular surface extraction
            m = extract_surface_mask(segmentation_map, name)
            if clean:
                m = clean_mask(m)
            masks[name] = m
    
    return masks
