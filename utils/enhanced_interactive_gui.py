"""
Enhanced Interactive Floor Correction GUI
==========================================
Advanced GUI integrating all smart selection tools:
- Photoshop-like interface
- Multiple selection modes
- Intelligent edge snapping
- Real-time preview
- Undo/redo with visual history
- Tool shortcuts and hints

This is the main interactive interface users will use to correct floor detection.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from enum import Enum
import sys
import os

# Import our smart tools
from .intelligent_edge_snapping import IntelligentEdgeSnapper
from .smart_selection_tools import SmartSelectionTools


class SelectionMode(Enum):
    """Selection tool modes"""
    BRUSH = 1           # Manual brush painting
    MAGIC_WAND = 2      # Click to select similar colors
    QUICK_SELECT = 3    # Paint to quick select
    SCISSORS = 4        # Intelligent scissors
    OBJECT_SELECT = 5   # Object selection
    COLOR_RANGE = 6     # Color range selection


class EnhancedFloorCorrectionGUI:
    """
    Advanced interactive GUI for floor mask correction
    Integrates all smart selection tools with Photoshop-like interface
    """
    
    def __init__(self, image: np.ndarray, initial_mask: Optional[np.ndarray] = None):
        """
        Initialize enhanced GUI
        
        Args:
            image: Original image (BGR)
            initial_mask: Initial floor detection mask
        """
        self.image = image.copy()
        self.original_image = image.copy()
        self.height, self.width = image.shape[:2]
        
        # Initialize mask
        if initial_mask is None:
            self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            self.mask = initial_mask.copy()
        
        # Initialize smart tools
        print("🔧 Initializing smart selection tools...")
        self.edge_snapper = IntelligentEdgeSnapper(image)
        self.selection_tools = SmartSelectionTools(image, self.edge_snapper.edge_map)
        
        # Tool state
        self.current_mode = SelectionMode.BRUSH
        self.brush_size = 20
        self.tolerance = 30
        self.is_adding = True  # True = add to selection, False = remove
        
        # Drawing state
        self.drawing = False
        self.last_point = None
        self.temp_scribbles = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # For scissors tool
        self.scissors_points = []
        self.scissors_temp_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # History for undo/redo
        self.history = [self.mask.copy()]
        self.history_index = 0
        self.max_history = 30
        
        # Display settings
        self.show_edges = False
        self.show_instructions = True
        self.overlay_alpha = 0.6
        
        # Window setup
        self.window_name = "Enhanced Floor Correction"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Create control panel
        self._create_trackbars()
        
        print("✅ Enhanced GUI ready!")
    
    def _create_trackbars(self):
        """Create control trackbars"""
        cv2.createTrackbar('Brush Size', self.window_name, 20, 100, 
                          lambda x: setattr(self, 'brush_size', max(1, x)))
        cv2.createTrackbar('Tolerance', self.window_name, 30, 100,
                          lambda x: setattr(self, 'tolerance', x))
        cv2.createTrackbar('Alpha', self.window_name, 60, 100,
                          lambda x: setattr(self, 'overlay_alpha', x / 100.0))
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events based on current tool mode"""
        
        if self.current_mode == SelectionMode.BRUSH:
            self._handle_brush_mouse(event, x, y, flags)
        
        elif self.current_mode == SelectionMode.MAGIC_WAND:
            self._handle_magic_wand_mouse(event, x, y, flags)
        
        elif self.current_mode == SelectionMode.QUICK_SELECT:
            self._handle_quick_select_mouse(event, x, y, flags)
        
        elif self.current_mode == SelectionMode.SCISSORS:
            self._handle_scissors_mouse(event, x, y, flags)
        
        elif self.current_mode == SelectionMode.OBJECT_SELECT:
            self._handle_object_select_mouse(event, x, y, flags)
        
        elif self.current_mode == SelectionMode.COLOR_RANGE:
            self._handle_color_range_mouse(event, x, y, flags)
    
    def _handle_brush_mouse(self, event, x, y, flags):
        """Handle brush tool mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
            self.is_adding = (event == cv2.EVENT_LBUTTONDOWN)
            
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            self.last_point = None
            if np.any(self.temp_scribbles > 0):
                self._apply_scribbles()
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.last_point:
                # Draw line with edge snapping
                corrected_points = self.edge_snapper.correct_drawn_line(
                    self.last_point, (x, y), num_samples=20
                )
                
                for point in corrected_points:
                    cv2.circle(self.temp_scribbles, point, self.brush_size, 255, -1)
            
            self.last_point = (x, y)
    
    def _handle_magic_wand_mouse(self, event, x, y, flags):
        """Handle magic wand tool"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🪄 Magic wand at ({x}, {y})...")
            
            # Use smart selection
            selection = self.selection_tools.magic_wand_select(
                (x, y),
                tolerance=self.tolerance,
                use_contiguous=(flags & cv2.EVENT_FLAG_CTRLKEY == 0)
            )
            
            # Add or remove from mask
            if flags & cv2.EVENT_FLAG_SHIFTKEY:  # Shift = add
                self.mask = cv2.bitwise_or(self.mask, selection)
            elif flags & cv2.EVENT_FLAG_ALTKEY:  # Alt = subtract
                self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(selection))
            else:  # Replace
                self.mask = selection
            
            self._save_history()
            print("✅ Selection applied")
    
    def _handle_quick_select_mouse(self, event, x, y, flags):
        """Handle quick selection tool"""
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
            self.is_adding = (event == cv2.EVENT_LBUTTONDOWN)
            
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            if np.any(self.temp_scribbles > 0):
                print("🖌️ Applying quick selection...")
                
                # Use smart quick selection
                result = self.selection_tools.quick_selection_brush(
                    self.temp_scribbles,
                    is_foreground=self.is_adding,
                    refinement_iterations=3
                )
                
                # Update mask
                if self.is_adding:
                    self.mask = cv2.bitwise_or(self.mask, result)
                else:
                    self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(result))
                
                self.temp_scribbles = np.zeros_like(self.temp_scribbles)
                self._save_history()
                print("✅ Quick selection applied")
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.last_point:
                cv2.line(self.temp_scribbles, self.last_point, (x, y), 255, self.brush_size)
            self.last_point = (x, y)
    
    def _handle_scissors_mouse(self, event, x, y, flags):
        """Handle intelligent scissors tool"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add control point
            self.scissors_points.append((x, y))
            print(f"✂️ Scissors point {len(self.scissors_points)}: ({x}, {y})")
            
            # If we have at least 2 points, show preview
            if len(self.scissors_points) >= 2:
                self.scissors_temp_mask = self.selection_tools.intelligent_scissors(
                    self.scissors_points,
                    closed_path=False
                )
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click to complete and fill
            if len(self.scissors_points) >= 3:
                print("✂️ Completing scissors selection...")
                
                # Create closed path
                final_mask = self.selection_tools.intelligent_scissors(
                    self.scissors_points,
                    closed_path=True
                )
                
                # Add to mask
                self.mask = cv2.bitwise_or(self.mask, final_mask)
                self._save_history()
                
                # Reset
                self.scissors_points = []
                self.scissors_temp_mask = np.zeros_like(self.scissors_temp_mask)
                print("✅ Scissors selection applied")
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle-click to cancel
            self.scissors_points = []
            self.scissors_temp_mask = np.zeros_like(self.scissors_temp_mask)
            print("❌ Scissors cancelled")
    
    def _handle_object_select_mouse(self, event, x, y, flags):
        """Handle object selection tool"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🎯 Object select at ({x}, {y})...")
            
            # Collect multiple points in a region
            positive_points = [(x, y)]
            
            # Use object selection
            result = self.selection_tools.object_select_tool(
                positive_points,
                negative_points=[]
            )
            
            # Add to mask
            self.mask = cv2.bitwise_or(self.mask, result)
            self._save_history()
            print("✅ Object selected")
    
    def _handle_color_range_mouse(self, event, x, y, flags):
        """Handle color range selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🎨 Color range select at ({x}, {y})...")
            
            fuzziness = self.tolerance
            result = self.selection_tools.color_range_select(
                (x, y),
                fuzziness=fuzziness,
                use_lab_space=True
            )
            
            # Add or replace
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.mask = cv2.bitwise_or(self.mask, result)
            else:
                self.mask = result
            
            self._save_history()
            print("✅ Color range selected")
    
    def _apply_scribbles(self):
        """Apply temporary scribbles to mask"""
        if self.is_adding:
            self.mask = cv2.bitwise_or(self.mask, self.temp_scribbles)
        else:
            self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(self.temp_scribbles))
        
        self.temp_scribbles = np.zeros_like(self.temp_scribbles)
        self._save_history()
    
    def _save_history(self):
        """Save current state to history"""
        self.history = self.history[:self.history_index + 1]
        self.history.append(self.mask.copy())
        self.history_index += 1
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
    
    def _undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.mask = self.history[self.history_index].copy()
            print("↶ Undo")
    
    def _redo(self):
        """Redo last action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.mask = self.history[self.history_index].copy()
            print("↷ Redo")
    
    def _get_visualization(self) -> np.ndarray:
        """Generate visualization for display"""
        # Start with original image
        display = self.original_image.copy()
        
        # Show edge map if enabled
        if self.show_edges:
            edge_overlay = np.zeros_like(display)
            edge_overlay[self.edge_snapper.edge_map > 0] = [0, 255, 255]
            display = cv2.addWeighted(display, 0.7, edge_overlay, 0.3, 0)
        
        # Overlay mask
        mask_overlay = display.copy()
        mask_overlay[self.mask > 0] = [0, 255, 0]  # Green for floor
        display = cv2.addWeighted(display, 1 - self.overlay_alpha, mask_overlay, self.overlay_alpha, 0)
        
        # Show temporary scribbles
        display[self.temp_scribbles > 0] = [255, 255, 0]  # Yellow
        
        # Show scissors points and preview
        if self.current_mode == SelectionMode.SCISSORS:
            for i, point in enumerate(self.scissors_points):
                cv2.circle(display, point, 5, (0, 0, 255), -1)
                cv2.putText(display, str(i+1), (point[0]+10, point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if np.any(self.scissors_temp_mask > 0):
                display[self.scissors_temp_mask > 0] = [255, 0, 255]  # Magenta preview
        
        # Add UI overlays
        if self.show_instructions:
            display = self._add_instructions(display)
        
        display = self._add_status_bar(display)
        display = self._add_tool_info(display)
        
        return display
    
    def _add_instructions(self, image: np.ndarray) -> np.ndarray:
        """Add instructions overlay"""
        h, w = image.shape[:2]
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (500, 450), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.75, image, 0.25, 0)
        
        # Instructions
        tool_names = {
            SelectionMode.BRUSH: "Brush",
            SelectionMode.MAGIC_WAND: "Magic Wand",
            SelectionMode.QUICK_SELECT: "Quick Selection",
            SelectionMode.SCISSORS: "Intelligent Scissors",
            SelectionMode.OBJECT_SELECT: "Object Select",
            SelectionMode.COLOR_RANGE: "Color Range"
        }
        
        instructions = [
            "ENHANCED FLOOR CORRECTION",
            "",
            f"Current Tool: {tool_names[self.current_mode]}",
            "",
            "SELECTION TOOLS:",
            "  1: Brush (manual paint)",
            "  2: Magic Wand (click similar)",
            "  3: Quick Selection (paint smart)",
            "  4: Intelligent Scissors (click path)",
            "  5: Object Select (click object)",
            "  6: Color Range (select color)",
            "",
            "MOUSE:",
            "  Left: Add to selection",
            "  Right: Remove from selection",
            "  Shift+Click: Add to existing",
            "  Alt+Click: Subtract from existing",
            "",
            "KEYBOARD:",
            "  Z: Undo | Y: Redo",
            "  G: Grow | S: Shrink",
            "  M: Smooth | F: Feather",
            "  E: Toggle edges | I: Toggle help",
            "  SPACE: Accept | ESC: Cancel"
        ]
        
        y = 30
        for line in instructions:
            if line.startswith("ENHANCED") or line.startswith("SELECTION") or line.startswith("MOUSE") or line.startswith("KEYBOARD"):
                color = (0, 255, 255)
                font_scale = 0.5
                thickness = 2
            elif line.startswith("Current"):
                color = (0, 255, 0)
                font_scale = 0.5
                thickness = 2
            elif line == "":
                y += 5
                continue
            else:
                color = (255, 255, 255)
                font_scale = 0.4
                thickness = 1
            
            cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, thickness)
            y += 20
        
        return image
    
    def _add_status_bar(self, image: np.ndarray) -> np.ndarray:
        """Add status bar at bottom"""
        h, w = image.shape[:2]
        
        # Background
        cv2.rectangle(image, (0, h-40), (w, h), (40, 40, 40), -1)
        
        # Statistics
        coverage = (np.sum(self.mask > 0) / self.mask.size) * 100
        
        status = f"Coverage: {coverage:.1f}% | Brush: {self.brush_size}px | " \
                f"Tolerance: {self.tolerance} | History: {self.history_index+1}/{len(self.history)}"
        
        cv2.putText(image, status, (10, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def _add_tool_info(self, image: np.ndarray) -> np.ndarray:
        """Add current tool info"""
        h, w = image.shape[:2]
        
        tool_hints = {
            SelectionMode.BRUSH: "Click & drag to paint | Auto-snaps to edges",
            SelectionMode.MAGIC_WAND: "Click to select similar colors | Ctrl=Non-contiguous",
            SelectionMode.QUICK_SELECT: "Paint to select intelligently | AI-powered",
            SelectionMode.SCISSORS: "Click points for path | Right-click to complete",
            SelectionMode.OBJECT_SELECT: "Click inside object to select",
            SelectionMode.COLOR_RANGE: "Click color to select all similar"
        }
        
        hint = tool_hints[self.current_mode]
        
        # Background
        cv2.rectangle(image, (0, 0), (w, 35), (40, 40, 40), -1)
        
        cv2.putText(image, hint, (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return image
    
    def run(self) -> Optional[np.ndarray]:
        """
        Run the interactive GUI
        
        Returns:
            Corrected mask or None if cancelled
        """
        print("\n" + "="*70)
        print("🎨 ENHANCED INTERACTIVE FLOOR CORRECTION")
        print("="*70)
        print("\nPress 'I' to toggle instructions")
        print("Press SPACE to accept, ESC to cancel")
        print("Press 1-6 to switch tools\n")
        
        while True:
            # Generate and show visualization
            display = self._get_visualization()
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard
            key = cv2.waitKey(10) & 0xFF
            
            if key == 27:  # ESC
                print("\n❌ Correction cancelled")
                cv2.destroyWindow(self.window_name)
                return None
            
            elif key == ord(' '):  # SPACE
                print("\n✅ Correction accepted")
                cv2.destroyWindow(self.window_name)
                return self.mask.copy()
            
            elif key == ord('1'):
                self.current_mode = SelectionMode.BRUSH
                print("🖌️ Tool: Brush")
            
            elif key == ord('2'):
                self.current_mode = SelectionMode.MAGIC_WAND
                print("🪄 Tool: Magic Wand")
            
            elif key == ord('3'):
                self.current_mode = SelectionMode.QUICK_SELECT
                print("⚡ Tool: Quick Selection")
            
            elif key == ord('4'):
                self.current_mode = SelectionMode.SCISSORS
                self.scissors_points = []
                print("✂️ Tool: Intelligent Scissors")
            
            elif key == ord('5'):
                self.current_mode = SelectionMode.OBJECT_SELECT
                print("🎯 Tool: Object Select")
            
            elif key == ord('6'):
                self.current_mode = SelectionMode.COLOR_RANGE
                print("🎨 Tool: Color Range")
            
            elif key == ord('z') or key == ord('Z'):
                self._undo()
            
            elif key == ord('y') or key == ord('Y'):
                self._redo()
            
            elif key == ord('g') or key == ord('G'):
                print("📈 Growing selection...")
                self.mask = self.selection_tools.grow_selection(self.mask, iterations=5)
                self._save_history()
            
            elif key == ord('s') or key == ord('S'):
                print("📉 Shrinking selection...")
                self.mask = self.selection_tools.shrink_selection(self.mask, pixels=5)
                self._save_history()
            
            elif key == ord('m') or key == ord('M'):
                print("🌊 Smoothing selection...")
                self.mask = self.selection_tools.smooth_selection(self.mask, radius=5)
                self._save_history()
            
            elif key == ord('f') or key == ord('F'):
                print("🪶 Feathering selection...")
                self.mask = self.selection_tools.feather_selection(self.mask, radius=10)
                self._save_history()
            
            elif key == ord('e') or key == ord('E'):
                self.show_edges = not self.show_edges
            
            elif key == ord('i') or key == ord('I'):
                self.show_instructions = not self.show_instructions
        
        cv2.destroyWindow(self.window_name)
        return None


def enhanced_floor_correction(image: np.ndarray, 
                             initial_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Launch enhanced interactive floor correction
    
    Args:
        image: Input image (BGR)
        initial_mask: Initial floor detection mask
    
    Returns:
        Corrected mask or None if cancelled
    """
    gui = EnhancedFloorCorrectionGUI(image, initial_mask)
    return gui.run()
