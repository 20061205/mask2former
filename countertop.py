"""
Countertop Tile Visualiser
==========================
Dedicated script for applying tiles to kitchen/pantry countertops.

Usage
-----
  python countertop.py --room kitchen.jpg --tile marble.jpg
  python countertop.py --room kitchen.jpg --tile marble.jpg --tile-size 80 --rotation 0
  python countertop.py --room kitchen.jpg --tile marble.jpg --grout 2 --debug
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import (
    COUNTERTOP_TILE_SIZE,
    COUNTERTOP_GROUT_WIDTH,
    COUNTERTOP_ROTATION_ANGLE,
)
from model import load_model, segment_image, get_label_map
from surfaces import extract_surface_mask, clean_mask
from tile_engine import build_full_tile_grid, composite_tile_on_surface


# ── Helpers ──────────────────────────────────────────────────────────

def save_preview(room_pil, mask, result_bgr, output_path: Path):
    """Save a 3-panel comparison: original | mask | result."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(room_pil)
    axes[0].set_title("Original Room")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Countertop Mask")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Tiled Countertop")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Preview saved: {output_path}")


def save_debug(outdir: Path, mask, full_tile, room_bgr):
    """Save intermediate images for debugging."""
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "debug_countertop_mask.png"), mask)
    cv2.imwrite(str(outdir / "debug_countertop_tile_grid.png"), full_tile)

    cutout = room_bgr.copy()
    cutout[mask > 0] = 0
    cv2.imwrite(str(outdir / "debug_countertop_room_cutout.png"), cutout)
    print(f"Debug images saved to {outdir}")


def save_mask_edges_on_room(room_bgr: np.ndarray, mask: np.ndarray,
                            output_path: Path, line_thickness: int = 3):
    """
    Draw red contour lines on the original room image showing the
    edges / borders of the white regions in the mask.

    Also marks corner points with green circles.
    """
    # Find contours of the mask
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Draw on a copy of the room image
    vis = room_bgr.copy()

    # Draw red contour edges
    cv2.drawContours(vis, contours, -1, (0, 0, 255), line_thickness)

    # Mark corner points with green circles
    for cnt in contours:
        # Skip tiny contours
        if cv2.contourArea(cnt) < 500:
            continue

        # Approximate polygon to find corners
        hull = cv2.convexHull(cnt)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        for pt in approx:
            x, y = pt[0]
            cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)       # filled green
            cv2.circle(vis, (x, y), 8, (0, 0, 0), 2)          # black outline

        # Also label corner coordinates
        for i, pt in enumerate(approx):
            x, y = pt[0]
            cv2.putText(vis, f"({x},{y})", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(vis, f"({x},{y})", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

    cv2.imwrite(str(output_path), vis)
    print(f"Mask edge overlay saved: {output_path}")


def detect_internal_edges(room_bgr: np.ndarray,
                          seg_map: np.ndarray,
                          label_id: int,
                          output_path: Path):
    """
    Detect only SHARP, straight internal edges within a label region
    (e.g. kitchen island id=73) — the hard boundary where the horizontal
    top meets the vertical front/side.

    Uses high-threshold Canny + LSD with strict length and gradient filters
    so only strong, well-defined edges are kept.

    Saves:
      - *_canny.png   : sharp Canny edges inside the region
      - *_lines.png   : detected straight lines only
      - *_overlay.png : red lines + green corners on room photo
    """
    h, w = room_bgr.shape[:2]
    region_mask = (seg_map == label_id).astype(np.uint8) * 255

    if region_mask.sum() == 0:
        print(f"  Label {label_id} not found in segmentation, skipping.")
        return None

    stem = output_path.stem
    parent = output_path.parent

    # ── 1. Sharp-only Canny (high thresholds) inside the region ──────
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)

    # Erode mask to ignore the outer segmentation border
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    region_eroded = cv2.erode(region_mask, kernel_erode, iterations=2)

    # Light blur only — preserve sharp transitions
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)

    # HIGH Canny thresholds → only strong gradient edges survive
    edges_raw = cv2.Canny(blurred, 100, 200)
    edges_internal = cv2.bitwise_and(edges_raw, region_eroded)

    cv2.imwrite(str(parent / f"{stem}_canny.png"), edges_internal)

    # ── 2. LSD with strict filters ──────────────────────────────────
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lsd_lines, lsd_widths, _, _ = lsd.detect(blurred)

    # Only keep long lines (>= 8% of image dimension)
    min_line_length = max(40, min(h, w) * 0.08)
    all_lines = []  # (x1, y1, x2, y2, length, angle_deg)

    if lsd_lines is not None:
        for i, seg in enumerate(lsd_lines):
            x1, y1, x2, y2 = seg[0]
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Midpoint must be inside the eroded region
            if my < 0 or my >= h or mx < 0 or mx >= w:
                continue
            if region_eroded[my, mx] == 0:
                continue

            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < min_line_length:
                continue

            # Check that the line crosses a strong gradient (sharp edge)
            # Sample gradient magnitude along the line midpoint
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            if grad_mag[my, mx] < 40:   # weak gradient → skip
                continue

            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            all_lines.append((x1, y1, x2, y2, length, angle))

    # ── 3. HoughLinesP on the sharp Canny edges only ────────────────
    hough_lines = cv2.HoughLinesP(edges_internal, rho=1, theta=np.pi / 180,
                                   threshold=40,
                                   minLineLength=int(min_line_length),
                                   maxLineGap=10)
    if hough_lines is not None:
        for seg in hough_lines:
            x1, y1, x2, y2 = seg[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < min_line_length:
                continue
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            all_lines.append((float(x1), float(y1), float(x2), float(y2),
                              length, angle))

    # ── 4. Deduplicate: keep longest, suppress nearby parallels ──────
    if not all_lines:
        print(f"  No sharp straight lines detected inside label {label_id}.")
        cv2.imwrite(str(parent / f"{stem}_lines.png"),
                    np.zeros((h, w), dtype=np.uint8))
        cv2.imwrite(str(parent / f"{stem}_overlay.png"), room_bgr.copy())
        return None

    all_lines.sort(key=lambda l: -l[4])

    kept_lines = []
    used = [False] * len(all_lines)
    for i in range(len(all_lines)):
        if used[i]:
            continue
        kept_lines.append(all_lines[i])
        x1i, y1i, x2i, y2i, li, ai = all_lines[i]
        mxi, myi = (x1i + x2i) / 2, (y1i + y2i) / 2
        for j in range(i + 1, len(all_lines)):
            if used[j]:
                continue
            x1j, y1j, x2j, y2j, lj, aj = all_lines[j]
            mxj, myj = (x1j + x2j) / 2, (y1j + y2j) / 2
            dist = np.sqrt((mxi - mxj)**2 + (myi - myj)**2)
            angle_diff = abs(ai - aj)
            if dist < 40 and angle_diff < 12:
                used[j] = True

    # ── 5. Draw results ──────────────────────────────────────────────
    lines_img = np.zeros((h, w), dtype=np.uint8)
    vis = room_bgr.copy()

    corner_pts = []
    for x1, y1, x2, y2, length, angle in kept_lines:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        if angle < 30:
            color = (0, 0, 255)      # red = horizontal
        elif angle > 60:
            color = (255, 100, 0)    # blue = vertical
        else:
            color = (0, 255, 255)    # yellow = diagonal

        cv2.line(lines_img, p1, p2, 255, 2)
        cv2.line(vis, p1, p2, color, 3)
        corner_pts.append(p1)
        corner_pts.append(p2)

    # Cluster corner points
    if corner_pts:
        pts = np.array(corner_pts, dtype=np.float32)
        n_clusters = min(8, len(pts))
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, _, centers = cv2.kmeans(pts, n_clusters, None, criteria,
                                   10, cv2.KMEANS_PP_CENTERS)
        for cx, cy in centers.astype(int):
            cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 0), 2)
            cv2.putText(vis, f"({cx},{cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(vis, f"({cx},{cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 0), 1)

    cv2.imwrite(str(parent / f"{stem}_lines.png"), lines_img)
    cv2.imwrite(str(parent / f"{stem}_overlay.png"), vis)

    print(f"Sharp edge detection: {len(kept_lines)} lines found")
    print(f"  Canny edges  : {parent / f'{stem}_canny.png'}")
    print(f"  Lines only   : {parent / f'{stem}_lines.png'}")
    print(f"  Overlay      : {parent / f'{stem}_overlay.png'}")

    return kept_lines


def create_island_top_mask(seg_map: np.ndarray,
                           label_id: int,
                           kept_lines: list,
                           output_path: Path,
                           room_bgr: np.ndarray = None,
                           other_ids: list = None):
    """
    Build a combined countertop mask:
      - id=label_id (e.g. 73): only pixels ABOVE the horizontal red lines
      - other_ids (e.g. [70, 45]): full region masks included as-is

    For each x-column, pixels in the label_id region that are ABOVE
    (smaller y) the topmost horizontal line at that x are kept as white;
    everything below is black.

    If room_bgr is given, also saves the mask overlaid on the room.

    Returns
    -------
    top_mask : np.ndarray (H, W) uint8,  255 = island top, 0 = elsewhere
    """
    h, w = seg_map.shape
    region_mask = (seg_map == label_id).astype(np.uint8) * 255

    # ── Filter: keep only horizontal lines (angle < 30°) ─────────────
    horiz_lines = [(x1, y1, x2, y2, l, a)
                   for x1, y1, x2, y2, l, a in kept_lines if a < 30]

    if not horiz_lines:
        print("  No horizontal lines found — returning full region mask.")
        cv2.imwrite(str(output_path), region_mask)
        return region_mask

    # ── Build a per-column "cutoff y" array ──────────────────────────
    # For every x, the cutoff is the MAXIMUM y of any horizontal line
    # passing through that x.  Pixels at y <= cutoff are "top".
    cutoff_y = np.full(w, -1, dtype=np.int32)

    for x1, y1, x2, y2, length, angle in horiz_lines:
        # Rasterise each line segment to get (x, y) pairs
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        n_pts = max(abs(x2i - x1i), abs(y2i - y1i), 1) + 1
        xs = np.linspace(x1, x2, n_pts).astype(int)
        ys = np.linspace(y1, y2, n_pts).astype(int)
        for xi, yi in zip(xs, ys):
            if 0 <= xi < w:
                cutoff_y[xi] = max(cutoff_y[xi], yi)

    # ── Interpolate gaps (columns not covered by any line) ───────────
    # For columns with no line data, interpolate from nearest columns
    valid = np.where(cutoff_y >= 0)[0]
    if len(valid) == 0:
        print("  Could not build cutoff line — returning full region mask.")
        cv2.imwrite(str(output_path), region_mask)
        return region_mask

    # Extend to fill gaps using nearest valid value
    from scipy.interpolate import interp1d
    interp_fn = interp1d(valid, cutoff_y[valid], kind='linear',
                         fill_value=(cutoff_y[valid[0]], cutoff_y[valid[-1]]),
                         bounds_error=False)
    cutoff_full = interp_fn(np.arange(w)).astype(int)

    # Add a small margin below the line to be inclusive (+5 pixels)
    cutoff_full = np.clip(cutoff_full + 5, 0, h - 1)

    # ── Create top-only mask ─────────────────────────────────────────
    top_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        top_mask[:cutoff_full[x], x] = 255

    # AND with the actual island region
    top_mask = cv2.bitwise_and(top_mask, region_mask)

    # ── Add full masks from other countertop IDs ─────────────────────
    if other_ids:
        for oid in other_ids:
            other_region = (seg_map == oid).astype(np.uint8) * 255
            if other_region.sum() > 0:
                top_mask = cv2.bitwise_or(top_mask, other_region)
                print(f"  Added full mask for id={oid} "
                      f"({(other_region > 0).sum():,} px)")

    # ── Save ─────────────────────────────────────────────────────────
    cv2.imwrite(str(output_path), top_mask)
    pixel_count = (top_mask > 0).sum()
    island_only = cv2.bitwise_and(top_mask, region_mask)
    island_px = (island_only > 0).sum()
    other_px = pixel_count - island_px
    print(f"Combined countertop mask: {pixel_count:,} px "
          f"(island top: {island_px:,}, other IDs: {other_px:,})")
    print(f"  Saved: {output_path}")

    # Overlay on room image if available
    if room_bgr is not None:
        overlay = room_bgr.copy()
        # Semi-transparent white highlight on the top surface
        overlay[top_mask > 0] = (overlay[top_mask > 0] * 0.4 +
                                  np.array([255, 255, 255]) * 0.6).astype(np.uint8)
        # Draw the cutoff line in red
        for x in range(1, w):
            y_prev, y_curr = int(cutoff_full[x - 1]), int(cutoff_full[x])
            cv2.line(overlay, (x - 1, y_prev), (x, y_curr), (0, 0, 255), 2)
        overlay_path = output_path.parent / (output_path.stem + "_overlay.png")
        cv2.imwrite(str(overlay_path), overlay)
        print(f"  Overlay: {overlay_path}")

    return top_mask


def save_segmentation_regions(room_pil, seg_map, id2label, output_path: Path):
    """
    Visualise every detected region in the segmentation map.
    Each region gets a unique colour and is labelled with its ADE20K name.
    """
    unique_ids = np.unique(seg_map)
    h, w = seg_map.shape

    # Generate distinct colours for each label
    np.random.seed(42)
    colours = {lid: np.random.randint(40, 255, size=3).tolist() for lid in unique_ids}

    # Build colour overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for lid in unique_ids:
        overlay[seg_map == lid] = colours[lid]

    # Blend with original image
    room_np = np.array(room_pil)
    blended = (0.45 * room_np + 0.55 * overlay).astype(np.uint8)

    # Compute label centroids for text placement
    labels_info = []
    for lid in unique_ids:
        mask_region = (seg_map == lid)
        pixel_count = mask_region.sum()
        coverage = pixel_count / seg_map.size * 100
        ys, xs = np.where(mask_region)
        cy, cx = int(ys.mean()), int(xs.mean())
        name = id2label.get(lid, f"id_{lid}") if id2label else f"id_{lid}"
        labels_info.append((lid, name, cx, cy, coverage, colours[lid]))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(room_pil)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title(f"Segmentation Regions ({len(unique_ids)} detected)", fontsize=14)
    axes[1].axis("off")

    # Add label text on the overlay
    for lid, name, cx, cy, coverage, colour in labels_info:
        # Skip tiny regions (<0.5%)
        if coverage < 0.5:
            continue
        label_text = f"{name}\n(id={lid}, {coverage:.1f}%)"
        axes[1].text(
            cx, cy, label_text,
            fontsize=8, fontweight="bold",
            color="white",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=[c / 255 for c in colour],
                edgecolor="white",
                alpha=0.85,
            ),
        )

    # Add legend
    legend_text = "Detected regions:\n"
    for lid, name, _, _, coverage, _ in sorted(labels_info, key=lambda x: -x[4]):
        legend_text += f"  {name} (id={lid}) — {coverage:.1f}%\n"
    fig.text(
        0.02, 0.02, legend_text.strip(),
        fontsize=8, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Segmentation region map saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply tiles to kitchen countertop using Mask2Former segmentation"
    )
    parser.add_argument("--room", required=True, help="Path to room / kitchen image")
    parser.add_argument("--tile", required=True, help="Path to tile / marble image")
    parser.add_argument(
        "--tile-size", type=int, default=COUNTERTOP_TILE_SIZE,
        help=f"Individual tile size in pixels (default: {COUNTERTOP_TILE_SIZE})"
    )
    parser.add_argument(
        "--grout", type=int, default=COUNTERTOP_GROUT_WIDTH,
        help=f"Grout line width in pixels (default: {COUNTERTOP_GROUT_WIDTH})"
    )
    parser.add_argument(
        "--rotation", type=float, default=COUNTERTOP_ROTATION_ANGLE,
        help=f"Tile rotation angle in degrees (default: {COUNTERTOP_ROTATION_ANGLE})"
    )
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug images (mask, grid, cutout)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────
    print(f"Room image : {args.room}")
    print(f"Tile image : {args.tile}")

    room_pil = Image.open(args.room).convert("RGB")
    tile_pil = Image.open(args.tile).convert("RGB")
    room_bgr = cv2.cvtColor(np.array(room_pil), cv2.COLOR_RGB2BGR)
    tile_bgr = cv2.cvtColor(np.array(tile_pil), cv2.COLOR_RGB2BGR)

    # ── Segment the room ─────────────────────────────────────────────
    print("Loading Mask2Former model...")
    processor, model, device = load_model()
    seg_map = segment_image(room_pil, processor, model, device)

    unique_labels = np.unique(seg_map).tolist()
    print(f"Segmentation complete – {len(unique_labels)} unique labels detected")
    print(f"Detected label IDs: {unique_labels}")

    # ── Visualise all segmentation regions ────────────────────────────
    id2label = get_label_map(model)
    save_segmentation_regions(
        room_pil, seg_map, id2label,
        outdir / "segmentation_regions.png",
    )

    # ── Detect internal edges within kitchen island (id=73) ──────────
    #    This shows where the TOP surface meets the SIDE/FRONT face.
    island_top_mask = None
    if 73 in unique_labels:
        print("Detecting internal edges in kitchen island (id=73)...")
        island_lines = detect_internal_edges(
            room_bgr, seg_map, label_id=73,
            output_path=outdir / "island_edges",
        )
        # Build island-top-only mask (above the red lines)
        if island_lines:
            # Other countertop IDs to include in full (not trimmed)
            other_counter_ids = [oid for oid in [70, 45]
                                 if oid in unique_labels]
            island_top_mask = create_island_top_mask(
                seg_map, label_id=73,
                kept_lines=island_lines,
                output_path=outdir / "island_top_mask.png",
                room_bgr=room_bgr,
                other_ids=other_counter_ids,
            )
    else:
        print("No kitchen island (id=73) detected, skipping edge detection.")

    # ── Choose tiling mask ─────────────────────────────────────────────
    #  Prefer the island_top_mask (island top + other countertop IDs)
    #  when available; otherwise fall back to the standard countertop mask.
    if island_top_mask is not None and island_top_mask.sum() > 0:
        mask = clean_mask(island_top_mask)
        print("Using island-top combined mask for tiling.")
    else:
        raw_mask = extract_surface_mask(seg_map, "countertop")
        if raw_mask.sum() == 0:
            print("\n⚠  No countertop detected in this image.")
            print("   The model uses ADE20K label ID 70 (countertop).")
            print("   Try a kitchen image with a clearly visible countertop.")
            print(f"   Detected label IDs: {unique_labels}")
            return
        mask = clean_mask(raw_mask)
        print("Using standard countertop mask for tiling.")

    pixel_count = (mask > 0).sum()
    coverage = pixel_count / mask.size * 100
    print(f"Countertop detected: {pixel_count:,} pixels ({coverage:.1f}% of image)")

    # ── Save mask edges overlaid on room image ───────────────────────
    save_mask_edges_on_room(
        room_bgr, mask,
        outdir / "countertop_edges.png",
    )

    # ── Build tile grid ──────────────────────────────────────────────
    print(f"Building tile grid (size={args.tile_size}, grout={args.grout}, "
          f"rotation={args.rotation}°)...")

    full_tile = build_full_tile_grid(
        room_bgr, mask, tile_bgr,
        rotation_angle=args.rotation,
        tile_size=args.tile_size,
        grout=args.grout,
    )

    if args.debug:
        save_debug(outdir, mask, full_tile, room_bgr)

    # ── Composite ────────────────────────────────────────────────────
    result_bgr = composite_tile_on_surface(room_bgr, mask, full_tile)

    # ── Save outputs ─────────────────────────────────────────────────
    out_path = outdir / "countertop_tiled.png"
    cv2.imwrite(str(out_path), result_bgr)
    print(f"Result saved: {out_path.resolve()}")

    # Save mask
    mask_path = outdir / "countertop_mask.png"
    Image.fromarray(mask).save(mask_path)

    # Save comparison preview
    save_preview(room_pil, mask, result_bgr, outdir / "countertop_preview.png")

    print("Done!")


if __name__ == "__main__":
    main()
