"""
HSV Blue-sleeve diagnostic script.

Purpose:
  Investigate why blue circles (sleeves) are not being detected by the
  HSV-based method in sleeve_detector.py.

Steps:
  1. Load the image docs/スクリーンショットタブ削除後拡大.png
  2. Apply the HSV blue filter (90,50,50)-(135,255,255) and save raw mask
  3. Apply CLOSE 3x3 morphology and save closed mask
  4. Run connectedComponentsWithStats on the closed mask
  5. For each component with area >= 80, find contours, check circularity
  6. Print component stats: area, contour count, best circularity, radius, bbox
  7. Sample HSV values around known sleeve approximate positions
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    img_path = PROJECT_ROOT / "docs" / "スクリーンショットタブ削除後拡大.png"
    print(f"Loading image: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: Cannot read image at {img_path}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"Image size: {w} x {h}")

    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. HSV blue filter — raw mask
    # ------------------------------------------------------------------
    hsv_lower = (90, 50, 50)
    hsv_upper = (135, 255, 255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_raw = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    raw_mask_path = out_dir / "debug_blue_mask.png"
    cv2.imwrite(str(raw_mask_path), mask_raw)
    print(f"\nSaved raw blue mask -> {raw_mask_path}")

    non_zero_raw = cv2.countNonZero(mask_raw)
    total_px = h * w
    print(f"  Raw mask non-zero pixels: {non_zero_raw} / {total_px}  "
          f"({100.0 * non_zero_raw / total_px:.4f}%)")

    # ------------------------------------------------------------------
    # 3. Morphology CLOSE 3x3
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)

    closed_mask_path = out_dir / "debug_blue_mask_closed.png"
    cv2.imwrite(str(closed_mask_path), mask_closed)
    print(f"Saved closed blue mask -> {closed_mask_path}")

    non_zero_closed = cv2.countNonZero(mask_closed)
    print(f"  Closed mask non-zero pixels: {non_zero_closed} / {total_px}  "
          f"({100.0 * non_zero_closed / total_px:.4f}%)")

    # ------------------------------------------------------------------
    # 4-6. Connected components + contour circularity analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Connected components analysis  (min_component_area = 80)")
    print("=" * 80)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed)
    print(f"Total connected components (incl. background): {num_labels}")

    min_component_area = 80
    qualifying = 0

    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        qualifying += 1
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Extract component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Evaluate circularity of every contour
        best_circularity = 0.0
        best_radius = 0.0
        best_center = (0, 0)
        contour_details = []

        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                circularity = 0.0
            else:
                circularity = 4 * np.pi * cnt_area / (perimeter * perimeter)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            contour_details.append({
                "area": cnt_area,
                "perimeter": perimeter,
                "circularity": circularity,
                "radius": radius,
                "center": (cx, cy),
            })
            if circularity > best_circularity:
                best_circularity = circularity
                best_radius = radius
                best_center = (cx, cy)

        print(f"\n--- Component {i} ---")
        print(f"  Component area  : {area}")
        print(f"  Bounding box    : x={bx}, y={by}, w={bw}, h={bh}")
        print(f"  Centroid        : ({centroids[i][0]:.1f}, {centroids[i][1]:.1f})")
        print(f"  Num contours    : {len(contours)}")
        print(f"  Best circularity: {best_circularity:.4f}")
        print(f"  Best radius     : {best_radius:.2f}")
        print(f"  Best center     : ({best_center[0]:.1f}, {best_center[1]:.1f})")

        # Show individual contour details if there are a reasonable number
        if len(contour_details) <= 20:
            for ci, cd in enumerate(contour_details):
                flag = ""
                if cd["circularity"] >= 0.65 and cd["radius"] <= 30:
                    flag = " <-- WOULD PASS (circ>=0.65, r<=30)"
                elif cd["circularity"] >= 0.50:
                    flag = " (circ>=0.50 but <0.65)"
                print(f"    contour[{ci}]: area={cd['area']:.1f}, "
                      f"perim={cd['perimeter']:.1f}, "
                      f"circ={cd['circularity']:.4f}, "
                      f"r={cd['radius']:.2f}, "
                      f"center=({cd['center'][0]:.1f},{cd['center'][1]:.1f})"
                      f"{flag}")
        else:
            # Too many contours — just show top 10 by circularity
            contour_details.sort(key=lambda c: c["circularity"], reverse=True)
            print(f"    (showing top 10 of {len(contour_details)} contours by circularity)")
            for ci, cd in enumerate(contour_details[:10]):
                flag = ""
                if cd["circularity"] >= 0.65 and cd["radius"] <= 30:
                    flag = " <-- WOULD PASS"
                print(f"    contour[{ci}]: area={cd['area']:.1f}, "
                      f"perim={cd['perimeter']:.1f}, "
                      f"circ={cd['circularity']:.4f}, "
                      f"r={cd['radius']:.2f}, "
                      f"center=({cd['center'][0]:.1f},{cd['center'][1]:.1f})"
                      f"{flag}")

    print(f"\nTotal components with area >= {min_component_area}: {qualifying}")

    # Also show components with smaller area (30-79) as they might be the sleeves
    print("\n" + "=" * 80)
    print("Smaller components (area 30-79) -- potential missed sleeves")
    print("=" * 80)
    small_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 30 <= area < min_component_area:
            small_count += 1
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            best_circ = 0.0
            best_r = 0.0
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circ = 4 * np.pi * cnt_area / (perimeter * perimeter)
                    if circ > best_circ:
                        best_circ = circ
                        _, r = cv2.minEnclosingCircle(cnt)
                        best_r = r

            if best_circ >= 0.4:  # only print somewhat circular ones
                print(f"  Component {i}: area={area}, bbox=({bx},{by},{bw},{bh}), "
                      f"centroid=({centroids[i][0]:.1f},{centroids[i][1]:.1f}), "
                      f"contours={len(contours)}, best_circ={best_circ:.4f}, "
                      f"best_r={best_r:.2f}")
    print(f"  Total small components (30-79): {small_count}")

    # ------------------------------------------------------------------
    # 7. Sample HSV values at known approximate sleeve positions
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HSV sampling at known approximate sleeve positions")
    print("=" * 80)

    # Based on previous results, sleeves are expected around these regions.
    # We sample a grid of candidate positions.
    sample_regions = [
        ("Region A (x~400-500, y~150-250)", 400, 150, 500, 250),
        ("Region B (x~500-600, y~150-250)", 500, 150, 600, 250),
        ("Region C (x~600-700, y~150-250)", 600, 150, 700, 250),
        ("Region D (x~400-500, y~250-350)", 400, 250, 500, 350),
        ("Region E (x~500-600, y~250-350)", 500, 250, 600, 350),
        ("Region F (x~600-700, y~250-350)", 600, 250, 700, 350),
        ("Region G (x~400-500, y~350-450)", 400, 350, 500, 450),
        ("Region H (x~500-600, y~350-450)", 500, 350, 600, 450),
        ("Region I (x~600-700, y~350-450)", 600, 350, 700, 450),
        ("Region J (x~400-500, y~450-560)", 400, 450, 500, 560),
        ("Region K (x~500-600, y~450-560)", 500, 450, 600, 560),
        ("Region L (x~600-700, y~450-560)", 600, 450, 700, 560),
    ]

    # First, do a fine-grained scan: look for blue-ish pixels in these regions
    # with a wider HSV range to understand what the actual colors are
    for name, x1, y1, x2, y2 in sample_regions:
        # Clamp to image bounds
        x1c = max(0, min(x1, w - 1))
        x2c = max(0, min(x2, w))
        y1c = max(0, min(y1, h - 1))
        y2c = max(0, min(y2, h))

        if x2c <= x1c or y2c <= y1c:
            print(f"\n{name}: OUT OF BOUNDS (image is {w}x{h})")
            continue

        roi_bgr = img[y1c:y2c, x1c:x2c]
        roi_hsv = hsv[y1c:y2c, x1c:x2c]

        # Find pixels that are NOT white/gray (i.e., saturation > 30 and not too dark)
        sat_channel = roi_hsv[:, :, 1]
        val_channel = roi_hsv[:, :, 2]
        colored_mask = (sat_channel > 30) & (val_channel > 30)

        colored_count = np.sum(colored_mask)

        print(f"\n{name}:")
        print(f"  Region size: {x2c - x1c} x {y2c - y1c} = {(x2c - x1c) * (y2c - y1c)} pixels")
        print(f"  Colored pixels (S>30 & V>30): {colored_count}")

        if colored_count > 0:
            h_vals = roi_hsv[:, :, 0][colored_mask]
            s_vals = roi_hsv[:, :, 1][colored_mask]
            v_vals = roi_hsv[:, :, 2][colored_mask]

            print(f"  H range: {h_vals.min()} - {h_vals.max()}  "
                  f"(mean={h_vals.mean():.1f}, median={np.median(h_vals):.1f})")
            print(f"  S range: {s_vals.min()} - {s_vals.max()}  "
                  f"(mean={s_vals.mean():.1f}, median={np.median(s_vals):.1f})")
            print(f"  V range: {v_vals.min()} - {v_vals.max()}  "
                  f"(mean={v_vals.mean():.1f}, median={np.median(v_vals):.1f})")

            # How many of these colored pixels fall within the HSV blue filter?
            in_range = ((h_vals >= hsv_lower[0]) & (h_vals <= hsv_upper[0]) &
                        (s_vals >= hsv_lower[1]) & (s_vals <= hsv_upper[1]) &
                        (v_vals >= hsv_lower[2]) & (v_vals <= hsv_upper[2]))
            in_count = np.sum(in_range)
            print(f"  Colored pixels in blue HSV range ({hsv_lower}-{hsv_upper}): "
                  f"{in_count} / {colored_count} ({100.0 * in_count / colored_count:.1f}%)")

            # Histogram of H values for colored pixels
            h_hist, _ = np.histogram(h_vals, bins=range(0, 181, 10))
            h_labels = [f"{i*10}-{i*10+9}" for i in range(18)]
            print(f"  H histogram (colored pixels):")
            for bi, count in enumerate(h_hist):
                if count > 0:
                    bar = "#" * min(50, count)
                    print(f"    H={h_labels[bi]:>7s}: {count:5d} {bar}")
        else:
            print(f"  No colored pixels found -- region is all white/gray")

    # ------------------------------------------------------------------
    # Extra: Fine 20x20 patch sampling at specific grid points
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Fine 20x20 patch HSV sampling at grid points")
    print("=" * 80)

    patch_size = 20
    half = patch_size // 2

    grid_points = []
    for x in range(400, 720, 20):
        for y in range(150, 570, 20):
            grid_points.append((x, y))

    # Find grid points that have blue pixels in the patch
    blue_hot_spots = []
    for (px, py) in grid_points:
        x1 = max(0, px - half)
        x2 = min(w, px + half)
        y1 = max(0, py - half)
        y2 = min(h, py + half)

        patch_mask = mask_raw[y1:y2, x1:x2]
        blue_count = cv2.countNonZero(patch_mask)
        if blue_count > 5:  # at least a few blue pixels
            blue_hot_spots.append((px, py, blue_count))

    if blue_hot_spots:
        print(f"Found {len(blue_hot_spots)} grid patches with blue pixels (>5 in 20x20):")
        for (px, py, cnt) in sorted(blue_hot_spots, key=lambda t: -t[2]):
            x1 = max(0, px - half)
            x2 = min(w, px + half)
            y1 = max(0, py - half)
            y2 = min(h, py + half)

            patch_hsv = hsv[y1:y2, x1:x2]
            patch_mask_px = mask_raw[y1:y2, x1:x2]

            # Get HSV of the blue pixels in this patch
            blue_px_mask = patch_mask_px > 0
            if np.any(blue_px_mask):
                bh = patch_hsv[:, :, 0][blue_px_mask]
                bs = patch_hsv[:, :, 1][blue_px_mask]
                bv = patch_hsv[:, :, 2][blue_px_mask]
                print(f"  ({px:4d},{py:4d}): {cnt:3d} blue px  "
                      f"H=[{bh.min():3d}-{bh.max():3d}] "
                      f"S=[{bs.min():3d}-{bs.max():3d}] "
                      f"V=[{bv.min():3d}-{bv.max():3d}]")
    else:
        print("No grid patches with blue pixels found in the expected sleeve region!")
        print("This likely means the sleeves use a different hue, or the image")
        print("coordinates differ from expectations.")

    # ------------------------------------------------------------------
    # Extra: Scan the entire image for blue pixel density map
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Blue pixel density map (50x50 tiles, whole image)")
    print("=" * 80)

    tile = 50
    print(f"Tile size: {tile}x{tile}")
    hot_tiles = []
    for ty in range(0, h, tile):
        for tx in range(0, w, tile):
            t_mask = mask_raw[ty:min(ty + tile, h), tx:min(tx + tile, w)]
            cnt = cv2.countNonZero(t_mask)
            if cnt > 10:
                hot_tiles.append((tx, ty, cnt))

    hot_tiles.sort(key=lambda t: -t[2])
    if hot_tiles:
        print(f"Tiles with >10 blue pixels ({len(hot_tiles)} tiles):")
        for (tx, ty, cnt) in hot_tiles[:40]:
            print(f"  tile ({tx:4d},{ty:4d})-({tx + tile:4d},{ty + tile:4d}): {cnt:5d} blue pixels")
    else:
        print("NO tiles with >10 blue pixels found in the ENTIRE image!")
        print("The HSV range (90,50,50)-(135,255,255) may not match any colors in this image.")

        # Do a broader scan to find what colors ARE present
        print("\nBroader color analysis of the whole image:")
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        colored = (sat > 20) & (val > 20)
        total_colored = np.sum(colored)
        print(f"  Total colored pixels (S>20 & V>20): {total_colored}")

        if total_colored > 0:
            all_h = hsv[:, :, 0][colored]
            all_s = hsv[:, :, 1][colored]
            all_v = hsv[:, :, 2][colored]

            print(f"  H range: {all_h.min()} - {all_h.max()}")
            print(f"  S range: {all_s.min()} - {all_s.max()}")
            print(f"  V range: {all_v.min()} - {all_v.max()}")

            h_hist, _ = np.histogram(all_h, bins=range(0, 181, 5))
            h_labels = [f"{i * 5:3d}-{i * 5 + 4:3d}" for i in range(36)]
            print(f"\n  Hue histogram (all colored pixels):")
            for bi, count in enumerate(h_hist):
                if count > 0:
                    pct = 100.0 * count / total_colored
                    bar = "#" * min(60, int(pct * 2))
                    print(f"    H={h_labels[bi]}: {count:7d} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 80)
    print("Diagnostic complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
