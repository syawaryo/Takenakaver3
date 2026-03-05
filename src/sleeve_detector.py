"""
スリーブ検出（HSV連結成分 + HoughCircles 併用）

方式A: HSV青フィルタ → 連結成分 → 円形度判定（従来）
方式B: グレースケール HoughCircles → 青色判定（新規）
両方の結果をマージして重複除去。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from models import BBox, PixelPoint, SleeveCircle


@dataclass
class SleeveDetection:
    """検出したスリーブ1個分"""
    circle: SleeveCircle
    text_bbox: Optional[BBox]
    component_bbox: BBox


# ==========================================================================
# 方式A: HSV連結成分方式（従来）
# ==========================================================================

def _find_all_circular_contours(
    contours: list[np.ndarray],
    min_area: float = 20.0,
    min_circularity: float = 0.5,
) -> list[tuple[PixelPoint, float, float]]:
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity >= min_circularity:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            results.append((PixelPoint(x=cx, y=cy), radius, circularity))
    return results


def _detect_by_hsv_components(
    work_img: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_component_area: int,
    min_circle_area: float,
    min_circularity: float,
    max_radius: float,
) -> list[tuple[PixelPoint, float, float]]:
    """HSV青フィルタ + 連結成分で円を検出。"""
    hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # 軽いCLOSEで近接ピクセルを結合
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    results: list[tuple[PixelPoint, float, float]] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        circles = _find_all_circular_contours(contours, min_circle_area, min_circularity)
        # 各連結成分から最も円形度の高い1つだけ採用
        valid = [(c, r, circ) for c, r, circ in circles if r <= max_radius]
        if valid:
            best = max(valid, key=lambda x: x[2])
            results.append(best)

    return results


# ==========================================================================
# 方式B: HoughCircles方式（新規）
# ==========================================================================

def _detect_by_hough(
    work_img: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_radius: int = 4,
    max_radius: int = 20,
    param1: int = 100,
    param2: int = 30,
    min_dist: int = 15,
) -> list[tuple[PixelPoint, float, float]]:
    """
    グレースケールでHoughCirclesを走らせて円を検出。
    検出後に円の周辺が青く、内部が明るい（中空）ことを判定。
    """
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

    # ブラーをかけてノイズ低減（HoughCirclesの前処理）
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    # 青色マスクを作成（判定用）
    hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    results: list[tuple[PixelPoint, float, float]] = []

    for x, y, r in circles[0]:
        x, y, r = float(x), float(y), float(r)

        # 円周上の青ピクセル比率を計算
        ring_mask = np.zeros(blue_mask.shape, dtype=np.uint8)
        cv2.circle(ring_mask, (int(x), int(y)), int(r + 2), 255, 3)
        ring_pixels = cv2.countNonZero(ring_mask)
        if ring_pixels == 0:
            continue
        blue_on_ring = cv2.countNonZero(blue_mask & ring_mask)
        blue_ratio = blue_on_ring / ring_pixels

        # 円内部の明るさチェック（スリーブ円は中空＝白背景）
        inner_mask = np.zeros(gray.shape, dtype=np.uint8)
        inner_r = max(1, int(r - 2))
        cv2.circle(inner_mask, (int(x), int(y)), inner_r, 255, -1)
        inner_pixels = cv2.countNonZero(inner_mask)
        if inner_pixels == 0:
            continue
        inner_brightness = cv2.mean(gray, mask=inner_mask)[0]

        # 青比率34%以上 + 内部が明るい（200以上 = 白に近い＝中空円）
        if blue_ratio >= 0.34 and inner_brightness >= 200:
            circularity = min(0.85, 0.5 + blue_ratio)
            results.append((PixelPoint(x=x, y=y), r, circularity))

    return results


# ==========================================================================
# 統合
# ==========================================================================

def _deduplicate(detections: list[SleeveDetection], min_distance: float) -> list[SleeveDetection]:
    if not detections:
        return detections

    detections.sort(key=lambda d: d.circle.circularity, reverse=True)
    kept: list[SleeveDetection] = []
    for det in detections:
        is_dup = False
        for k in kept:
            dx = det.circle.center_px.x - k.circle.center_px.x
            dy = det.circle.center_px.y - k.circle.center_px.y
            if (dx * dx + dy * dy) ** 0.5 < min_distance:
                is_dup = True
                break
        if not is_dup:
            kept.append(det)
    return kept


def detect_sleeves_with_annotations(
    img: np.ndarray,
    roi: Optional[tuple[int, int, int, int]] = None,
    hsv_lower: tuple[int, int, int] = (90, 50, 50),
    hsv_upper: tuple[int, int, int] = (135, 255, 255),
    min_component_area: int = 80,
    min_circle_area: float = 20.0,
    min_circularity: float = 0.65,
    max_radius: float = 30.0,
) -> list[SleeveDetection]:
    """
    HSV連結成分 + HoughCircles の併用でスリーブ青丸を検出。
    """
    # ROI適用
    y_offset, x_offset = 0, 0
    work_img = img
    if roi:
        y_start, y_end, x_start, x_end = roi
        work_img = img[y_start:y_end, x_start:x_end]
        y_offset = y_start
        x_offset = x_start

    # 方式A: HSV連結成分
    circles_a = _detect_by_hsv_components(
        work_img, hsv_lower, hsv_upper,
        min_component_area, min_circle_area, min_circularity, max_radius,
    )
    # 方式B: HoughCircles + 青判定
    circles_b = _detect_by_hough(
        work_img, hsv_lower, hsv_upper,
        min_radius=3, max_radius=int(max_radius),
    )

    # マージ
    all_circles = circles_a + circles_b

    # SleeveDetectionに変換
    results: list[SleeveDetection] = []
    for center, radius, circularity in all_circles:
        center_global = PixelPoint(x=center.x + x_offset, y=center.y + y_offset)

        sleeve_circle = SleeveCircle(
            center_px=center_global,
            radius_px=radius,
            circularity=circularity,
            color_confidence=0.9,
        )

        r_int = max(1, int(radius))
        comp_bbox = BBox(
            x=center_global.x - r_int,
            y=center_global.y - r_int,
            w=r_int * 2,
            h=r_int * 2,
        )

        results.append(
            SleeveDetection(
                circle=sleeve_circle,
                text_bbox=None,  # テキスト紐付けはmain.pyの距離ベースに任せる
                component_bbox=comp_bbox,
            )
        )

    results = _deduplicate(results, min_distance=10.0)
    return results
