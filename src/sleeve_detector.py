"""
スリーブ検出（連結成分方式）

HSV青フィルタ → ハッチング除去 → 連結成分分析
→ 各連結成分から丸い部分=スリーブ座標、残り=テキスト領域
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from models import BBox, PixelPoint, SleeveCircle


@dataclass
class SleeveDetection:
    """連結成分から検出したスリーブ1個分"""
    circle: SleeveCircle
    text_bbox: Optional[BBox]  # テキスト領域
    component_bbox: BBox       # 連結成分全体のBBox


def _find_most_circular_contour(
    contours: list[np.ndarray],
    min_area: float = 20.0,
) -> Optional[tuple[PixelPoint, float, float]]:
    """輪郭群から最も円形度の高いものを返す。"""
    best = None
    best_circ = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > best_circ:
            best_circ = circularity
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            best = (PixelPoint(x=cx, y=cy), radius, circularity)

    return best


def _find_all_circular_contours(
    contours: list[np.ndarray],
    min_area: float = 20.0,
    min_circularity: float = 0.5,
) -> list[tuple[PixelPoint, float, float]]:
    """輪郭群から円形度の高い全ての輪郭を返す。"""
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


def _get_text_region_bbox(
    component_mask: np.ndarray,
    circle_center: PixelPoint,
    circle_radius: float,
) -> Optional[BBox]:
    """連結成分マスクから丸い部分を除いた残りのBBoxを求める。"""
    mask_no_circle = component_mask.copy()
    cv2.circle(
        mask_no_circle,
        (int(circle_center.x), int(circle_center.y)),
        int(circle_radius * 1.5),
        0,
        -1,
    )

    coords = cv2.findNonZero(mask_no_circle)
    if coords is None or len(coords) < 10:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return BBox(x=x, y=y, w=w, h=h)


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
    青色連結成分方式でスリーブを検出。

    Parameters
    ----------
    img : BGR画像
    roi : (y_start, y_end, x_start, x_end) 検出対象領域。Noneなら全体。
    hsv_lower, hsv_upper : HSV青色フィルタの範囲
    min_component_area : 連結成分の最小面積（ノイズ除去）
    min_circle_area : 円検出の最小面積
    min_circularity : 円形度の最小値
    max_radius : 最大半径（px）。大きすぎるものを除外。

    Returns
    -------
    検出されたスリーブのリスト
    """
    # ROI適用
    y_offset, x_offset = 0, 0
    work_img = img
    if roi:
        y_start, y_end, x_start, x_end = roi
        work_img = img[y_start:y_end, x_start:x_end]
        y_offset = y_start
        x_offset = x_start

    # 1. HSV青フィルタ
    hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # 2. 軽いCLOSE(3x3)でピクセル隙間を埋めるだけ（大きいカーネルだと複数スリーブが結合する）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. 連結成分分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    results: list[SleeveDetection] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        component_mask = (labels == i).astype(np.uint8) * 255

        sx = stats[i, cv2.CC_STAT_LEFT]
        sy = stats[i, cv2.CC_STAT_TOP]
        sw = stats[i, cv2.CC_STAT_WIDTH]
        sh = stats[i, cv2.CC_STAT_HEIGHT]

        # 4. 全輪郭から円形のものを探す（RETR_LIST で内側輪郭も取得）
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        circles = _find_all_circular_contours(contours, min_circle_area, min_circularity)

        if not circles:
            # 輪郭全体で1つの円として判定
            circle_info = _find_most_circular_contour(contours, min_circle_area)
            if circle_info and circle_info[2] >= min_circularity:
                circles = [circle_info]

        for center, radius, circularity in circles:
            if radius > max_radius:
                continue

            # オフセット適用（ROI座標 → 元画像座標）
            center_global = PixelPoint(
                x=center.x + x_offset,
                y=center.y + y_offset,
            )

            sleeve_circle = SleeveCircle(
                center_px=center_global,
                radius_px=radius,
                circularity=circularity,
                color_confidence=0.9,
            )

            # BBox（元画像座標系）
            comp_bbox = BBox(
                x=sx + x_offset,
                y=sy + y_offset,
                w=sw,
                h=sh,
            )

            text_bbox = _get_text_region_bbox(component_mask, center, radius)
            if text_bbox:
                text_bbox = BBox(
                    x=text_bbox.x + x_offset,
                    y=text_bbox.y + y_offset,
                    w=text_bbox.w,
                    h=text_bbox.h,
                )

            results.append(
                SleeveDetection(
                    circle=sleeve_circle,
                    text_bbox=text_bbox,
                    component_bbox=comp_bbox,
                )
            )

    # 重複除去: 近い円は1つにまとめる
    results = _deduplicate(results, min_distance=10.0)

    return results


def _deduplicate(detections: list[SleeveDetection], min_distance: float) -> list[SleeveDetection]:
    """近接するスリーブ検出を統合。"""
    if not detections:
        return detections

    # 信頼度（circularity）順にソート
    detections.sort(key=lambda d: d.circle.circularity, reverse=True)

    kept: list[SleeveDetection] = []
    for det in detections:
        is_dup = False
        for k in kept:
            dx = det.circle.center_px.x - k.circle.center_px.x
            dy = det.circle.center_px.y - k.circle.center_px.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_distance:
                is_dup = True
                break
        if not is_dup:
            kept.append(det)

    return kept
