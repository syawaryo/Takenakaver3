"""
スリーブ検出（テンプレートマッチング方式）

テンプレート画像（青丸スリーブのサンプル）をマルチスケールで
図面上をスライドして一致箇所を検出する。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from models import BBox, PixelPoint, SleeveCircle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "docs"

# テンプレートファイル一覧
TEMPLATE_FILES = [
    "template1.png",
    "template2.png",
    "template3.png",
    "template4.png",
    "template5.png",
    "template6.png",
    "スリーブ例画像.png",
]


@dataclass
class SleeveDetection:
    """検出したスリーブ1個分"""
    circle: SleeveCircle
    text_bbox: Optional[BBox]
    component_bbox: BBox


def _nms_points(
    points: list[tuple[PixelPoint, float, float]],
    min_dist: float = 10.0,
) -> list[tuple[PixelPoint, float, float]]:
    """Non-Maximum Suppression: スコア降順で近接点を除去。"""
    if not points:
        return points

    points.sort(key=lambda p: p[1], reverse=True)  # スコア降順
    kept: list[tuple[PixelPoint, float, float]] = []
    for pt, score, radius in points:
        is_dup = False
        for kpt, _, kr in kept:
            dx = pt.x - kpt.x
            dy = pt.y - kpt.y
            # 半径ベースの動的距離
            dynamic_dist = max(min_dist, max(radius, kr) * 1.5)
            if (dx * dx + dy * dy) ** 0.5 < dynamic_dist:
                is_dup = True
                break
        if not is_dup:
            kept.append((pt, score, radius))
    return kept


def _load_templates(
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """テンプレート画像を読み込み、青マスク化して返す。"""
    templates = []
    for fname in TEMPLATE_FILES:
        path = TEMPLATE_DIR / fname
        if not path.exists():
            print(f"       [WARN] Template not found: {path}")
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"       [WARN] Cannot read template: {path}")
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        templates.append((fname, mask))
    return templates


def detect_sleeves_with_annotations(
    img: np.ndarray,
    roi: Optional[tuple[int, int, int, int]] = None,
    threshold: float = 0.8,
    scales: list[float] | None = None,
    min_dist: float = 10.0,
    **_kwargs,
) -> list[SleeveDetection]:
    """
    マルチテンプレートマッチングでスリーブ青丸を検出。

    Parameters
    ----------
    img : BGR画像
    roi : (y_start, y_end, x_start, x_end) 検出対象領域
    threshold : マッチング閾値（TM_CCOEFF_NORMED）
    scales : テンプレートのスケール倍率リスト
    min_dist : NMS最小距離(px)
    """
    # HSV青フィルタ共通パラメータ
    hsv_lower = np.array([90, 50, 50])
    hsv_upper = np.array([135, 255, 255])

    # テンプレート読み込み
    templates = _load_templates(hsv_lower, hsv_upper)
    if not templates:
        print("       [WARN] No templates loaded")
        return []
    print(f"       [INFO] Loaded {len(templates)} templates")

    # ROI適用
    y_offset, x_offset = 0, 0
    work_img = img
    if roi:
        y_start, y_end, x_start, x_end = roi
        work_img = img[y_start:y_end, x_start:x_end]
        y_offset = y_start
        x_offset = x_start

    # 図面も青マスク化
    work_hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
    drawing_mask = cv2.inRange(work_hsv, hsv_lower, hsv_upper)
    dh, dw = drawing_mask.shape[:2]

    if scales is None:
        scales = [i / 100 for i in range(1, 400, 1)]

    all_detections: list[tuple[PixelPoint, float, float]] = []

    def _match_single_template(
        tmpl_name: str,
        tmpl_mask: np.ndarray,
        drawing_mask: np.ndarray,
        scales: list[float],
        threshold: float,
        dw: int,
        dh: int,
    ) -> list[tuple[PixelPoint, float, float]]:
        """1テンプレート分のマルチスケールマッチング。"""
        hits: list[tuple[PixelPoint, float, float]] = []
        for scale in scales:
            tw = max(3, int(tmpl_mask.shape[1] * scale))
            th = max(3, int(tmpl_mask.shape[0] * scale))
            resized = cv2.resize(tmpl_mask, (tw, th), interpolation=cv2.INTER_AREA)

            if tw < 8 or th < 8 or tw > dw or th > dh:
                continue

            result = cv2.matchTemplate(drawing_mask, resized, cv2.TM_CCOEFF_NORMED)

            locs = np.where(result >= threshold)
            for py, px in zip(*locs):
                score = float(result[py, px])
                cx = px + tw / 2
                cy = py + th / 2
                radius = max(tw, th) / 2
                hits.append((PixelPoint(x=cx, y=cy), score, radius))
        return hits

    # テンプレートごとに並列実行
    with ThreadPoolExecutor(max_workers=len(templates)) as executor:
        futures = {
            executor.submit(
                _match_single_template,
                tmpl_name, tmpl_mask, drawing_mask, scales, threshold, dw, dh,
            ): tmpl_name
            for tmpl_name, tmpl_mask in templates
        }
        for future in as_completed(futures):
            tmpl_name = futures[future]
            hits = future.result()
            print(f"       [INFO] {tmpl_name}: {len(hits)} raw hits")
            all_detections.extend(hits)

    # NMSで重複除去
    all_detections = _nms_points(all_detections, min_dist=min_dist)
    print(f"       [INFO] Template matches after NMS: {len(all_detections)}")

    # デバッグ画像保存
    debug_img = work_img.copy()
    for pt, score, radius in all_detections:
        cv2.drawMarker(debug_img, (int(pt.x), int(pt.y)), (0, 0, 255),
                        cv2.MARKER_CROSS, 12, 2)
        cv2.putText(debug_img, f"{score:.2f}", (int(pt.x) + 8, int(pt.y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    debug_dir = PROJECT_ROOT / "output"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "debug_sleeve_template_match.png"), debug_img)

    # SleeveDetectionに変換
    results: list[SleeveDetection] = []
    for pt, score, radius in all_detections:
        center_global = PixelPoint(x=pt.x + x_offset, y=pt.y + y_offset)

        sleeve_circle = SleeveCircle(
            center_px=center_global,
            radius_px=radius,
            circularity=score,
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
                text_bbox=None,
                component_bbox=comp_bbox,
            )
        )

    return results
