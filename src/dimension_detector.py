"""
寸法線接続点（丸印）検出 — OpenCVテンプレートマッチング方式

テンプレート画像の丸印を図面上でマルチスケールマッチング。
Gemini画像生成は精度が出ないため不採用。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from models import DimensionPoint, OcrText, PixelPoint

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "template" / "image.png"


def _match_nearby_text(
    point: PixelPoint,
    ocr_texts: list[OcrText],
    max_distance: float = 80.0,
) -> str | None:
    """接続点の近傍にある数値テキスト（寸法値）を見つける。"""
    import re

    best_dist = max_distance
    best_text = None

    for t in ocr_texts:
        cleaned = t.text.strip().replace(",", "")
        if not re.match(r"^\d{2,6}$", cleaned):
            continue
        dist = ((point.x - t.position_px.x) ** 2 + (point.y - t.position_px.y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_text = t.text.strip()

    return best_text


def _nms_points(points: list[tuple[PixelPoint, float]], min_dist: float = 15.0) -> list[tuple[PixelPoint, float]]:
    """近接する検出点を統合（Non-Maximum Suppression）。"""
    if not points:
        return points

    # スコア降順ソート
    points.sort(key=lambda x: x[1], reverse=True)
    kept: list[tuple[PixelPoint, float]] = []

    for pt, score in points:
        too_close = False
        for kpt, _ in kept:
            d = ((pt.x - kpt.x) ** 2 + (pt.y - kpt.y) ** 2) ** 0.5
            if d < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append((pt, score))

    return kept


def detect_dimension_points(
    drawing_image_bytes: bytes,
    ocr_texts: list[OcrText] | None = None,
    original_size: tuple[int, int] | None = None,
    template_path: str | Path = DEFAULT_TEMPLATE,
    threshold: float = 0.75,
    scales: list[float] | None = None,
) -> list[DimensionPoint]:
    """
    OpenCV テンプレートマッチングで寸法線接続点の丸印を検出。

    Parameters
    ----------
    drawing_image_bytes : 図面画像のバイト列
    ocr_texts : OCRテキスト（近傍寸法値の紐付け用）
    original_size : 未使用（互換性のため残す）
    template_path : テンプレート画像パス
    threshold : マッチング閾値（0-1、高いほど厳密）
    scales : テンプレートのスケール倍率リスト
    """
    template_path = Path(template_path)
    if not template_path.exists():
        print(f"[WARN] Template not found: {template_path}, skipping")
        return []

    # テンプレート読み込み
    tmpl_color = cv2.imread(str(template_path))
    if tmpl_color is None:
        print(f"[WARN] Cannot read template: {template_path}")
        return []
    tmpl_gray = cv2.cvtColor(tmpl_color, cv2.COLOR_BGR2GRAY)

    # 図面画像を復元
    arr = np.frombuffer(drawing_image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[WARN] Cannot decode drawing image")
        return []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if scales is None:
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

    all_detections: list[tuple[PixelPoint, float]] = []

    for scale in scales:
        # テンプレートをリサイズ
        th, tw = tmpl_gray.shape[:2]
        new_w = max(3, int(tw * scale))
        new_h = max(3, int(th * scale))
        scaled_tmpl = cv2.resize(tmpl_gray, (new_w, new_h))

        # 図面が小さすぎたらスキップ
        if new_w > img_gray.shape[1] or new_h > img_gray.shape[0]:
            continue

        # テンプレートマッチング
        result = cv2.matchTemplate(img_gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED)

        # 閾値以上の位置を取得
        locs = np.where(result >= threshold)
        for py, px in zip(*locs):
            cx = px + new_w / 2
            cy = py + new_h / 2
            score = float(result[py, px])
            all_detections.append((PixelPoint(x=cx, y=cy), score))

    # NMSで重複除去
    all_detections = _nms_points(all_detections, min_dist=12.0)
    print(f"       [INFO] Template matches after NMS: {len(all_detections)}")

    # デバッグ画像保存
    debug_img = img.copy()
    for pt, score in all_detections:
        cv2.drawMarker(debug_img, (int(pt.x), int(pt.y)), (0, 0, 255),
                       cv2.MARKER_CROSS, 12, 2)
        cv2.putText(debug_img, f"{score:.2f}", (int(pt.x) + 8, int(pt.y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.imwrite("output/debug_template_match.png", debug_img)

    # 結果構築
    results: list[DimensionPoint] = []
    for pt, score in all_detections:
        nearby = None
        if ocr_texts:
            nearby = _match_nearby_text(pt, ocr_texts)

        results.append(
            DimensionPoint(
                position_px=pt,
                nearby_text=nearby,
                confidence=score,
            )
        )

    return results
