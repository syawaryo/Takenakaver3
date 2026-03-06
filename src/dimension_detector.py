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

TEMPLATE_DIR = PROJECT_ROOT / "docs"
CONNECT_TEMPLATE_FILES = [
    "connecttemplate1.png",
    "connecttemplate2.png",
    "connecttemplate3.png",
    "connecttemplate4.png",
    "connecttemplate5.png",
]


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


def _remove_blue(img_bgr: np.ndarray) -> np.ndarray:
    """画像から青い領域を白に置換（カラーのまま返す）。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
    img_no_blue = img_bgr.copy()
    img_no_blue[blue_mask > 0] = [255, 255, 255]
    return img_no_blue


def _load_connect_templates() -> list[tuple[str, np.ndarray]]:
    """接続点テンプレートを読み込み、青除去（カラー）して返す。"""
    templates = []
    for fname in CONNECT_TEMPLATE_FILES:
        path = TEMPLATE_DIR / fname
        if not path.exists():
            print(f"       [WARN] Connect template not found: {path}")
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"       [WARN] Cannot read connect template: {path}")
            continue
        no_blue = _remove_blue(bgr)
        templates.append((fname, no_blue))
    return templates


def detect_dimension_points(
    drawing_image_bytes: bytes,
    ocr_texts: list[OcrText] | None = None,
    original_size: tuple[int, int] | None = None,
    template_path: str | Path = DEFAULT_TEMPLATE,
    threshold: float = 0.75,
    scales: list[float] | None = None,
) -> list[DimensionPoint]:
    """
    OpenCV マルチテンプレートマッチング（カラー）で寸法線接続点の丸印を検出。

    Parameters
    ----------
    drawing_image_bytes : 図面画像のバイト列
    ocr_texts : OCRテキスト（近傍寸法値の紐付け用）
    original_size : 未使用（互換性のため残す）
    template_path : 旧テンプレート画像パス（フォールバック用）
    threshold : マッチング閾値（0-1、高いほど厳密）
    scales : テンプレートのスケール倍率リスト
    """
    # 図面画像を復元（カラー）
    arr = np.frombuffer(drawing_image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[WARN] Cannot decode drawing image")
        return []

    # 図面画像も青除去（カラーのまま）
    img_no_blue = _remove_blue(img)

    # テンプレート読み込み（connecttemplate1~5 + 旧テンプレート）
    templates = _load_connect_templates()

    # 旧テンプレートもフォールバックとして追加
    old_path = Path(template_path)
    if old_path.exists():
        old_tmpl = cv2.imread(str(old_path))
        if old_tmpl is not None:
            templates.append(("legacy_template", _remove_blue(old_tmpl)))

    if not templates:
        print("       [WARN] No connect templates loaded")
        return []
    print(f"       [INFO] Loaded {len(templates)} connect templates")

    if scales is None:
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

    all_detections: list[tuple[PixelPoint, float]] = []

    for tmpl_name, tmpl_color in templates:
        th, tw = tmpl_color.shape[:2]

        for scale in scales:
            new_w = max(3, int(tw * scale))
            new_h = max(3, int(th * scale))
            scaled_tmpl = cv2.resize(tmpl_color, (new_w, new_h))

            if new_w > img_no_blue.shape[1] or new_h > img_no_blue.shape[0]:
                continue

            # カラー（青除去済み）でテンプレートマッチング
            result = cv2.matchTemplate(img_no_blue, scaled_tmpl, cv2.TM_CCOEFF_NORMED)

            locs = np.where(result >= threshold)
            for py, px in zip(*locs):
                cx = px + new_w / 2
                cy = py + new_h / 2
                score = float(result[py, px])
                all_detections.append((PixelPoint(x=cx, y=cy), score))

    # NMSで重複除去
    all_detections = _nms_points(all_detections, min_dist=8.0)
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
