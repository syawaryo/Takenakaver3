"""
通り芯（グリッドライン）検出

モルフォロジー演算で長い直線を抽出し、
射影ヒストグラムでピーク位置を取得。
端部のOCRテキストからラベルを紐付ける。
"""

from __future__ import annotations

import cv2
import numpy as np

from models import GridLine, OcrText, PixelPoint


def _detect_lines_morphology(
    gray: np.ndarray,
    direction: str,
    kernel_ratio: float = 1 / 3,
) -> np.ndarray:
    """モルフォロジーOPENで縦線 or 横線のみを抽出。"""
    h, w = gray.shape[:2]

    # 二値化（OTSU）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if direction == "vertical":
        ksize = max(1, int(h * kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
    else:
        ksize = max(1, int(w * kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 1))

    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


def _find_peaks(projection: np.ndarray, min_distance: int = 20, threshold_ratio: float = 0.3) -> list[int]:
    """射影ヒストグラムからピーク位置を検出。"""
    threshold = projection.max() * threshold_ratio
    peaks: list[int] = []

    in_peak = False
    start = 0
    for i, val in enumerate(projection):
        if val > threshold and not in_peak:
            in_peak = True
            start = i
        elif val <= threshold and in_peak:
            in_peak = False
            peak_pos = (start + i) // 2
            if not peaks or (peak_pos - peaks[-1]) >= min_distance:
                peaks.append(peak_pos)

    # 最後のピーク
    if in_peak:
        peak_pos = (start + len(projection)) // 2
        if not peaks or (peak_pos - peaks[-1]) >= min_distance:
            peaks.append(peak_pos)

    return peaks


def _match_label_to_position(
    position: float,
    direction: str,
    ocr_texts: list[OcrText],
    img_shape: tuple[int, int],
    edge_margin: int = 200,
) -> str | None:
    """
    通り芯端部（画像端の近く）のOCRテキストからラベルを見つける。

    通り芯ラベルは通常:
    - vertical線: 画像の上端 or 下端にある
    - horizontal線: 画像の左端 or 右端にある
    """
    h, w = img_shape
    candidates: list[tuple[float, str]] = []

    for t in ocr_texts:
        text = t.text.strip()
        # 通り芯ラベルっぽいか（X1, Y2, A, B, 1, 2 等）
        if not _is_grid_label(text):
            continue

        tx, ty = t.position_px.x, t.position_px.y

        if direction == "vertical":
            # 端部（上か下）にいるか
            if ty < edge_margin or ty > h - edge_margin:
                # X方向の距離
                dist = abs(tx - position)
                candidates.append((dist, text))
        else:
            # 端部（左か右）にいるか
            if tx < edge_margin or tx > w - edge_margin:
                # Y方向の距離
                dist = abs(ty - position)
                candidates.append((dist, text))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def _is_grid_label(text: str) -> bool:
    """通り芯ラベルっぽいかの厳密な判定。"""
    import re

    text = text.strip()
    if not text:
        return False

    # 厳密なパターンのみ: X1, Y2, X1', A1 等（英字+数字の組み合わせ）
    # 単独の英字や数字は誤検出が多いので除外
    if re.match(r"^[A-Z]\d{1,2}'?$", text):  # X1, Y2 等
        return True
    if re.match(r"^\d{1,2}[A-Z]$", text):  # 1A 等
        return True
    return False


def detect_grid_lines(
    gray: np.ndarray,
    ocr_texts: list[OcrText] | None = None,
    roi: tuple[int, int, int, int] | None = None,
    kernel_ratio: float = 1 / 3,
    min_peak_distance: int = 30,
) -> list[GridLine]:
    """
    通り芯を検出してラベルを紐付ける。

    Parameters
    ----------
    gray : グレースケール画像
    ocr_texts : OCRテキスト（ラベル紐付け用）
    roi : (y_start, y_end, x_start, x_end) 検出対象領域
    kernel_ratio : モルフォロジーカーネルの画像サイズ比
    min_peak_distance : ピーク間の最小距離(px)

    Returns
    -------
    検出された通り芯のリスト
    """
    # ROI適用
    y_off, x_off = 0, 0
    work_gray = gray
    if roi:
        y_start, y_end, x_start, x_end = roi
        work_gray = gray[y_start:y_end, x_start:x_end]
        y_off = y_start
        x_off = x_start

    rh, rw = work_gray.shape[:2]
    h, w = gray.shape[:2]
    results: list[GridLine] = []

    for direction in ("vertical", "horizontal"):
        line_img = _detect_lines_morphology(work_gray, direction, kernel_ratio)

        # 射影ヒストグラム
        if direction == "vertical":
            projection = np.sum(line_img, axis=0) / 255
        else:
            projection = np.sum(line_img, axis=1) / 255

        peaks = _find_peaks(projection, min_peak_distance)

        for pos in peaks:
            # ROI座標 → 元画像座標
            global_pos = float(pos + (x_off if direction == "vertical" else y_off))

            label = None
            if ocr_texts:
                label = _match_label_to_position(
                    global_pos, direction, ocr_texts, (h, w)
                )

            results.append(
                GridLine(
                    label=label or f"{'V' if direction == 'vertical' else 'H'}-{int(global_pos)}",
                    direction=direction,
                    position_px=global_pos,
                    confidence=0.8,
                )
            )

    return results


def compute_scale(
    grid_lines: list[GridLine],
    ocr_texts: list[OcrText],
) -> float | None:
    """
    隣接通り芯間のピクセル距離とOCR寸法値からスケール(px/mm)を算出。

    寸法テキスト例: "3,050" → 3050mm
    """
    import re

    # 寸法テキストを探す（カンマ付き数値 or 4-5桁の数値）
    dimension_values: list[tuple[float, PixelPoint]] = []
    for t in ocr_texts:
        text = t.text.strip().replace(",", "").replace(".", "")
        m = re.match(r"^(\d{3,6})$", text)
        if m:
            dimension_values.append((float(m.group(1)), t.position_px))

    if not dimension_values:
        return None

    # 同方向の隣接グリッド線ペア
    verticals = sorted(
        [g for g in grid_lines if g.direction == "vertical"],
        key=lambda g: g.position_px,
    )

    for i in range(len(verticals) - 1):
        px_dist = verticals[i + 1].position_px - verticals[i].position_px
        mid_x = (verticals[i].position_px + verticals[i + 1].position_px) / 2

        # 2本の通り芯間にある寸法テキストを探す
        for mm_val, pos in dimension_values:
            if verticals[i].position_px < pos.x < verticals[i + 1].position_px:
                if mm_val > 100:  # 妥当な寸法値
                    return px_dist / mm_val

    return None
