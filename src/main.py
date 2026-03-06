"""
床スリーブ図 座標抽出デモ — パイプライン + CLI

使い方:
    python src/main.py <図面ファイル>
    python src/main.py drawing.pdf --no-vlm
    python src/main.py drawing.png --no-nanobanana
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

# src/ をパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid_detector import compute_scale, detect_grid_lines
from models import (
    FloorSleeveDrawingAnalysis,
    OcrText,
    PixelPoint,
    Sleeve,
    SleeveAnnotationParsed,
    SleeveCircle,
)
from sleeve_detector import detect_sleeves_with_annotations

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================================
# 描画領域（ROI）の自動検出
# ==========================================================================

def detect_drawing_roi(img: np.ndarray) -> tuple[int, int, int, int]:
    """
    CADスクリーンショットから図面描画領域のROIを推定。
    ツールバーやステータスバーを除外する。

    方法: ツールバーの区切り行（明るく均一な行）の最後を見つけて、
    その直後から図面領域とする。

    Returns: (y_start, y_end, x_start, x_end)
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    row_std = np.std(gray.astype(float), axis=1)
    bright_pct = np.array([np.sum(gray[y, :] > 230) / w for y in range(h)])

    # 上端: y<200 の範囲で、最後の「明るく均一な行」(bright%>90, std<12)を見つける
    last_toolbar_row = 0
    for y in range(min(200, h)):
        if bright_pct[y] > 0.90 and row_std[y] < 12:
            last_toolbar_row = y

    y_start = last_toolbar_row + 1

    # 下端: 下からスキャンして最初の「明るく均一な行」を見つける（ステータスバーの上端）
    y_end = h
    for y in range(h - 1, max(h - 100, y_start), -1):
        if bright_pct[y] > 0.90 and row_std[y] < 12:
            y_end = y
            break

    return (y_start, y_end, 0, w)


# ==========================================================================
# 画像読込
# ==========================================================================

def load_image(path: str) -> np.ndarray:
    """PNG/PDF を読み込んで BGR numpy array を返す。"""
    p = Path(path)

    if p.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(str(p), dpi=300)
        if not pages:
            raise ValueError(f"PDF has no pages: {path}")
        # 1ページ目のみ
        pil_img = pages[0]
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return img


def image_to_bytes(img: np.ndarray) -> bytes:
    """BGR numpy array → PNG bytes"""
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ==========================================================================
# テキスト紐付け
# ==========================================================================

def _match_ocr_to_bbox(
    ocr_texts: list[OcrText],
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    margin: float = 10.0,
) -> list[OcrText]:
    """BBox内 (マージン含む) にあるOCRテキストを返す。"""
    matched: list[OcrText] = []
    for t in ocr_texts:
        tx, ty = t.position_px.x, t.position_px.y
        if (bbox_x - margin <= tx <= bbox_x + bbox_w + margin and
                bbox_y - margin <= ty <= bbox_y + bbox_h + margin):
            matched.append(t)
    return matched


def _assign_slab_id(
    center: PixelPoint,
    grid_lines: list,
) -> str | None:
    """スリーブ中心座標から所属スラブIDを算出。"""
    from models import GridLine

    verticals = sorted(
        [g for g in grid_lines if g.direction == "vertical"],
        key=lambda g: g.position_px,
    )
    horizontals = sorted(
        [g for g in grid_lines if g.direction == "horizontal"],
        key=lambda g: g.position_px,
    )

    v_label = None
    for i in range(len(verticals) - 1):
        if verticals[i].position_px <= center.x <= verticals[i + 1].position_px:
            v_label = f"{verticals[i].label}-{verticals[i + 1].label}"
            break

    h_label = None
    for i in range(len(horizontals) - 1):
        if horizontals[i].position_px <= center.y <= horizontals[i + 1].position_px:
            h_label = f"{horizontals[i].label}-{horizontals[i + 1].label}"
            break

    if v_label and h_label:
        return f"{v_label}_{h_label}"
    return v_label or h_label


# ==========================================================================
# オーバーレイ描画
# ==========================================================================

def draw_overlay(img: np.ndarray, result: FloorSleeveDrawingAnalysis) -> np.ndarray:
    """解析結果をオーバーレイ描画。"""
    overlay = img.copy()
    h, w = overlay.shape[:2]

    # 通り芯: 緑線 + ラベル
    for g in result.grid_lines:
        color = (0, 200, 0)
        if g.direction == "vertical":
            x = int(g.position_px)
            cv2.line(overlay, (x, 0), (x, h), color, 1)
            cv2.putText(overlay, g.label, (x + 3, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            y = int(g.position_px)
            cv2.line(overlay, (0, y), (w, y), color, 1)
            cv2.putText(overlay, g.label, (5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # スリーブ: 赤丸 + テキストボックス
    for s in result.sleeves:
        cx = int(s.circle.center_px.x)
        cy = int(s.circle.center_px.y)
        r = max(3, int(s.circle.radius_px))

        # 赤丸
        cv2.circle(overlay, (cx, cy), r, (0, 0, 255), 2)
        # 十字
        cv2.drawMarker(overlay, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 6, 1)

        # テキスト（SK番号のみ）
        label = s.parsed.sleeve_no or s.detection_id
        cv2.putText(overlay, label, (cx + r + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # 寸法接続点: 黄色マーカー（数値テキストは非表示）
    for dp in result.dimension_points:
        px = int(dp.position_px.x)
        py = int(dp.position_px.y)
        cv2.drawMarker(overlay, (px, py), (0, 255, 255),
                       cv2.MARKER_DIAMOND, 8, 2)

    return overlay


def draw_reconstruction_map(
    w: int, h: int, result: FloorSleeveDrawingAnalysis,
) -> np.ndarray:
    """
    白背景に検出結果だけで図面を再構成する。
    通り芯 + スリーブ + 接続点 + スリーブ間距離を1枚にまとめる。
    """
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # --- 通り芯（薄いグレー線 + ラベル）---
    grid_color = (180, 180, 180)
    label_color = (100, 100, 100)
    for g in result.grid_lines:
        if g.direction == "vertical":
            x = int(g.position_px)
            cv2.line(canvas, (x, 0), (x, h), grid_color, 1)
            cv2.putText(canvas, g.label, (x + 3, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)
        else:
            y = int(g.position_px)
            cv2.line(canvas, (0, y), (w, y), grid_color, 1)
            cv2.putText(canvas, g.label, (5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1)

    # --- 通り芯付近の寸法値を表示 ---
    import re as _re_grid
    dim_value_color = (60, 60, 200)
    margin_px = 20  # 通り芯からの検索範囲(px)

    for g in result.grid_lines:
        pos = g.position_px
        for t in result.all_texts:
            cleaned = t.text.strip().replace(",", "").replace(".", "")
            if not _re_grid.match(r"^\d{3,4}$", cleaned):
                continue
            tx, ty = t.position_px.x, t.position_px.y
            if g.direction == "vertical" and abs(tx - pos) < margin_px:
                cv2.putText(canvas, t.text.strip(), (int(tx), int(ty)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim_value_color, 1)
            elif g.direction == "horizontal" and abs(ty - pos) < margin_px:
                cv2.putText(canvas, t.text.strip(), (int(tx), int(ty)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim_value_color, 1)

    # --- スリーブ（青丸 + SK番号 + スペック）---
    sleeve_color = (200, 80, 0)  # 青系
    sleeve_positions: list[tuple[int, int, str]] = []
    for s in result.sleeves:
        cx = int(s.circle.center_px.x)
        cy = int(s.circle.center_px.y)
        r = max(5, int(s.circle.radius_px))

        cv2.circle(canvas, (cx, cy), r, sleeve_color, 2)
        cv2.drawMarker(canvas, (cx, cy), sleeve_color, cv2.MARKER_CROSS, 8, 1)

        label = s.parsed.sleeve_no or s.detection_id
        cv2.putText(canvas, label, (cx + r + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, sleeve_color, 1)

        sleeve_positions.append((cx, cy, label))

    # --- 接続点（黄色マーカーのみ）---
    dim_color = (0, 180, 180)
    for dp in result.dimension_points:
        px = int(dp.position_px.x)
        py = int(dp.position_px.y)
        cv2.drawMarker(canvas, (px, py), dim_color, cv2.MARKER_DIAMOND, 10, 2)


    return canvas


# ==========================================================================
# メインパイプライン
# ==========================================================================

def analyze(
    path: str,
    use_vlm: bool = True,
    use_nanobanana: bool = True,
    output_dir: str = "output",
) -> FloorSleeveDrawingAnalysis:
    """図面解析パイプライン。"""
    print(f"[1/5] Loading image: {path}")
    img = load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    img_bytes = image_to_bytes(img)

    print(f"       Image size: {w} x {h}")

    # --- OCR ---
    print("[2/5] Running Azure DI OCR...")
    try:
        from ocr_extractor import run_azure_ocr
        ocr_texts = run_azure_ocr(img_bytes, w, h)
        print(f"       Found {len(ocr_texts)} text elements")
    except Exception as e:
        print(f"       [WARN] OCR failed: {e}")
        ocr_texts = []

    # --- 通り芯検出 ---
    print("[3/5] Detecting grid lines...")
    grid_lines = detect_grid_lines(gray, ocr_texts)
    print(f"       Found {len(grid_lines)} grid lines")

    # --- スリーブ検出 + 寸法接続点検出（並列実行）---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_sleeve_detection():
        print("[4/5] Detecting sleeves (template matching)...")
        dets = detect_sleeves_with_annotations(img)
        print(f"       Found {len(dets)} sleeve candidates")
        return dets

    def _run_dimension_detection():
        print("[5/5] Detecting dimension connection points...")
        from dimension_detector import detect_dimension_points
        pts = detect_dimension_points(img_bytes, ocr_texts, original_size=(w, h))
        print(f"       Found {len(pts)} dimension points (before filtering)")
        return pts

    sleeve_detections = []
    dim_points = []

    if use_nanobanana:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_sleeve = executor.submit(_run_sleeve_detection)
            fut_dim = executor.submit(_run_dimension_detection)
            try:
                sleeve_detections = fut_sleeve.result()
            except Exception as e:
                print(f"       [WARN] Sleeve detection failed: {e}")
            try:
                dim_points = fut_dim.result()
            except Exception as e:
                print(f"       [WARN] Dimension detection failed: {e}")

        # スリーブ位置と重複する接続点を除外
        filtered = []
        for dp in dim_points:
            too_close = False
            for det in sleeve_detections:
                sx = det.circle.center_px.x
                sy = det.circle.center_px.y
                sr = det.circle.radius_px
                dist = ((dp.position_px.x - sx) ** 2 + (dp.position_px.y - sy) ** 2) ** 0.5
                if dist < sr:
                    too_close = True
                    break
            if not too_close:
                filtered.append(dp)
        print(f"       After sleeve overlap filter: {len(filtered)} dimension points")
        dim_points = filtered
    else:
        sleeve_detections = _run_sleeve_detection()
        print("[5/5] Skipping dimension detection (--no-nanobanana)")

    # --- テキスト紐付け + スリーブ構築 ---
    print("[6/6] Matching text to sleeves...")
    import re as _re

    # Step 1: スリーブ番号テキスト(SK-xxx)を先にスリーブ円に排他的に割り当て
    # 各SK-xxxテキストを最も近い円に紐付け
    sk_pattern = _re.compile(r"[A-Za-z]{1,4}[-\s]?\d{1,4}")
    sk_texts: list[tuple[int, OcrText]] = []  # (index, text)
    for ti, t in enumerate(ocr_texts):
        if sk_pattern.search(t.text.strip()):
            sk_texts.append((ti, t))

    # 各スリーブ検出に最も近いSK-xxxテキストを割り当て
    sleeve_sk_assignment: dict[int, tuple[int, OcrText]] = {}  # det_idx -> (text_idx, text)
    claimed_sk: set[int] = set()

    for det_idx, det in enumerate(sleeve_detections):
        cx, cy = det.circle.center_px.x, det.circle.center_px.y
        best_dist = 200.0  # 最大検索距離
        best = None
        for ti, t in sk_texts:
            if ti in claimed_sk:
                continue
            dist = ((cx - t.position_px.x) ** 2 + (cy - t.position_px.y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = (ti, t)
        if best:
            sleeve_sk_assignment[det_idx] = best
            claimed_sk.add(best[0])

    # Step 2: SKテキストから同一行のテキストをまとめて取得
    # 同じy座標 + x座標が連続しているテキストを1行として収集
    line_y_tolerance = 15   # 同一行とみなすy座標の許容差(px)
    line_x_gap_max = 150    # x方向の最大ギャップ(px) — これ以上離れていたら別行とみなす

    def _collect_line_texts(anchor_ti: int, anchor_text: OcrText) -> tuple[list[OcrText], set[int]]:
        """SKテキストと同一行のテキストをx連続性を考慮して収集。"""
        anchor_y = anchor_text.position_px.y
        # 同じy座標のテキスト候補を集める
        candidates: list[tuple[float, int, OcrText]] = [(anchor_text.position_px.x, anchor_ti, anchor_text)]
        for tj, t2 in enumerate(ocr_texts):
            if tj == anchor_ti:
                continue
            if abs(t2.position_px.y - anchor_y) < line_y_tolerance:
                candidates.append((t2.position_px.x, tj, t2))
        # x座標順にソート
        candidates.sort(key=lambda c: c[0])

        # x連続性チェック: アンカーから左右に連続しているもののみ採用
        anchor_pos = next(i for i, c in enumerate(candidates) if c[1] == anchor_ti)
        line: list[tuple[float, int, OcrText]] = [candidates[anchor_pos]]
        # 右方向
        for i in range(anchor_pos + 1, len(candidates)):
            if candidates[i][0] - candidates[i - 1][0] < line_x_gap_max:
                line.append(candidates[i])
            else:
                break
        # 左方向
        for i in range(anchor_pos - 1, -1, -1):
            if candidates[i + 1][0] - candidates[i][0] < line_x_gap_max:
                line.insert(0, candidates[i])
            else:
                break

        line.sort(key=lambda c: c[0])
        texts = [t for _, _, t in line]
        indices = {idx for _, idx, _ in line}
        return texts, indices

    claimed_texts: set[int] = set(claimed_sk)
    sleeves: list[Sleeve] = []
    sleeve_raw_texts: list[str] = []

    for idx, det in enumerate(sleeve_detections):
        cx, cy = det.circle.center_px.x, det.circle.center_px.y

        matched_texts: list[OcrText] = []
        if idx in sleeve_sk_assignment:
            sk_ti, sk_text = sleeve_sk_assignment[idx]
            # SK行を丸ごと取得
            line_texts, line_indices = _collect_line_texts(sk_ti, sk_text)
            matched_texts = line_texts
            claimed_texts.update(line_indices)
        else:
            # SKテキストなし → 円の近傍テキストをフォールバック収集
            nearby: list[tuple[float, int, OcrText]] = []
            for ti, t in enumerate(ocr_texts):
                if ti in claimed_texts:
                    continue
                dist = ((cx - t.position_px.x) ** 2 + (cy - t.position_px.y) ** 2) ** 0.5
                if dist < 120.0:
                    nearby.append((dist, ti, t))
            nearby.sort(key=lambda c: c[0])
            for dist, ti, t in nearby[:8]:
                matched_texts.append(t)
                claimed_texts.add(ti)

        text_center = None
        if matched_texts:
            text_center = PixelPoint(
                x=matched_texts[0].position_px.x,
                y=matched_texts[0].position_px.y,
            )

        raw_text = " ".join(t.text for t in matched_texts)
        sleeve_raw_texts.append(raw_text)
        slab_id = _assign_slab_id(det.circle.center_px, grid_lines)

        sleeves.append(
            Sleeve(
                circle=det.circle,
                raw_text=raw_text,
                text_position_px=text_center,
                parsed=SleeveAnnotationParsed(),
                slab_id=slab_id,
                detection_id=f"DET-{idx + 1:03d}",
                confidence=det.circle.circularity * det.circle.color_confidence,
            )
        )

    # --- テキスト解析（VLM or 正規表現）---
    if sleeve_raw_texts:
        from vlm_analyzer import parse_annotation_regex

        if use_vlm:
            print("       Parsing sleeve texts with VLM...")
            try:
                from vlm_analyzer import parse_annotations_vlm
                parsed_list = parse_annotations_vlm(img_bytes, sleeve_raw_texts)
                for i, s in enumerate(sleeves):
                    if i < len(parsed_list):
                        s.parsed = parsed_list[i]
                    # VLMの結果が空ならregexフォールバック
                    if not s.parsed.sleeve_no and s.raw_text:
                        s.parsed = parse_annotation_regex(s.raw_text)
            except Exception as e:
                print(f"       [WARN] VLM parse failed: {e}, using regex")
                for s in sleeves:
                    s.parsed = parse_annotation_regex(s.raw_text)
        else:
            print("       Parsing sleeve texts with regex...")
            for s in sleeves:
                s.parsed = parse_annotation_regex(s.raw_text)

    # --- スケール算出 ---
    px_per_mm = compute_scale(grid_lines, ocr_texts) if ocr_texts else None

    # --- フロア名抽出 ---
    floor_name = None
    source_name = Path(path).stem
    import re
    m = re.search(r"(B?\d+F)", source_name)
    if m:
        floor_name = m.group(1)

    result = FloorSleeveDrawingAnalysis(
        source_file=Path(path).name,
        floor=floor_name,
        image_width_px=w,
        image_height_px=h,
        grid_lines=grid_lines,
        sleeves=sleeves,
        all_texts=ocr_texts,
        dimension_points=dim_points,
        px_per_mm=px_per_mm,
        analyzed_at=datetime.now(),
    )

    # --- 出力 ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON出力
    json_path = out_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    print(f"\n[OUTPUT] {json_path}")

    # オーバーレイPNG出力
    overlay = draw_overlay(img, result)
    overlay_path = out_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"[OUTPUT] {overlay_path}")

    # 再構成マップ出力（白背景に検出結果だけで図面を再現）
    recon_map = draw_reconstruction_map(w, h, result)
    recon_path = out_dir / "reconstruction_map.png"
    cv2.imwrite(str(recon_path), recon_map)
    print(f"[OUTPUT] {recon_path}")

    # サマリー
    print(f"\n=== Summary ===")
    print(f"  Grid lines : {len(result.grid_lines)}")
    print(f"  Sleeves    : {len(result.sleeves)}")
    print(f"  OCR texts  : {len(result.all_texts)}")
    print(f"  Dim points : {len(result.dimension_points)}")
    if px_per_mm:
        print(f"  Scale      : {px_per_mm:.4f} px/mm")

    return result


# ==========================================================================
# CLI
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="床スリーブ図 座標抽出デモ",
    )
    parser.add_argument(
        "drawing",
        help="図面ファイルパス (PNG or PDF)",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="VLMによるテキスト解析をスキップ（正規表現のみ）",
    )
    parser.add_argument(
        "--no-nanobanana",
        action="store_true",
        help="Gemini Nano Banana Pro による寸法接続点検出をスキップ",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="出力ディレクトリ（デフォルト: output/）",
    )

    args = parser.parse_args()

    analyze(
        path=args.drawing,
        use_vlm=not args.no_vlm,
        use_nanobanana=not args.no_nanobanana,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
