"""
床スリーブ図 AI チェック システム ― データモデル定義 v3 (デモ用シンプル版)

==========================================================================
デモの目的:
  「通り芯とスリーブの座標がちゃんと取れてる」ことを証明する。

必要なのは2つだけ:
  1. 通り芯（グリッド線）― ラベルと座標
  2. スリーブ（青丸 + 矢印 + テキスト）― 座標とテキスト情報

将来拡張（梁・柱・スラブ・段差・寸法整合チェック等）は
このコアが正しく動いてから載せる。
==========================================================================
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ==========================================================================
# 基本型
# ==========================================================================

class PixelPoint(BaseModel):
    x: float
    y: float


# ==========================================================================
# 通り芯（グリッド線）
# ==========================================================================

class GridLine(BaseModel):
    """
    通り芯: 建物の柱芯グリッドライン。
    図面の端から端まで走る長い基準線。端部に丸囲みラベル（X1, Y1等）。

    検出: OpenCV モルフォロジー演算（画像幅/高さの1/3以上のカーネル）
    ラベル: 端部の丸囲み文字を HoughCircles + Azure DI OCR で読取
    """
    label: str = Field(
        ..., description="通り芯ラベル（例: 'X1', 'Y2'）"
    )
    direction: str = Field(
        ..., description="'vertical'（X系）or 'horizontal'（Y系）"
    )
    position_px: float = Field(
        ..., description="ピクセル位置（vertical=X座標, horizontal=Y座標）"
    )
    position_mm: Optional[float] = Field(
        None, description="実寸位置（mm）。スケール変換後。"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="検出信頼度"
    )


# ==========================================================================
# スリーブ = 青丸 + 矢印 + テキスト の3点セット
# ==========================================================================

class SleeveCircle(BaseModel):
    """
    青丸: スリーブシンボル本体。
    検出: HSV色フィルタ (H=100-130, S=80-255, V=80-255) + findContours
    """
    center_px: PixelPoint = Field(
        ..., description="青丸の中心座標 = スリーブ芯"
    )
    radius_px: float = Field(
        ..., description="半径（px）"
    )
    circularity: float = Field(
        0.0, ge=0.0, le=1.0, description="円形度 (1.0=完全な円)"
    )
    color_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="青色の確信度"
    )


class SleeveAnnotationParsed(BaseModel):
    """
    テキスト解析結果。
    元テキスト例: "SK-001 排水100A-150Φ(155Φ) B"

    各部の意味:
      SK-001   = スリーブ番号（協力会社頭文字-連番）
      排水     = 用途
      100A     = 呼び径（配管の公称サイズ、A=JIS呼称）
      150Φ     = スリーブ口径
      (155Φ)   = スリーブ外径
      B        = 設備区分コード
    """
    sleeve_no: Optional[str] = Field(None, description="スリーブ番号 例:'SK-001'")
    purpose: Optional[str] = Field(None, description="用途 例:'排水','消火','給水'")
    nominal_size: Optional[str] = Field(None, description="呼び径 例:'100A'")
    bore_diameter: Optional[str] = Field(None, description="口径 例:'150Φ'")
    outer_diameter: Optional[str] = Field(None, description="外径 例:'155Φ'")
    category: Optional[str] = Field(None, description="設備区分 例:'B'")
    level_reference: Optional[str] = Field(None, description="基準レベル 例:'B1FL+500'")


class Sleeve(BaseModel):
    """
    スリーブ: 青丸 + 矢印 + テキスト の3点セット。

    図面上の見え方:
      [テキスト: SK-001 排水100A-150Φ(155Φ) B]
            ↑ 矢印（引出線）
            ● 青丸（スリーブシンボル）
    """
    # --- 3点セット ---
    circle: SleeveCircle = Field(
        ..., description="青丸（OpenCV検出）"
    )
    raw_text: str = Field(
        ..., description="OCR生テキスト 例:'SK-001 排水100A-150Φ(155Φ) B'"
    )
    parsed: SleeveAnnotationParsed = Field(
        default_factory=SleeveAnnotationParsed,
        description="テキスト解析結果"
    )

    # --- 位置 ---
    slab_id: Optional[str] = Field(
        None, description="所属スラブID 例:'X1-X2_Y1-Y2'（通り芯から自動算出）"
    )

    # --- メタ ---
    detection_id: str = Field(
        ..., description="自動生成ID 例:'DET-001'"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="総合信頼度"
    )


# ==========================================================================
# トップレベル
# ==========================================================================

class FloorSleeveDrawingAnalysis(BaseModel):
    """
    床スリーブ図の解析結果。
    通り芯 + スリーブ、これだけ。
    """
    # ソース
    source_file: str
    floor: Optional[str] = None
    image_width_px: int
    image_height_px: int

    # コア: 通り芯とスリーブ
    grid_lines: list[GridLine] = Field(
        ..., description="検出された通り芯"
    )
    sleeves: list[Sleeve] = Field(
        ..., description="検出されたスリーブ（青丸+テキスト）"
    )

    # スケール（任意）
    px_per_mm: Optional[float] = Field(
        None, description="ピクセル/mm変換係数（通り芯間距離から算出）"
    )

    # メタ
    analyzed_at: datetime = Field(
        default_factory=datetime.now
    )


# ==========================================================================
# 出力例
# ==========================================================================

EXAMPLE_OUTPUT = {
    "source_file": "B1F床スリーブ図(消).png",
    "floor": "B1F",
    "image_width_px": 1642,
    "image_height_px": 937,

    "grid_lines": [
        {"label": "X1", "direction": "vertical",   "position_px": 180.0, "position_mm": 0.0,    "confidence": 0.95},
        {"label": "X2", "direction": "vertical",   "position_px": 460.0, "position_mm": 3050.0, "confidence": 0.97},
        {"label": "X3", "direction": "vertical",   "position_px": 1020.0,"position_mm": 9130.0, "confidence": 0.96},
        {"label": "Y1", "direction": "horizontal", "position_px": 120.0, "position_mm": 0.0,    "confidence": 0.94},
        {"label": "Y2", "direction": "horizontal", "position_px": 380.0, "position_mm": 2760.0, "confidence": 0.96},
        {"label": "Y3", "direction": "horizontal", "position_px": 720.0, "position_mm": 6560.0, "confidence": 0.93},
    ],

    "sleeves": [
        {
            "circle": {
                "center_px": {"x": 320.5, "y": 290.2},
                "radius_px": 8.5,
                "circularity": 0.94,
                "color_confidence": 0.92,
            },
            "raw_text": "SK-001 排水100A-150Φ(155Φ) B",
            "parsed": {
                "sleeve_no": "SK-001",
                "purpose": "排水",
                "nominal_size": "100A",
                "bore_diameter": "150Φ",
                "outer_diameter": "155Φ",
                "category": "B",
                "level_reference": "B1FL+500",
            },
            "slab_id": "X1-X2_Y1-Y2",
            "detection_id": "DET-001",
            "confidence": 0.93,
        },
        {
            "circle": {
                "center_px": {"x": 410.3, "y": 260.8},
                "radius_px": 7.2,
                "circularity": 0.91,
                "color_confidence": 0.88,
            },
            "raw_text": "SK-002 消火65A-100Φ(105Φ) B",
            "parsed": {
                "sleeve_no": "SK-002",
                "purpose": "消火",
                "nominal_size": "65A",
                "bore_diameter": "100Φ",
                "outer_diameter": "105Φ",
                "category": "B",
                "level_reference": None,
            },
            "slab_id": "X1-X2_Y1-Y2",
            "detection_id": "DET-002",
            "confidence": 0.87,
        },
        {
            "circle": {
                "center_px": {"x": 750.7, "y": 530.4},
                "radius_px": 10.0,
                "circularity": 0.89,
                "color_confidence": 0.85,
            },
            "raw_text": "SK-003 排水150A-200Φ(205Φ) B",
            "parsed": {
                "sleeve_no": "SK-003",
                "purpose": "排水",
                "nominal_size": "150A",
                "bore_diameter": "200Φ",
                "outer_diameter": "205Φ",
                "category": "B",
                "level_reference": "B1FL-550",
            },
            "slab_id": "X2-X3_Y2-Y3",
            "detection_id": "DET-003",
            "confidence": 0.81,
        },
    ],

    "px_per_mm": 0.092,
    "analyzed_at": "2026-03-04T14:30:00",
}


if __name__ == "__main__":
    result = FloorSleeveDrawingAnalysis.model_validate(EXAMPLE_OUTPUT)
    print(result.model_dump_json(indent=2))
    print(f"\n通り芯: {len(result.grid_lines)}本")
    print(f"スリーブ: {len(result.sleeves)}個")
