"""
床スリーブ図 座標抽出デモ ― データモデル定義

全要素をピクセル座標付きで保持する。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ==========================================================================
# 基本型
# ==========================================================================

class PixelPoint(BaseModel):
    x: float
    y: float


class BBox(BaseModel):
    """バウンディングボックス (左上原点)"""
    x: float
    y: float
    w: float
    h: float


# ==========================================================================
# OCR テキスト
# ==========================================================================

class OcrText(BaseModel):
    text: str = Field(..., description="OCR認識テキスト")
    position_px: PixelPoint = Field(..., description="テキスト中心のピクセル座標")
    bbox: BBox = Field(..., description="テキストのバウンディングボックス")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# ==========================================================================
# 通り芯（グリッド線）
# ==========================================================================

class GridLine(BaseModel):
    label: str = Field(..., description="通り芯ラベル（例: 'X1', 'Y2'）")
    direction: str = Field(..., description="'vertical' or 'horizontal'")
    position_px: float = Field(..., description="ピクセル位置")
    position_mm: Optional[float] = Field(None, description="実寸位置(mm)")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# ==========================================================================
# スリーブ
# ==========================================================================

class SleeveCircle(BaseModel):
    center_px: PixelPoint = Field(..., description="青丸の中心座標")
    radius_px: float = Field(..., description="半径(px)")
    circularity: float = Field(0.0, ge=0.0, le=1.0)
    color_confidence: float = Field(0.0, ge=0.0, le=1.0)


class SleeveAnnotationParsed(BaseModel):
    sleeve_no: Optional[str] = Field(None, description="スリーブ番号 例:'SK-001'")
    purpose: Optional[str] = Field(None, description="用途 例:'排水'")
    nominal_size: Optional[str] = Field(None, description="呼び径 例:'100A'")
    bore_diameter: Optional[str] = Field(None, description="口径 例:'150Φ'")
    outer_diameter: Optional[str] = Field(None, description="外径 例:'155Φ'")
    category: Optional[str] = Field(None, description="設備区分 例:'B'")
    level_reference: Optional[str] = Field(None, description="基準レベル 例:'B1FL+500'")


class Sleeve(BaseModel):
    circle: SleeveCircle = Field(..., description="青丸（OpenCV検出）")
    raw_text: str = Field("", description="OCR生テキスト")
    text_position_px: Optional[PixelPoint] = Field(None, description="テキスト領域の中心座標")
    parsed: SleeveAnnotationParsed = Field(
        default_factory=SleeveAnnotationParsed,
        description="テキスト解析結果",
    )
    slab_id: Optional[str] = Field(None, description="所属スラブID")
    detection_id: str = Field(..., description="自動生成ID 例:'DET-001'")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# ==========================================================================
# 寸法接続点
# ==========================================================================

class DimensionPoint(BaseModel):
    position_px: PixelPoint = Field(..., description="寸法線接続点の座標")
    nearby_text: Optional[str] = Field(None, description="近傍の寸法テキスト")
    connected_grid_label: Optional[str] = Field(None, description="接続先の通り芯ラベル")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# ==========================================================================
# トップレベル解析結果
# ==========================================================================

class FloorSleeveDrawingAnalysis(BaseModel):
    source_file: str
    floor: Optional[str] = None
    image_width_px: int
    image_height_px: int

    grid_lines: list[GridLine] = Field(default_factory=list, description="通り芯")
    sleeves: list[Sleeve] = Field(default_factory=list, description="スリーブ")
    all_texts: list[OcrText] = Field(default_factory=list, description="全OCRテキスト")
    dimension_points: list[DimensionPoint] = Field(
        default_factory=list, description="寸法接続点"
    )

    px_per_mm: Optional[float] = Field(None, description="ピクセル/mm変換係数")
    analyzed_at: datetime = Field(default_factory=datetime.now)
