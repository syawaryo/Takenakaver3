"""
Azure Document Intelligence OCR

prebuilt-layout モデルでテキスト抽出。
座標はOpenCVピクセル座標系に変換して返す。
"""

from __future__ import annotations

import io
import os

from dotenv import load_dotenv

from models import BBox, OcrText, PixelPoint

load_dotenv()


def _polygon_to_bbox_and_center(
    polygon: list[float],
    img_width: int,
    img_height: int,
) -> tuple[BBox, PixelPoint]:
    """
    Azure DI のポリゴン座標（インチ単位）からピクセル座標のBBoxと中心を算出。

    Azure DI の座標系: ポイント座標はインチ単位（72 DPI基準ではなく実座標）。
    ページの width/height (インチ) で正規化してピクセルに変換。
    """
    # polygon = [x1, y1, x2, y2, x3, y3, x4, y4] (4点、インチ単位)
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    bbox = BBox(
        x=min_x,
        y=min_y,
        w=max_x - min_x,
        h=max_y - min_y,
    )
    center = PixelPoint(x=center_x, y=center_y)
    return bbox, center


def run_azure_ocr(
    image_bytes: bytes,
    img_width: int,
    img_height: int,
) -> list[OcrText]:
    """
    Azure Document Intelligence でOCR実行。

    Parameters
    ----------
    image_bytes : PNG/JPEG 画像バイト列
    img_width : 画像幅(px) — 座標変換用
    img_height : 画像高さ(px) — 座標変換用

    Returns
    -------
    OcrText のリスト（ピクセル座標付き）
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential

    endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
    key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    poller = client.begin_analyze_document(
        "prebuilt-layout",
        body=io.BytesIO(image_bytes),
        content_type="image/png",
    )
    result = poller.result()

    texts: list[OcrText] = []

    if not result.pages:
        return texts

    page = result.pages[0]
    # ページサイズ（インチ）
    page_w_inch = page.width or 1.0
    page_h_inch = page.height or 1.0

    # インチ → ピクセル変換係数
    scale_x = img_width / page_w_inch
    scale_y = img_height / page_h_inch

    if page.words:
        for word in page.words:
            if not word.polygon or len(word.polygon) < 8:
                continue

            bbox_inch, center_inch = _polygon_to_bbox_and_center(
                word.polygon, img_width, img_height
            )

            # インチ座標 → ピクセル座標
            center_px = PixelPoint(
                x=center_inch.x * scale_x,
                y=center_inch.y * scale_y,
            )
            bbox_px = BBox(
                x=bbox_inch.x * scale_x,
                y=bbox_inch.y * scale_y,
                w=bbox_inch.w * scale_x,
                h=bbox_inch.h * scale_y,
            )

            texts.append(
                OcrText(
                    text=word.content,
                    position_px=center_px,
                    bbox=bbox_px,
                    confidence=word.confidence or 0.0,
                )
            )

    return texts
