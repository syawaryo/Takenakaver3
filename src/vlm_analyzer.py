"""
スリーブテキスト解析

正規表現フォールバック + Gemini VLM での構造化パース。
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from models import SleeveAnnotationParsed

load_dotenv()


def parse_annotation_regex(text: str) -> SleeveAnnotationParsed:
    """
    正規表現でスリーブテキストをパース。

    例: "SK-001 排水100A-150Φ(155Φ) B"
    """
    result = SleeveAnnotationParsed()

    if not text:
        return result

    # スリーブ番号: 英字1-3文字 + "-" + 数字2-4桁
    m = re.search(r"([A-Za-z]{1,4}[-\s]?\d{1,4})", text)
    if m:
        result.sleeve_no = m.group(1).strip()

    # 用途: 日本語キーワード
    purposes = ["排水", "消火", "給水", "空調", "通気", "給湯", "冷媒", "ガス", "電気", "冷温水"]
    for p in purposes:
        if p in text:
            result.purpose = p
            break

    # 呼び径: 数字+A
    m = re.search(r"(\d{2,4})A", text)
    if m:
        result.nominal_size = m.group(1) + "A"

    # 口径: 数字+Φ (括弧外)
    m = re.search(r"(?<!\()(\d{2,4})[Φφ]", text)
    if m:
        result.bore_diameter = m.group(1) + "Φ"

    # 外径: (数字Φ)
    m = re.search(r"\((\d{2,4})[Φφ]\)", text)
    if m:
        result.outer_diameter = m.group(1) + "Φ"

    # 設備区分: 末尾の単独英字
    m = re.search(r"\s([A-Z])\s*$", text)
    if m:
        result.category = m.group(1)

    # 基準レベル: B?FL±数字
    m = re.search(r"(B?\d?FL[+\-±]\d+)", text)
    if m:
        result.level_reference = m.group(1)

    return result


def parse_annotations_vlm(
    image_bytes: bytes,
    sleeve_texts: list[str],
) -> list[SleeveAnnotationParsed]:
    """
    Gemini VLM でスリーブテキストを一括構造化パース。

    画像全体とテキスト一覧を渡し、各テキストの解析結果を返す。
    失敗時は正規表現フォールバック。
    """
    import json

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[WARN] GEMINI_API_KEY not set, using regex fallback")
        return [parse_annotation_regex(t) for t in sleeve_texts]

    client = genai.Client(api_key=api_key)

    prompt = f"""以下は建築図面のスリーブ注釈テキストです。各テキストを構造化してください。

テキスト一覧:
{json.dumps(sleeve_texts, ensure_ascii=False, indent=2)}

各テキストについて以下のJSON形式で返してください:
[
  {{
    "sleeve_no": "スリーブ番号 (例: SK-001)",
    "purpose": "用途 (例: 排水, 消火, 給水)",
    "nominal_size": "呼び径 (例: 100A)",
    "bore_diameter": "口径 (例: 150Φ)",
    "outer_diameter": "外径 (例: 155Φ)",
    "category": "設備区分 (例: B)",
    "level_reference": "基準レベル (例: B1FL+500)"
  }}
]

JSONのみ返してください。"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        types.Part.from_text(text=prompt),
                    ]
                )
            ],
        )

        # JSONを抽出
        response_text = response.text.strip()
        # マークダウンコードブロックを除去
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        parsed_list = json.loads(response_text)

        results: list[SleeveAnnotationParsed] = []
        for item in parsed_list:
            results.append(SleeveAnnotationParsed(**{
                k: v for k, v in item.items()
                if k in SleeveAnnotationParsed.model_fields and v
            }))

        return results

    except Exception as e:
        print(f"[WARN] VLM parse failed: {e}, using regex fallback")
        return [parse_annotation_regex(t) for t in sleeve_texts]
