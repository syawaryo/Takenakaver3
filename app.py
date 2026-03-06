"""
床スリーブ図 座標抽出システム — Streamlit UI
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

# src/ をパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

st.set_page_config(
    page_title="床スリーブ図 座標抽出",
    page_icon="📐",
    layout="wide",
)

st.title("床スリーブ図 座標抽出システム")

# --- サイドバー: オプション ---
with st.sidebar:
    st.header("解析オプション")
    use_vlm = st.checkbox("VLMテキスト解析を使用", value=True,
                          help="OFFにすると正規表現のみでテキスト解析")
    use_nanobanana = st.checkbox("寸法接続点検出 (Gemini)", value=True,
                                help="OFFにすると寸法接続点検出をスキップ")

# --- ファイルアップロード ---
uploaded = st.file_uploader(
    "図面ファイルをアップロード (PNG / PDF)",
    type=["png", "jpg", "jpeg", "pdf"],
)

if uploaded is not None:
    # 一時ファイルに保存
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    if st.button("解析を実行", type="primary"):
        with st.spinner("解析中..."):
            import io
            import contextlib

            from main import analyze, load_image, draw_overlay

            # 一時出力ディレクトリ
            with tempfile.TemporaryDirectory() as out_dir:
                # stdout をキャプチャしてログ表示
                log_buffer = io.StringIO()
                with contextlib.redirect_stdout(log_buffer):
                    result = analyze(
                        path=tmp_path,
                        use_vlm=use_vlm,
                        use_nanobanana=use_nanobanana,
                        output_dir=out_dir,
                    )

                # --- 結果表示 ---
                st.success("解析完了")

                # ログ
                with st.expander("処理ログ", expanded=False):
                    st.code(log_buffer.getvalue())

                # オーバーレイ画像
                import cv2
                import numpy as np

                img = load_image(tmp_path)
                overlay = draw_overlay(img, result)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("元画像")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.subheader("検出結果オーバーレイ")
                    st.image(overlay_rgb, use_container_width=True)

                # サマリー
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("通り芯", len(result.grid_lines))
                c2.metric("スリーブ", len(result.sleeves))
                c3.metric("OCRテキスト", len(result.all_texts))
                c4.metric("寸法接続点", len(result.dimension_points))

                if result.px_per_mm:
                    st.info(f"スケール: {result.px_per_mm:.4f} px/mm")

                # スリーブ一覧テーブル
                if result.sleeves:
                    st.subheader("スリーブ一覧")
                    rows = []
                    for s in result.sleeves:
                        rows.append({
                            "ID": s.detection_id,
                            "スリーブ番号": s.parsed.sleeve_no or "",
                            "用途": s.parsed.purpose or "",
                            "呼び径": s.parsed.nominal_size or "",
                            "口径": s.parsed.bore_diameter or "",
                            "スラブID": s.slab_id or "",
                            "X (px)": f"{s.circle.center_px.x:.0f}",
                            "Y (px)": f"{s.circle.center_px.y:.0f}",
                            "信頼度": f"{s.confidence:.2f}",
                        })
                    st.dataframe(rows, use_container_width=True)

                # JSON ダウンロード
                st.subheader("結果JSON")
                json_str = result.model_dump_json(indent=2)
                st.download_button(
                    "JSONをダウンロード",
                    data=json_str,
                    file_name=f"{Path(uploaded.name).stem}_result.json",
                    mime="application/json",
                )
                with st.expander("JSON プレビュー", expanded=False):
                    st.json(result.model_dump())
