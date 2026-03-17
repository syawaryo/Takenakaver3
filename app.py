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

            from main import analyze, load_image, draw_overlay, draw_reconstruction_map

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

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("元画像")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.subheader("検出結果オーバーレイ")
                    st.image(overlay_rgb, use_container_width=True)
                with col3:
                    st.subheader("再構成マップ")
                    h_img, w_img = img.shape[:2]
                    recon = draw_reconstruction_map(w_img, h_img, result)
                    st.image(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB), use_container_width=True)

                # サマリー
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("通り芯", len(result.grid_lines))
                c2.metric("スリーブ", len(result.sleeves))
                c3.metric("OCRテキスト", len(result.all_texts))
                c4.metric("寸法接続点", len(result.dimension_points))

                if result.px_per_mm:
                    st.info(f"スケール: {result.px_per_mm:.4f} px/mm")

                # ============================================
                # チェックリスト審査
                # ============================================
                from src.checklist_checker import run_checklist

                report = run_checklist(result)

                st.divider()
                st.subheader("施工図チェックリスト審査")

                # サマリーメトリクス
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("OK", report.ok_count)
                rc2.metric("NG", report.ng_count)
                rc3.metric("WARN", report.warn_count)
                rc4.metric("SKIP（検証不可）", report.skip_count)

                # 検証可能な項目の結果
                STATUS_ICON = {"OK": "✅", "NG": "❌", "WARN": "⚠️", "SKIP": "⏭️"}

                # NG/WARN を先に表示
                verifiable = [c for c in report.checks if c.status != "SKIP"]
                skipped = [c for c in report.checks if c.status == "SKIP"]

                for chk in verifiable:
                    icon = STATUS_ICON.get(chk.status, "")
                    with st.expander(
                        f"{icon} [{chk.item_id}] {chk.title} — {chk.status}",
                        expanded=(chk.status == "NG"),
                    ):
                        st.write(chk.detail)
                        if chk.targets:
                            st.write("**対象:**", ", ".join(chk.targets))

                with st.expander("検証不可項目（参考データ不足）", expanded=False):
                    for chk in skipped:
                        st.write(f"⏭️ **[{chk.item_id}] {chk.title}**: {chk.detail}")

                # ============================================
                # スリーブ一覧テーブル
                # ============================================
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
                            "外径": s.parsed.outer_diameter or "",
                            "基準レベル": s.parsed.level_reference or "",
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
