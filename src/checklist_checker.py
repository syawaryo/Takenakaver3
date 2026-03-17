"""
設備施工図チェックリスト審査モジュール

チェックリスト（docs/260127_設備施工図チェックリスト.pdf）に基づき、
解析結果を自動審査する。
"""

from __future__ import annotations

from src.models import (
    ChecklistReport,
    CheckResult,
    FloorSleeveDrawingAnalysis,
    Sleeve,
)


def run_checklist(result: FloorSleeveDrawingAnalysis) -> ChecklistReport:
    """全チェック項目を実行してレポートを返す。"""
    checks: list[CheckResult] = []

    checks.append(_check_sleeve_no(result))
    checks.append(_check_purpose(result))
    checks.append(_check_diameter(result))

    # 検証不可能な項目（情報提供のみ）
    checks.extend(_skipped_items())

    ok = sum(1 for c in checks if c.status == "OK")
    ng = sum(1 for c in checks if c.status == "NG")
    warn = sum(1 for c in checks if c.status == "WARN")
    skip = sum(1 for c in checks if c.status == "SKIP")

    return ChecklistReport(
        checks=checks,
        total=len(checks),
        ok_count=ok,
        ng_count=ng,
        warn_count=warn,
        skip_count=skip,
    )


# ==========================================================================
# 個別チェック関数
# ==========================================================================

def _sleeve_label(s: Sleeve) -> str:
    return s.parsed.sleeve_no or s.detection_id


def _check_sleeve_no(result: FloorSleeveDrawingAnalysis) -> CheckResult:
    """C-01: スリーブNoは記載しているか。例）SD-001"""
    missing = [
        _sleeve_label(s) for s in result.sleeves
        if not s.parsed.sleeve_no
    ]
    if not result.sleeves:
        return CheckResult(
            item_id="C-01", title="スリーブNo記載",
            status="SKIP", detail="スリーブが検出されていません",
        )
    if missing:
        return CheckResult(
            item_id="C-01", title="スリーブNo記載",
            status="NG",
            detail=f"{len(missing)}件のスリーブでNo未記載",
            targets=missing,
        )
    return CheckResult(
        item_id="C-01", title="スリーブNo記載",
        status="OK", detail="全スリーブにNo記載あり",
    )


def _check_purpose(result: FloorSleeveDrawingAnalysis) -> CheckResult:
    """C-02: スリーブ用途・利用設備種別を記載したか。"""
    missing = [
        _sleeve_label(s) for s in result.sleeves
        if not s.parsed.purpose
    ]
    if not result.sleeves:
        return CheckResult(
            item_id="C-02", title="用途・設備種別記載",
            status="SKIP", detail="スリーブが検出されていません",
        )
    if missing:
        return CheckResult(
            item_id="C-02", title="用途・設備種別記載",
            status="NG",
            detail=f"{len(missing)}件のスリーブで用途未記載",
            targets=missing,
        )
    return CheckResult(
        item_id="C-02", title="用途・設備種別記載",
        status="OK", detail="全スリーブに用途記載あり",
    )


def _check_diameter(result: FloorSleeveDrawingAnalysis) -> CheckResult:
    """C-03: スリーブ呼び口径及び外径を記載したか。例：200Φ（外径216）"""
    missing_bore = []
    missing_outer = []
    for s in result.sleeves:
        label = _sleeve_label(s)
        has_bore = bool(s.parsed.bore_diameter or s.parsed.nominal_size)
        has_outer = bool(s.parsed.outer_diameter)
        if not has_bore:
            missing_bore.append(label)
        if not has_outer:
            missing_outer.append(label)

    if not result.sleeves:
        return CheckResult(
            item_id="C-03", title="呼び口径・外径記載",
            status="SKIP", detail="スリーブが検出されていません",
        )

    issues = []
    targets = []
    if missing_bore:
        issues.append(f"口径未記載: {len(missing_bore)}件")
        targets.extend(missing_bore)
    if missing_outer:
        issues.append(f"外径未記載: {len(missing_outer)}件")
        targets.extend(missing_outer)

    if missing_bore:
        return CheckResult(
            item_id="C-03", title="呼び口径・外径記載",
            status="NG", detail="、".join(issues),
            targets=list(set(targets)),
        )
    if missing_outer:
        return CheckResult(
            item_id="C-03", title="呼び口径・外径記載",
            status="WARN",
            detail=f"外径未記載: {len(missing_outer)}件（口径は記載あり）",
            targets=missing_outer,
        )
    return CheckResult(
        item_id="C-03", title="呼び口径・外径記載",
        status="OK", detail="全スリーブに口径・外径記載あり",
    )




def _skipped_items() -> list[CheckResult]:
    """検証不可能な項目（参考データ不足等）"""
    return [
        CheckResult(
            item_id="C-04", title="基準レベル記載",
            status="SKIP",
            detail="床スリーブ図にレベル情報が記載されていないケースが多く、スリーブ図単体での判断が困難",
        ),
        CheckResult(
            item_id="C-05", title="通り芯寸法整合性",
            status="SKIP",
            detail="寸法線とスリーブ・通り芯の紐付け精度が不十分なため検証不可",
        ),
        CheckResult(
            item_id="C-06", title="寄り寸法の基準通り芯",
            status="SKIP",
            detail="寸法接続点と通り芯の対応関係の解析精度が不十分なため検証不可",
        ),
        CheckResult(
            item_id="C-07", title="スリーブ芯基準の寄り寸法",
            status="SKIP",
            detail="寸法接続点と通り芯の対応関係の解析精度が不十分なため検証不可",
        ),
        CheckResult(
            item_id="C-08", title="施工図位置と構造図の整合",
            status="SKIP",
            detail="施工図位置情報と構造図の照合が必要（構造図データなし）",
        ),
        CheckResult(
            item_id="C-09", title="勾配確保（排水管ルート・流れ方向）",
            status="SKIP",
            detail="排水管ルート・流れ方向・勾配の確認にはスリーブ図以外の情報が必要",
        ),
        CheckResult(
            item_id="C-10", title="下階スラブtoスラブ壁・複合耐火壁との干渉",
            status="SKIP",
            detail="下階の壁情報がないため検証不可",
        ),
        CheckResult(
            item_id="C-11", title="段差スラブ近辺のスリーブ施工可否",
            status="SKIP",
            detail="構造図がないため検証不可",
        ),
        CheckResult(
            item_id="C-12", title="型枠段差・凹みからの寄り寸法",
            status="SKIP",
            detail="型枠情報がないため検証不可",
        ),
        CheckResult(
            item_id="C-13", title="柱面・仕上げ面からの寄り寸法",
            status="SKIP",
            detail="仕上げ図データがないため検証不可",
        ),
        CheckResult(
            item_id="C-14", title="建築・EPAスリーブの寄り寸法表記統一",
            status="SKIP",
            detail="建築スリーブとEPAスリーブの区別に追加データが必要",
        ),
    ]
