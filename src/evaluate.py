"""
Оценка Context Discovery: Context Handoff Score (CHS).

Запуск: python3 src/evaluate.py
Методика: README.md (раздел CHS)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "data" / "golden.jsonl"
DEFAULT_PRED = ROOT / "outputs" / "predictions.jsonl"

CRITICAL_STATES = {"needs_clarification", "semantic_gap", "out_of_scope"}

CHS_VERSION = "1.1"
CHS_THRESHOLD_STAGING = 0.95
CHS_THRESHOLD_MIN = 0.85

SCORE_PARTIAL = 0.7


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def norm_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def norm_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({norm_str(v) for v in values if norm_str(v)})


def reason_codes(row: dict) -> set[str]:
    value = row.get("reason_codes", [])
    if not isinstance(value, list):
        return set()
    return {c for c in value if isinstance(c, str)}


def load_matched(
    golden_path: Path,
    predictions_path: Path,
) -> tuple[list[dict], list[tuple[dict, dict]]]:
    golden_rows = load_jsonl(golden_path)
    pred_rows = load_jsonl(predictions_path)
    preds = {r["id"]: r for r in pred_rows}

    matched: list[tuple[dict, dict]] = []
    for gold in golden_rows:
        pred = preds.get(gold["id"])
        if pred is None:
            continue
        matched.append((gold, pred))

    return golden_rows, matched


def score_undefined_conservatism(gold_state: str, pred: dict) -> float:
    ps = norm_str(pred.get("state"))
    p_metrics = norm_list(pred.get("metric_ids"))
    p_sources = norm_list(pred.get("sources"))

    if ps == gold_state:
        return 1.0
    if ps == "ready_for_sql" and not p_metrics and not p_sources:
        return SCORE_PARTIAL
    if ps == "ready_for_sql":
        return 0.0
    return 0.6


def score_mcp_contract(pred: dict) -> float:
    ps = norm_str(pred.get("state"))
    pr = reason_codes(pred)
    domain = pred.get("domain_id")

    if not ps or not pr:
        return 0.0
    if not domain:
        return SCORE_PARTIAL
    return 1.0


def compute_context_handoff_score(matched: list[tuple[dict, dict]]) -> dict[str, float]:
    total = len(matched)
    if total == 0:
        return {}

    ready_gold = 0
    confirmed_path_hits = 0
    undefined_scores: list[float] = []
    contract_scores: list[float] = []

    for gold, pred in matched:
        gs = norm_str(gold.get("state"))
        ps = norm_str(pred.get("state"))

        if gs == "ready_for_sql":
            ready_gold += 1
            if ps == "ready_for_sql":
                confirmed_path_hits += 1

        if gs in CRITICAL_STATES:
            undefined_scores.append(score_undefined_conservatism(gs, pred))

        contract_scores.append(score_mcp_contract(pred))

    operational_reach = safe_div(total, total)
    confirmed_path_alignment = safe_div(confirmed_path_hits, ready_gold)
    undefined_task_conservatism = safe_div(sum(undefined_scores), len(undefined_scores))
    mcp_contract_compliance = safe_div(sum(contract_scores), len(contract_scores))

    chs = (
        operational_reach
        + confirmed_path_alignment
        + undefined_task_conservatism
        + mcp_contract_compliance
    ) / 4

    return {
        "operational_reach": operational_reach,
        "confirmed_path_alignment": confirmed_path_alignment,
        "undefined_task_conservatism": undefined_task_conservatism,
        "mcp_contract_compliance": mcp_contract_compliance,
        "context_handoff_score": chs,
        "ready_gold_cases": float(ready_gold),
        "undefined_gold_cases": float(len(undefined_scores)),
        "matched_cases": float(total),
    }


def chs_status(chs: float) -> str:
    if chs >= CHS_THRESHOLD_STAGING:
        return f"в норме для staging (≥ {CHS_THRESHOLD_STAGING:.2f})"
    if chs >= CHS_THRESHOLD_MIN:
        return f"допустимо при доработке ({CHS_THRESHOLD_MIN:.2f} – {CHS_THRESHOLD_STAGING:.2f})"
    return f"ниже порога выкладки (< {CHS_THRESHOLD_MIN:.2f})"


def weakest_chs_component(kpi: dict[str, float]) -> str:
    components = {
        "operational_reach": kpi["operational_reach"],
        "confirmed_path_alignment": kpi["confirmed_path_alignment"],
        "undefined_task_conservatism": kpi["undefined_task_conservatism"],
        "mcp_contract_compliance": kpi["mcp_contract_compliance"],
    }
    return min(components, key=components.get)


def print_context_handoff_score(kpi: dict[str, float], *, matched: int, golden_total: int) -> None:
    print()
    print("=" * 52)
    print(f"Context Handoff Score (CHS-{CHS_VERSION})")
    print("=" * 52)
    print(f"Сопоставлено задач: {matched}/{golden_total}")
    print()
    print("Компоненты (вес 0.25):")
    print(f"  operational_reach             {kpi['operational_reach']:.3f}")
    print(f"  confirmed_path_alignment        {kpi['confirmed_path_alignment']:.3f}")
    print(f"  undefined_task_conservatism   {kpi['undefined_task_conservatism']:.3f}")
    print(f"  mcp_contract_compliance       {kpi['mcp_contract_compliance']:.3f}")
    print()
    print(f"context_handoff_score = {kpi['context_handoff_score']:.3f}")
    print(f"Статус CHS: {chs_status(kpi['context_handoff_score'])}")
    weak = weakest_chs_component(kpi)
    print(f"Фокус улучшения: компонент `{weak}` (наименьший вклад)")
    print()
    print("Методика: README.md (раздел CHS)")


def run_evaluation(
    predictions_path: Path = DEFAULT_PRED,
    golden_path: Path = DEFAULT_GOLDEN,
) -> dict[str, float] | None:
    if not golden_path.exists():
        print(f"Не найден эталон: {golden_path}", file=sys.stderr)
        return None
    if not predictions_path.exists():
        print(f"Не найдены предикты: {predictions_path}", file=sys.stderr)
        print("Сначала запустите: python3 src/pipeline.py", file=sys.stderr)
        return None

    golden_rows, matched = load_matched(golden_path, predictions_path)
    if not matched:
        print("Нет пересечения id между predictions и golden.")
        return None

    kpi = compute_context_handoff_score(matched)
    print_context_handoff_score(kpi, matched=len(matched), golden_total=len(golden_rows))
    return kpi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка Context Discovery: Context Handoff Score (CHS).",
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args.predictions, args.golden)


if __name__ == "__main__":
    main()
