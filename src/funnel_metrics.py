"""Funnel-style evaluation for Context Discovery.

Separate from `evaluate.py` (the official CHS). This module computes a richer,
diagnostic view of agent performance as a 5-stage funnel plus per-tier subscores.

Design (see chat discussion):

  Tier 1 - technical reach (TRUE survival, multiplicative gates):
    stage 1  api_ok          - the pipeline produced a response for the task
    stage 2  json_ok         - the response is a JSON object
    stage 3  schema_ok       - required keys present, correct types, state in enum
  Tier 2 - the decision hinge:
    stage 4  state_correct   - predicted state == golden state
  Tier 3 - context correctness (parallel dims collapsed to one strict gate):
    stage 5  context_correct - all configured context fields match golden
                               (empty-is-correct, so it spans gap/out_of_scope too)

Each funnel bar is a binary AND-gate on top of the previous one, with the
denominator fixed at ALL golden tasks. That keeps the curve monotone and
readable as "fraction of tasks still fully correct after stage k".

The funnel is intentionally coarse. The detail lives in the per-tier subscores:
  - Tier 1: contract validity (no hallucinated ids) + failure breakdown
  - Tier 2: 4-class state confusion matrix, per-class precision/recall, macro-F1,
            reason-code accuracy
  - Tier 3: per-field exact-match + precision/recall/F1 vector, with a 3-way
            conditioning mode (on golden / on predictions / intersection),
            mandatory-filter recall, and the empty-field invariant check.

Everything is plain dict/list output so the dashboard (or a future pruned
version) can pick whatever it needs without importing heavy deps.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "data" / "golden.jsonl"
DEFAULT_SL = ROOT / "data" / "semantic_layer.json"

# ---------------------------------------------------------------------------
# Configuration (kept here so the funnel is easy to prune / re-tune later).
# ---------------------------------------------------------------------------

STATES: tuple[str, ...] = (
    "ready_for_sql",
    "needs_clarification",
    "semantic_gap",
    "out_of_scope",
)
READY = "ready_for_sql"
# States where the golden contract requires fully empty context.
EMPTY_CONTEXT_STATES: tuple[str, ...] = ("semantic_gap", "out_of_scope")

ALLOWED_REASON_CODES: tuple[str, ...] = (
    "good_context_match",
    "partial_context",
    "wrong_domain",
    "wrong_metric",
    "wrong_source",
    "missing_required_filter",
    "ambiguous_request",
    "semantic_gap",
    "out_of_scope",
)

# Context fields evaluated as a vector. The strict tier-3 funnel gate is the AND
# of exact-match over CONTEXT_GATE_FIELDS only (filters left out by default
# because golden filters mix machine-mandatory and free-text question params).
LIST_FIELDS: tuple[str, ...] = ("metric_ids", "sources", "entity_ids", "required_filters")
SINGLE_FIELDS: tuple[str, ...] = ("domain_id",)
CONTEXT_FIELDS: tuple[str, ...] = SINGLE_FIELDS + LIST_FIELDS
CONTEXT_GATE_FIELDS: tuple[str, ...] = ("domain_id", "metric_ids", "sources", "entity_ids")

CONDITION_MODES: tuple[str, ...] = ("golden", "predicted", "intersection")

MIN_RELIABLE_N = 5  # below this, cells are flagged as low-confidence
Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def norm_str(value: Any) -> str | None:
    """Lowercase/strip a scalar. None stays None (null != empty string)."""
    if value is None:
        return None
    return str(value).strip().lower()


def norm_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    out = set()
    for v in value:
        s = norm_str(v)
        if s:
            out.add(s)
    return out


_FILTER_FIELD_RE = re.compile(r"^\s*([a-zA-Z_][\w.]*)")


def filter_field(predicate: str) -> str | None:
    """Extract the field name from a free-text filter predicate.

    'sku_id = ''Total''' -> 'sku_id'; 'check_year between ...' -> 'check_year'.
    """
    if not isinstance(predicate, str):
        return None
    m = _FILTER_FIELD_RE.match(predicate.strip())
    return m.group(1).lower() if m else None


def filter_field_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {f for f in (filter_field(p) for p in value) if f}


# ---------------------------------------------------------------------------
# Set-comparison primitives (empty-is-correct convention)
# ---------------------------------------------------------------------------


@dataclass
class SetScore:
    precision: float
    recall: float
    f1: float
    exact: float


def score_sets(gold: set[str], pred: set[str]) -> SetScore:
    if not gold and not pred:
        return SetScore(1.0, 1.0, 1.0, 1.0)
    inter = len(gold & pred)
    precision = 1.0 if not pred else inter / len(pred)
    recall = 1.0 if not gold else inter / len(gold)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    exact = 1.0 if gold == pred else 0.0
    return SetScore(precision, recall, f1, exact)


def wilson_interval(k: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Semantic-layer derived vocab
# ---------------------------------------------------------------------------


@dataclass
class Vocab:
    domains: set[str]
    metrics: set[str]
    sources: set[str]
    entities: set[str]
    metric_required_fields: dict[str, set[str]]  # metric_id -> set of mandatory filter fields

    @classmethod
    def from_semantic_layer(cls, sl: dict) -> "Vocab":
        domains = {norm_str(d["id"]) for d in sl.get("domains", [])}
        metrics = {norm_str(m["id"]) for m in sl.get("metrics", [])}
        sources = {norm_str(s["id"]) for s in sl.get("sources", [])}
        entities = {norm_str(e["id"]) for e in sl.get("entities", [])}
        req: dict[str, set[str]] = {}
        for m in sl.get("metrics", []):
            mid = norm_str(m["id"])
            fields = {norm_str(rf.get("field")) for rf in m.get("required_filters", [])}
            req[mid] = {f for f in fields if f}
        return cls(domains, metrics, sources, entities, req)


# ---------------------------------------------------------------------------
# Per-task scoring
# ---------------------------------------------------------------------------


@dataclass
class TaskScore:
    id: str
    gold_state: str | None
    pred_state: str | None
    present: bool          # api_ok: prediction exists
    api_ok: bool
    json_ok: bool
    schema_ok: bool
    state_correct: bool
    context_correct: bool  # strict tier-3 gate
    vocab_ok: bool         # all ids in semantic layer
    vocab_detail: dict[str, bool]
    field_scores: dict[str, SetScore]      # per context field
    domain_exact: float
    reason_primary_correct: float | None   # None if golden has no reason codes
    reason_f1: float
    mandatory_filter_recall: float | None  # None if not applicable
    empty_invariant_ok: bool | None        # None if not an empty-context state
    error_type: str | None

    def funnel_flags(self) -> list[bool]:
        """Cumulative AND gates for the 5 funnel stages."""
        s1 = self.api_ok
        s2 = s1 and self.json_ok
        s3 = s2 and self.schema_ok
        s4 = s3 and self.state_correct
        s5 = s4 and self.context_correct
        return [s1, s2, s3, s4, s5]


def _state_in_enum(state: Any) -> bool:
    return norm_str(state) in STATES


def score_task(
    gold: dict,
    pred: dict | None,
    vocab: Vocab,
    *,
    error_type: str | None = None,
    context_gate_fields: tuple[str, ...] = CONTEXT_GATE_FIELDS,
) -> TaskScore:
    gid = gold["id"]
    gold_state = norm_str(gold.get("state"))

    present = pred is not None
    # A missing row means the pipeline dropped the task. Meta error_type tells us
    # whether the API answered (JSON parse error) or never answered (timeout/etc).
    json_err = bool(error_type and "json" in error_type.lower())
    if present:
        api_ok = True
        json_ok = isinstance(pred, dict)
        schema_ok = json_ok and _schema_ok(pred)
    else:
        api_ok = json_err  # JSON error => API responded but body was unparseable
        json_ok = False
        schema_ok = False
        pred = {}

    pred_state = norm_str(pred.get("state")) if present else None

    # --- vocab / contract validity (no hallucinated ids) ---
    vocab_detail = _vocab_detail(pred, vocab) if present else {
        "domain_id": False, "metric_ids": False, "sources": False,
        "entity_ids": False, "reason_codes": False,
    }
    vocab_ok = all(vocab_detail.values())

    # --- state ---
    state_correct = bool(schema_ok and pred_state == gold_state)

    # --- per-field context scores ---
    field_scores: dict[str, SetScore] = {}
    for f in LIST_FIELDS:
        field_scores[f] = score_sets(norm_set(gold.get(f)), norm_set(pred.get(f)))
    g_dom, p_dom = norm_str(gold.get("domain_id")), norm_str(pred.get("domain_id"))
    domain_exact = 1.0 if g_dom == p_dom else 0.0

    # --- strict tier-3 context gate (empty-is-correct) ---
    context_correct = schema_ok
    for f in context_gate_fields:
        if f in SINGLE_FIELDS:
            context_correct = context_correct and (domain_exact == 1.0)
        else:
            context_correct = context_correct and (field_scores[f].exact == 1.0)

    # --- reason codes ---
    reason_primary_correct, reason_f1 = _reason_scores(gold, pred if present else {})

    # --- mandatory filter recall (uses golden metrics) ---
    mandatory_filter_recall = _mandatory_filter_recall(gold, pred if present else {}, vocab)

    # --- empty-field invariant for gap / out_of_scope ---
    empty_invariant_ok: bool | None = None
    if gold_state in EMPTY_CONTEXT_STATES and present:
        empty_invariant_ok = not (
            norm_set(pred.get("metric_ids"))
            or norm_set(pred.get("sources"))
            or norm_set(pred.get("entity_ids"))
        )

    return TaskScore(
        id=gid,
        gold_state=gold_state,
        pred_state=pred_state,
        present=present,
        api_ok=api_ok,
        json_ok=json_ok,
        schema_ok=schema_ok,
        state_correct=state_correct,
        context_correct=bool(context_correct),
        vocab_ok=vocab_ok,
        vocab_detail=vocab_detail,
        field_scores=field_scores,
        domain_exact=domain_exact,
        reason_primary_correct=reason_primary_correct,
        reason_f1=reason_f1,
        mandatory_filter_recall=mandatory_filter_recall,
        empty_invariant_ok=empty_invariant_ok,
        error_type=error_type,
    )


def _schema_ok(pred: dict) -> bool:
    if not isinstance(pred, dict):
        return False
    if not _state_in_enum(pred.get("state")):
        return False
    for f in ("metric_ids", "sources", "reason_codes"):
        if not isinstance(pred.get(f), list):
            return False
    dom = pred.get("domain_id")
    if dom is not None and not isinstance(dom, str):
        return False
    for opt in ("entity_ids", "required_filters"):
        if opt in pred and not isinstance(pred.get(opt), list):
            return False
    return True


def _vocab_detail(pred: dict, vocab: Vocab) -> dict[str, bool]:
    dom = norm_str(pred.get("domain_id"))
    metrics = norm_set(pred.get("metric_ids"))
    sources = norm_set(pred.get("sources"))
    entities = norm_set(pred.get("entity_ids"))
    codes = norm_set(pred.get("reason_codes"))
    return {
        "domain_id": dom is None or dom in vocab.domains,
        "metric_ids": metrics <= vocab.metrics,
        "sources": sources <= vocab.sources,
        "entity_ids": entities <= vocab.entities,
        "reason_codes": codes <= {c.lower() for c in ALLOWED_REASON_CODES},
    }


def _reason_scores(gold: dict, pred: dict) -> tuple[float | None, float]:
    g_codes = gold.get("reason_codes") or []
    p_codes = pred.get("reason_codes") or []
    primary: float | None = None
    if isinstance(g_codes, list) and g_codes:
        g0 = norm_str(g_codes[0])
        p0 = norm_str(p_codes[0]) if isinstance(p_codes, list) and p_codes else None
        primary = 1.0 if g0 == p0 else 0.0
    f1 = score_sets(norm_set(g_codes), norm_set(p_codes)).f1
    return primary, f1


def _mandatory_filter_recall(gold: dict, pred: dict, vocab: Vocab) -> float | None:
    gold_metrics = norm_set(gold.get("metric_ids"))
    required: set[str] = set()
    for m in gold_metrics:
        required |= vocab.metric_required_fields.get(m, set())
    if not required:
        return None
    pred_fields = filter_field_set(pred.get("required_filters"))
    return len(required & pred_fields) / len(required)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

FUNNEL_STAGES = [
    ("api_ok", "API responded", 1),
    ("json_ok", "Valid JSON", 1),
    ("schema_ok", "Schema / contract valid", 1),
    ("state_correct", "State correct", 2),
    ("context_correct", "Context correct", 3),
]


def build_funnel(tasks: list[TaskScore]) -> list[dict]:
    n = len(tasks)
    flags = [t.funnel_flags() for t in tasks]
    out = []
    prev = n
    for i, (key, label, tier) in enumerate(FUNNEL_STAGES):
        passed = sum(1 for f in flags if f[i])
        rate = passed / n if n else 0.0
        lo, hi = wilson_interval(passed, n)
        out.append({
            "key": key,
            "label": label,
            "tier": tier,
            "passed": passed,
            "n": n,
            "rate": rate,
            "ci_low": lo,
            "ci_high": hi,
            "drop_from_prev": (prev - passed) / n if n else 0.0,
            "dropped_count": prev - passed,
        })
        prev = passed
    return out


def build_handoff_by_state(tasks: list[TaskScore]) -> dict[str, dict]:
    """Segment the funnel endpoint (fully-correct handoff) by golden state.

    For each golden state: how many of those tasks pass every funnel stage, plus
    the cumulative pass rate at each stage (a survival curve per segment).
    """
    out: dict[str, dict] = {}
    for s in STATES:
        sub = [t for t in tasks if t.gold_state == s]
        n = len(sub)
        flags = [t.funnel_flags() for t in sub]
        stage_rates = []
        for i in range(len(FUNNEL_STAGES)):
            passed = sum(1 for f in flags if f[i])
            stage_rates.append(passed / n if n else 0.0)
        endpoint_passed = sum(1 for f in flags if f[-1])
        lo, hi = wilson_interval(endpoint_passed, n)
        out[s] = {
            "n": n,
            "stage_rates": stage_rates,
            "endpoint_passed": endpoint_passed,
            "endpoint_rate": endpoint_passed / n if n else 0.0,
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


def build_state_metrics(tasks: list[TaskScore]) -> dict:
    cols = list(STATES) + ["<invalid/missing>"]
    matrix = {g: {c: 0 for c in cols} for g in STATES}
    for t in tasks:
        if t.gold_state not in STATES:
            continue
        pred = t.pred_state if (t.schema_ok and t.pred_state in STATES) else "<invalid/missing>"
        matrix[t.gold_state][pred] += 1

    per_class = {}
    for c in STATES:
        gold_c = sum(1 for t in tasks if t.gold_state == c)
        pred_c = sum(1 for t in tasks if t.schema_ok and t.pred_state == c)
        tp = matrix.get(c, {}).get(c, 0)
        recall = tp / gold_c if gold_c else None
        precision = tp / pred_c if pred_c else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = 0.0 if (gold_c or pred_c) else None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gold_n": gold_c,
            "pred_n": pred_c,
            "tp": tp,
        }
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    macro_f1 = mean(f1s)
    accuracy = mean(1.0 if t.state_correct else 0.0 for t in tasks)

    reason_primary = [t.reason_primary_correct for t in tasks if t.reason_primary_correct is not None]
    return {
        "confusion": matrix,
        "columns": cols,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "reason_primary_accuracy": mean(reason_primary) if reason_primary else None,
        "reason_primary_n": len(reason_primary),
        "reason_f1": mean(t.reason_f1 for t in tasks),
    }


def _population(tasks: list[TaskScore], mode: str) -> list[TaskScore]:
    if mode == "golden":
        return [t for t in tasks if t.gold_state == READY]
    if mode == "predicted":
        return [t for t in tasks if t.pred_state == READY]
    if mode == "intersection":
        return [t for t in tasks if t.gold_state == READY and t.pred_state == READY]
    raise ValueError(f"unknown condition mode: {mode}")


def build_context_metrics(tasks: list[TaskScore], mode: str = "golden") -> dict:
    pop = _population(tasks, mode)
    n = len(pop)
    fields = []
    for f in CONTEXT_FIELDS:
        if f in SINGLE_FIELDS:
            exact = mean(t.domain_exact for t in pop)
            row = {"field": f, "type": "scalar", "exact": exact,
                   "precision": None, "recall": None, "f1": None, "n": n}
        else:
            exact = mean(t.field_scores[f].exact for t in pop)
            row = {
                "field": f,
                "type": "set",
                "exact": exact,
                "precision": mean(t.field_scores[f].precision for t in pop),
                "recall": mean(t.field_scores[f].recall for t in pop),
                "f1": mean(t.field_scores[f].f1 for t in pop),
                "n": n,
            }
        fields.append(row)

    mand = [t.mandatory_filter_recall for t in pop if t.mandatory_filter_recall is not None]
    inv = [t for t in tasks if t.empty_invariant_ok is not None]
    inv_violations = [t.id for t in inv if t.empty_invariant_ok is False]
    return {
        "mode": mode,
        "n": n,
        "fields": fields,
        "mandatory_filter_recall": mean(mand) if mand else None,
        "mandatory_filter_n": len(mand),
        "strict_context_rate": mean(1.0 if t.context_correct else 0.0 for t in pop),
        "invariant_checked": len(inv),
        "invariant_violations": inv_violations,
    }


def build_tier1_detail(tasks: list[TaskScore]) -> dict:
    n = len(tasks)
    vocab_keys = ["domain_id", "metric_ids", "sources", "entity_ids", "reason_codes"]
    return {
        "api_rate": mean(1.0 if t.api_ok else 0.0 for t in tasks),
        "json_rate": mean(1.0 if t.json_ok else 0.0 for t in tasks),
        "schema_rate": mean(1.0 if t.schema_ok else 0.0 for t in tasks),
        "vocab_rate": mean(1.0 if t.vocab_ok else 0.0 for t in tasks),
        "vocab_breakdown": {
            k: mean(1.0 if t.vocab_detail.get(k) else 0.0 for t in tasks) for k in vocab_keys
        },
        "failures": [
            {"id": t.id, "error_type": t.error_type, "present": t.present,
             "schema_ok": t.schema_ok, "vocab_ok": t.vocab_ok}
            for t in tasks
            if not (t.api_ok and t.json_ok and t.schema_ok and t.vocab_ok)
        ],
        "n": n,
    }


def task_rows(tasks: list[TaskScore]) -> list[dict]:
    """Flat per-task table for the drill-down view."""
    rows = []
    for t in tasks:
        flags = t.funnel_flags()
        rows.append({
            "id": t.id,
            "gold_state": t.gold_state,
            "pred_state": t.pred_state,
            "api_ok": flags[0],
            "json_ok": flags[1],
            "schema_ok": flags[2],
            "state_correct": flags[3],
            "context_correct": flags[4],
            "vocab_ok": t.vocab_ok,
            "domain_exact": t.domain_exact,
            "metric_f1": round(t.field_scores["metric_ids"].f1, 3),
            "source_f1": round(t.field_scores["sources"].f1, 3),
            "entity_f1": round(t.field_scores["entity_ids"].f1, 3),
            "filter_recall": round(t.field_scores["required_filters"].recall, 3),
            "mandatory_filter_recall": (
                None if t.mandatory_filter_recall is None
                else round(t.mandatory_filter_recall, 3)
            ),
            "reason_primary_correct": t.reason_primary_correct,
            "reason_f1": round(t.reason_f1, 3),
            "empty_invariant_ok": t.empty_invariant_ok,
        })
    return rows


@dataclass
class Report:
    n: int
    funnel: list[dict]
    tier1: dict
    state: dict
    context_by_mode: dict[str, dict]
    rows: list[dict]
    handoff_by_state: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    raw_gold: dict[str, dict] = field(default_factory=dict)
    raw_pred: dict[str, dict] = field(default_factory=dict)


def evaluate(
    golden: list[dict],
    predictions: list[dict],
    semantic_layer: dict,
    *,
    meta: dict | None = None,
    context_gate_fields: tuple[str, ...] = CONTEXT_GATE_FIELDS,
) -> Report:
    vocab = Vocab.from_semantic_layer(semantic_layer)
    pred_by_id = {p.get("id"): p for p in predictions if isinstance(p, dict)}
    error_by_id = {}
    for e in (meta or {}).get("errors", []) or []:
        error_by_id[e.get("id")] = e.get("error_type")

    tasks: list[TaskScore] = []
    for g in golden:
        gid = g["id"]
        tasks.append(
            score_task(
                g,
                pred_by_id.get(gid),
                vocab,
                error_type=error_by_id.get(gid),
                context_gate_fields=context_gate_fields,
            )
        )

    context_by_mode = {m: build_context_metrics(tasks, m) for m in CONDITION_MODES}
    return Report(
        n=len(tasks),
        funnel=build_funnel(tasks),
        tier1=build_tier1_detail(tasks),
        state=build_state_metrics(tasks),
        context_by_mode=context_by_mode,
        rows=task_rows(tasks),
        handoff_by_state=build_handoff_by_state(tasks),
        meta=meta or {},
        raw_gold={g["id"]: g for g in golden},
        raw_pred={pid: p for pid, p in pred_by_id.items() if pid is not None},
    )


def evaluate_paths(
    predictions_path: str | Path,
    golden_path: str | Path = DEFAULT_GOLDEN,
    semantic_layer_path: str | Path = DEFAULT_SL,
    meta_path: str | Path | None = None,
) -> Report:
    golden = load_jsonl(golden_path)
    predictions = load_jsonl(predictions_path)
    sl = load_json(semantic_layer_path)
    meta = None
    if meta_path is None:
        guess = Path(predictions_path).with_suffix(".meta.json")
        if guess.exists():
            meta_path = guess
    if meta_path and Path(meta_path).exists():
        meta = load_json(meta_path)
    return evaluate(golden, predictions, sl, meta=meta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Funnel metrics (diagnostic, JSON dump).")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--semantic-layer", type=Path, default=DEFAULT_SL)
    args = parser.parse_args()
    report = evaluate_paths(args.predictions, args.golden, args.semantic_layer)
    print(json.dumps({
        "n": report.n,
        "funnel": report.funnel,
        "tier1": {k: v for k, v in report.tier1.items() if k != "failures"},
        "state_macro_f1": report.state["macro_f1"],
        "state_accuracy": report.state["accuracy"],
        "context_golden": report.context_by_mode["golden"],
    }, ensure_ascii=False, indent=2))
