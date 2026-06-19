"""Сравнение двух прогонов (A — кандидат, B — бейзлайн) с оценкой значимости.

Идея (см. обсуждение):
- Оба прогона считаются на одних и тех же эталонных задачах -> сравнение ПАРНОЕ.
- Значимость различий считаем ПАРНЫМ БУТСТРЭПОМ по разностям d_i = a_i - b_i
  на общей популяции задач (для метрик с фиксированным знаменателем).
- Для метрик с плавающим знаменателем (precision, conditioning «по ответу») пары
  нет -> непарный бутстрэп, помечаем kind="non-paired".
- Множественные сравнения -> поправка Benjamini-Hochberg (FDR), q-value.
- Эффект: абсолютная и относительная (к бейзлайну B) дельта; сортировка по величине.

Весь вывод — простые dataclass/словарь, чтобы дашборд только рисовал.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from funnel_metrics import STATES, Report

N_BOOT = 2000
SEED = 12345
SIG_Q = 0.05
REL_MIN_BASE = 0.02   # если |B| меньше — относительную дельту не считаем (деление на ~0)
MIN_RELIABLE_N = 5

# Полярность: +1 если больше = лучше (все наши метрики такие; поле оставлено на будущее).
HIGHER = 1


# ---------------------------------------------------------------------------
# Реестр метрик
# ---------------------------------------------------------------------------


@dataclass
class MetricSpec:
    id: str
    name: str
    section: str           # overview | tier1 | tier2 | tier3 (адрес для перехода)
    kind: str              # paired-binary | paired-continuous | non-paired
    polarity: int          # HIGHER
    vector: Callable[[list[dict]], dict[str, float]]  # rows -> {task_id: value}


def _by(field: str, *, where=None, cast=float):
    def f(rows: list[dict]) -> dict[str, float]:
        out = {}
        for r in rows:
            if where is not None and not where(r):
                continue
            v = r.get(field)
            if v is None:
                continue
            out[r["id"]] = cast(v)
        return out
    return f


def _recall_class(c: str):
    return lambda rows: {r["id"]: float(r["pred_state"] == c)
                         for r in rows if r["gold_state"] == c}


def _precision_class(c: str):
    # знаменатель = pred==c -> плавает между прогонами -> non-paired
    return lambda rows: {r["id"]: float(r["gold_state"] == c)
                         for r in rows if r["pred_state"] == c}


def _handoff_state(c: str):
    return lambda rows: {r["id"]: float(r["context_correct"])
                         for r in rows if r["gold_state"] == c}


READY = "ready_for_sql"

REGISTRY: list[MetricSpec] = [
    # --- Обзор ---
    MetricSpec("handoff", "Полностью корректный handoff", "overview", "paired-binary", HIGHER,
               _by("context_correct")),
    MetricSpec("state_acc", "Точность state", "overview", "paired-binary", HIGHER,
               _by("state_correct")),
    *[MetricSpec(f"handoff_{c}", f"Handoff · {c}", "overview", "paired-binary", HIGHER,
                 _handoff_state(c)) for c in STATES],
    # --- Уровень 1 ---
    MetricSpec("api", "Сервис ответил", "tier1", "paired-binary", HIGHER, _by("api_ok")),
    MetricSpec("schema", "Контракт валиден", "tier1", "paired-binary", HIGHER, _by("schema_ok")),
    MetricSpec("vocab", "Без выдуманных id", "tier1", "paired-binary", HIGHER, _by("vocab_ok")),
    # --- Уровень 2 ---
    *[MetricSpec(f"recall_{c}", f"recall · {c}", "tier2", "paired-binary", HIGHER,
                 _recall_class(c)) for c in STATES],
    *[MetricSpec(f"precision_{c}", f"precision · {c}", "tier2", "non-paired", HIGHER,
                 _precision_class(c)) for c in STATES],
    MetricSpec("reason_primary", "reason_code: главный", "tier2", "paired-binary", HIGHER,
               _by("reason_primary_correct")),
    MetricSpec("reason_f1", "reason_codes: F1", "tier2", "paired-continuous", HIGHER,
               _by("reason_f1")),
    # --- Уровень 3 (популяция: эталон ready_for_sql) ---
    MetricSpec("domain", "domain · exact", "tier3", "paired-binary", HIGHER,
               _by("domain_exact", where=lambda r: r["gold_state"] == READY)),
    MetricSpec("metric_f1", "metric_ids · F1", "tier3", "paired-continuous", HIGHER,
               _by("metric_f1", where=lambda r: r["gold_state"] == READY)),
    MetricSpec("source_f1", "sources · F1", "tier3", "paired-continuous", HIGHER,
               _by("source_f1", where=lambda r: r["gold_state"] == READY)),
    MetricSpec("entity_f1", "entity_ids · F1", "tier3", "paired-continuous", HIGHER,
               _by("entity_f1", where=lambda r: r["gold_state"] == READY)),
    MetricSpec("filter_recall", "required_filters · recall", "tier3", "paired-continuous", HIGHER,
               _by("filter_recall", where=lambda r: r["gold_state"] == READY)),
    MetricSpec("mandatory_filter", "Обязат. фильтры · recall", "tier3", "paired-continuous", HIGHER,
               _by("mandatory_filter_recall")),
    MetricSpec("empty_ok", "Пустота при gap/oos · ok", "tier3", "paired-binary", HIGHER,
               _by("empty_invariant_ok")),
]


# ---------------------------------------------------------------------------
# Бутстрэп и поправка на множественность
# ---------------------------------------------------------------------------


def _two_sided_p(boot: np.ndarray) -> float:
    if boot.size == 0:
        return 1.0
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(min(1.0, max(0.0, p)))


def paired_bootstrap(a: dict[str, float], b: dict[str, float], rng: np.random.Generator):
    common = sorted(set(a) & set(b))
    if not common:
        return None
    da = np.array([a[i] for i in common], dtype=float)
    db = np.array([b[i] for i in common], dtype=float)
    d = da - db
    n = len(d)
    obs = float(d.mean())
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi),
            "p": _two_sided_p(boot), "n": n}


def unpaired_bootstrap(a: dict[str, float], b: dict[str, float], rng: np.random.Generator):
    va = np.array(list(a.values()), dtype=float)
    vb = np.array(list(b.values()), dtype=float)
    if va.size == 0 or vb.size == 0:
        return None
    obs = float(va.mean() - vb.mean())
    ba = va[rng.integers(0, va.size, size=(N_BOOT, va.size))].mean(axis=1)
    bb = vb[rng.integers(0, vb.size, size=(N_BOOT, vb.size))].mean(axis=1)
    boot = ba - bb
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi),
            "p": _two_sided_p(boot), "n": min(va.size, vb.size)}


def bh_fdr(pvals: dict[str, float]) -> dict[str, float]:
    items = [(k, v) for k, v in pvals.items() if v is not None]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda x: x[1])
    q: dict[str, float] = {}
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        k, p = items[rank]
        prev = min(prev, p * m / (rank + 1))
        q[k] = prev
    return q


# ---------------------------------------------------------------------------
# Расчёт различий
# ---------------------------------------------------------------------------


@dataclass
class MetricDiff:
    id: str
    name: str
    section: str
    kind: str
    polarity: int
    a: float
    b: float
    n: int
    abs_diff: float
    rel_diff: float | None
    from_zero: bool
    ci_low: float
    ci_high: float
    p: float
    q: float
    significant: bool
    discordant: tuple[int, int] | None  # (A прав & B нет, B прав & A нет) для бинарных
    low_power: bool

    @property
    def uplift(self) -> float:
        """>0 => в пользу A (зелёный), <0 => в пользу B (красный)."""
        return self.polarity * self.abs_diff

    @property
    def sort_magnitude(self) -> float:
        return self.abs_diff if self.from_zero else abs(self.rel_diff or 0.0)


def _discordant(a: dict[str, float], b: dict[str, float]) -> tuple[int, int]:
    common = set(a) & set(b)
    bb = sum(1 for i in common if a[i] >= 0.5 and b[i] < 0.5)
    cc = sum(1 for i in common if a[i] < 0.5 and b[i] >= 0.5)
    return bb, cc


def compute_diffs(report_a: Report, report_b: Report) -> list[MetricDiff]:
    rng = np.random.default_rng(SEED)
    rows_a, rows_b = report_a.rows, report_b.rows

    raw: list[dict] = []
    pvals: dict[str, float] = {}
    for spec in REGISTRY:
        va = spec.vector(rows_a)
        vb = spec.vector(rows_b)
        if not va or not vb:
            continue
        a_val = float(np.mean(list(va.values())))
        b_val = float(np.mean(list(vb.values())))
        boot = (unpaired_bootstrap(va, vb, rng) if spec.kind == "non-paired"
                else paired_bootstrap(va, vb, rng))
        if boot is None:
            continue
        abs_diff = a_val - b_val
        from_zero = abs(b_val) < REL_MIN_BASE and abs(abs_diff) > 1e-9
        rel = None if from_zero else (abs_diff / b_val if abs(b_val) > 1e-9 else None)
        discordant = _discordant(va, vb) if spec.kind == "paired-binary" else None
        low_power = boot["n"] < MIN_RELIABLE_N or (
            discordant is not None and sum(discordant) < 6
        )
        raw.append({
            "spec": spec, "a": a_val, "b": b_val, "abs": abs_diff, "rel": rel,
            "from_zero": from_zero, "boot": boot, "discordant": discordant,
            "low_power": low_power,
        })
        pvals[spec.id] = boot["p"]

    qmap = bh_fdr(pvals)
    diffs: list[MetricDiff] = []
    for item in raw:
        spec = item["spec"]
        q = qmap.get(spec.id, 1.0)
        diffs.append(MetricDiff(
            id=spec.id, name=spec.name, section=spec.section, kind=spec.kind,
            polarity=spec.polarity, a=item["a"], b=item["b"], n=item["boot"]["n"],
            abs_diff=item["abs"], rel_diff=item["rel"], from_zero=item["from_zero"],
            ci_low=item["boot"]["ci_low"], ci_high=item["boot"]["ci_high"],
            p=item["boot"]["p"], q=q, significant=(q < SIG_Q),
            discordant=item["discordant"], low_power=item["low_power"],
        ))
    return diffs


def rank_significant(diffs: list[MetricDiff]) -> list[MetricDiff]:
    sig = [d for d in diffs if d.significant]
    sig.sort(key=lambda d: (not d.from_zero, -d.sort_magnitude))
    return sig
