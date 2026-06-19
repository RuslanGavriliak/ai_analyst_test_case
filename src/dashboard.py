"""Streamlit-дашборд воронки качества Context Discovery.

Запуск:
    streamlit run src/dashboard.py

Читает предсказания из outputs/*.jsonl и считает метрики в funnel_metrics.py.
Дашборд диагностический: он дополняет официальный Context Handoff Score (CHS)
из evaluate.py, не заменяя его. Вся логика подсчёта — в funnel_metrics.py;
здесь только отрисовка. Термины совпадают с README.md и AGENT_FLOW_GUIDE.md.

Логика воронки (для автора метрик):
  - Стадии 1-3 (api/json/контракт) — настоящая "воронка выживаемости":
    падение обнуляет всё дальше по цепочке.
  - Стадия 4 (state) — точка решения (статус готовности).
  - Стадия 5 (контекст) — строгий AND по полям пакета контекста, причём для
    semantic_gap / out_of_scope пустые поля считаются ВЕРНЫМ ответом, поэтому
    стадия определена на всех задачах. Это конечная точка воронки.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import compare
import funnel_metrics as fm

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
TASKS_PATH = ROOT / "data" / "tasks.jsonl"

TIER_COLORS = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
TIER_NAMES = {1: "Технические метрики", 2: "State метрики", 3: "Semantic метрики"}

# Русские подписи стадий воронки (ключи приходят из funnel_metrics.FUNNEL_STAGES).
STAGE_LABELS = {
    "api_ok": "Ответ получен",
    "json_ok": "Валидный JSON",
    "schema_ok": "Контракт валиден",
    "state_correct": "State верен",
    "context_correct": "Контекст верен",
}


def stage_label(stage: dict) -> str:
    return STAGE_LABELS.get(stage["key"], stage["label"])

# Только два режима: третий (intersection) убран по просьбе.
CONDITION_MODES = ("golden", "predicted")
MODE_LABEL = {"golden": "По эталону", "predicted": "По ответу агента"}
MODE_HELP = {
    "golden": "Знаменатель = задачи, где ЭТАЛОННЫЙ state = ready_for_sql. "
              "Стабильно и сравнимо между прогонами (по умолчанию).",
    "predicted": "Знаменатель = задачи, где АГЕНТ поставил ready_for_sql. "
                 "Это precision-взгляд: знаменатель плавает между прогонами, метрику "
                 "легко «накрутить» осторожностью.",
}

H = {
    "prime": "Доля задач, где пакет контекста верен на ВСЕХ 5 стадиях воронки "
             "(строгий AND). Для semantic_gap / out_of_scope пустые поля = верно. "
             "Это конечная точка воронки и главный показатель качества handoff.",
    "n": "Сколько эталонных задач сопоставлено и оценено.",
    "state_acc": "Доля задач с точным совпадением state с эталоном (exact match по 4 классам).",
    "sql_ctx": "Только задачи, где ЭТАЛОННЫЙ state = ready_for_sql: доля, где поля "
               "контекста (домен, метрики, источники, сущности) точно совпали с эталоном. "
               "Отличие от воронки: здесь нет gap / out_of_scope, оценивается именно "
               "контекст под черновик SQL.",
    "violations": "Задачи с эталоном semantic_gap / out_of_scope, где агент всё же вернул "
                  "непустые метрики / источники / сущности (выдуманный контекст).",
    "api": "operational_reach: сервис вернул ответ. Падение здесь обнуляет все стадии дальше.",
    "json": "Ответ — корректный JSON-объект.",
    "schema": "Контракт валиден: есть обязательные ключи нужных типов и state входит "
              "в список допустимых (ready_for_sql / needs_clarification / semantic_gap / "
              "out_of_scope).",
    "vocab": "Все id (домен, метрики, источники, сущности, reason_codes) есть в semantic "
             "layer — агент не выдумал новых имён. Часть mcp_contract_compliance.",
    "reason_primary": "Первый reason_code совпал с эталоном. По AGENT_FLOW_GUIDE первый код — "
                      "главная причина state. Считается только там, где у эталона есть коды.",
    "reason_f1": "F1 по МНОЖЕСТВУ reason_codes (без учёта порядка): "
                 "пересечение предсказанных и эталонных кодов, гармоническое среднее "
                 "точности и полноты.",
    "mand_filter": "Доля обязательных фильтров метрики (из semantic layer, напр. свёртки "
                   "'Total' в кубе чеков), чьё поле присутствует в ответе. Пропуск "
                   "критичного фильтра = неверный grain даже при верном state.",
    "strict_ctx_pop": "Доля задач выбранной популяции, где все поля контракта-контекста "
                      "точно совпали с эталоном (пустое = верно).",
    "state_acc_tier": "Доля точных совпадений state с эталоном.",
    "field_vector": "Для каждого поля: exact (точное совпадение множеств), "
                    "precision (точность — нет лишних/выдуманных), recall (полнота — "
                    "не потеряли нужное), F1. Для домена — только exact (это одно значение).",
    "confusion": "Строки — эталонный state, столбцы — ответ агента. По диагонали — верные "
                 "ответы, вне диагонали — ошибки.",
    "per_class": "precision / recall / F1 по каждому классу state. Полнота (recall) "
                 "условлена на эталоне, точность (precision) — на ответах агента.",
}

# Определение каждой метрики из реестра сравнения (то же, что под «?» в одиночном режиме).
# Ключи = compare.MetricSpec.id.
METRIC_DEF = {
    "handoff": H["prime"],
    "state_acc": H["state_acc"],
    "api": H["api"],
    "schema": H["schema"],
    "vocab": H["vocab"],
    "reason_primary": H["reason_primary"],
    "reason_f1": H["reason_f1"],
    "mandatory_filter": H["mand_filter"],
    "domain": "domain: точное совпадение домена с эталоном. Популяция — эталон ready_for_sql.",
    "metric_f1": "F1 по множеству metric_ids (без учёта порядка). Популяция — эталон ready_for_sql.",
    "source_f1": "F1 по множеству sources. Популяция — эталон ready_for_sql.",
    "entity_f1": "F1 по множеству entity_ids. Популяция — эталон ready_for_sql.",
    "filter_recall": "recall обязательных предикатов required_filters. Популяция — эталон ready_for_sql.",
    "empty_ok": "Инвариант пустоты: для эталона semantic_gap / out_of_scope поля контекста "
                "(метрики / источники / сущности) должны быть пустыми.",
}
for _c in fm.STATES:
    METRIC_DEF[f"handoff_{_c}"] = (
        f"Полностью корректный handoff только на задачах, где эталонный state = {_c}.")
    METRIC_DEF[f"recall_{_c}"] = (
        f"recall класса {_c}: среди задач с эталоном {_c} — доля, где агент тоже поставил {_c}.")
    METRIC_DEF[f"precision_{_c}"] = (
        f"precision класса {_c}: среди задач, где агент поставил {_c} — доля, где эталон тоже {_c}. "
        "Знаменатель зависит от ответов агента (непарная метрика).")


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


def list_prediction_files() -> list[Path]:
    return sorted(OUTPUTS.glob("predictions*.jsonl"), reverse=True)


@st.cache_data(show_spinner=False)
def load_report(pred_path: str, mtime: float) -> fm.Report:
    return fm.evaluate_paths(pred_path)


@st.cache_data(show_spinner=False)
def load_questions() -> dict[str, str]:
    if not TASKS_PATH.exists():
        return {}
    return {r["id"]: r.get("question", "") for r in fm.load_jsonl(TASKS_PATH)}


@st.cache_data(show_spinner=False)
def load_diffs(a_path: str, a_mtime: float, b_path: str, b_mtime: float) -> list:
    a = load_report(a_path, a_mtime)
    b = load_report(b_path, b_mtime)
    return compare.compute_diffs(a, b)


def fmt(value, pct: bool = True, dash: str = "—") -> str:
    if value is None:
        return dash
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def funnel_chart(funnel: list[dict]) -> go.Figure:
    labels = [f"{i + 1}. {stage_label(s)}" for i, s in enumerate(funnel)]
    rates = [s["rate"] for s in funnel]
    colors = [TIER_COLORS[s["tier"]] for s in funnel]
    err_plus = [s["ci_high"] - s["rate"] for s in funnel]
    err_minus = [s["rate"] - s["ci_low"] for s in funnel]
    text = [f"{s['rate'] * 100:.0f}%<br>{s['passed']}/{s['n']}" for s in funnel]
    fig = go.Figure(
        go.Bar(
            x=labels, y=rates, marker_color=colors, text=text, textposition="outside",
            cliponaxis=False,
            error_y=dict(type="data", array=err_plus, arrayminus=err_minus, color="#999"),
            hovertemplate="%{x}<br>пройдено %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Накопленная доля прохождения", range=[0, 1.12], tickformat=".0%"),
        height=440, margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
    )
    return fig


def waterfall_chart(funnel: list[dict]) -> go.Figure:
    measures = ["absolute"] + ["relative"] * (len(funnel) - 1)
    x = [f"{i + 1}. {stage_label(s)}" for i, s in enumerate(funnel)]
    y = [funnel[0]["rate"]] + [-(s["drop_from_prev"]) for s in funnel[1:]]
    text = [f"{funnel[0]['rate'] * 100:.0f}%"] + [
        (f"−{s['dropped_count']}" if s["dropped_count"] else "0") for s in funnel[1:]
    ]
    fig = go.Figure(
        go.Waterfall(
            x=x, y=y, measure=measures, text=text, textposition="outside",
            cliponaxis=False,
            connector=dict(line=dict(color="#bbb")),
            decreasing=dict(marker=dict(color="#E45756")),
            totals=dict(marker=dict(color=TIER_COLORS[1])),
            hovertemplate="%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420, margin=dict(t=40, b=70, l=10, r=10),
        yaxis=dict(tickformat=".0%", title="Потеря задач на стадии", range=[0, 1.12]),
    )
    return fig


def confusion_heatmap(state: dict) -> go.Figure:
    """Двухцветная матрица ошибок.

    Диагональ (верные) — шкала RdYlGn по recall: мало верных = тревожный красный,
    много = спокойный зелёный. Вне диагонали (ошибки) — шкала Reds по доле строки:
    чем больше ошибок ушло в класс, тем тревожнее красный.
    """
    rows = list(fm.STATES)
    cols = state["columns"]
    M = state["confusion"]
    pc = state["per_class"]

    diag_z, off_z, annotations = [], [], []
    for i, g in enumerate(rows):
        row_total = sum(M[g][c] for c in cols)
        drow, orow = [], []
        for c in cols:
            cnt = M[g][c]
            if c == g:
                rec = pc[g]["recall"]
                drow.append(rec if rec is not None else None)
                orow.append(None)
            else:
                drow.append(None)
                orow.append((cnt / row_total) if (row_total and cnt > 0) else None)
            if cnt > 0 or c == g:
                annotations.append(dict(
                    x=c, y=g, text=str(cnt), showarrow=False,
                    font=dict(color="#222", size=14),
                ))
        diag_z.append(drow)
        off_z.append(orow)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=off_z, x=cols, y=rows, colorscale="Reds", zmin=0, zmax=1,
        showscale=False, hoverongaps=False,
        hovertemplate="эталон %{y} → агент %{x}<br>доля ошибок строки %{z:.0%}<extra></extra>",
    ))
    fig.add_trace(go.Heatmap(
        z=diag_z, x=cols, y=rows, colorscale="RdYlGn", zmin=0, zmax=1,
        showscale=False, hoverongaps=False,
        hovertemplate="эталон=агент %{y}<br>recall %{z:.0%}<extra></extra>",
    ))
    fig.update_layout(
        height=380, margin=dict(t=40, b=10, l=10, r=10),
        xaxis=dict(title="Ответ агента", side="top"),
        yaxis=dict(title="Эталон", autorange="reversed"),
        annotations=annotations,
    )
    return fig


def grouped_bars(categories: list[str], series: dict[str, list[float]]) -> go.Figure:
    fig = go.Figure()
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for i, (name, values) in enumerate(series.items()):
        fig.add_bar(name=name, x=categories, y=values, marker_color=palette[i % len(palette)])
    fig.update_layout(
        barmode="group", height=360, margin=dict(t=30, b=10, l=10, r=10),
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------------------


def render_overview(report: fm.Report) -> None:
    f = report.funnel
    endpoint = f[-1]

    # --- Главная метрика: вынесена наверх и визуально отделена ---
    seg = report.handoff_by_state
    seg_lines = "\n".join(
        f"- `{s}`: {fmt(seg[s]['endpoint_rate'])} ({seg[s]['endpoint_passed']}/{seg[s]['n']})"
        for s in fm.STATES
    )
    prime_help = H["prime"] + "\n\n**В разрезе эталонного state:**\n\n" + seg_lines
    with st.container(border=True):
        left, right = st.columns([1, 2])
        left.metric("🎯 Полностью корректный handoff", fmt(endpoint["rate"]), help=prime_help)
        right.markdown(
            "**Главная метрика.** Доля задач, где пакет контекста верен целиком "
            "(все стадии воронки). Для `semantic_gap` / `out_of_scope` правильный "
            "ответ — пустые поля, и он тоже засчитывается.\n\n"
            f"Прошли все стадии: **{endpoint['passed']}/{endpoint['n']}** "
            f"(95% Wilson: {endpoint['ci_low'] * 100:.0f}–{endpoint['ci_high'] * 100:.0f}%)."
        )

    st.divider()

    ctx_gold = report.context_by_mode["golden"]
    c = st.columns(3)
    c[0].metric("Задач", report.n, help=H["n"])
    c[1].metric("Точность state", fmt(report.state["accuracy"]), help=H["state_acc"])
    c[2].metric("SQL-контекст верен (эталон ready)", fmt(ctx_gold["strict_context_rate"]),
                help=H["sql_ctx"])

    st.markdown("#### Воронка из 5 стадий")
    st.caption(
        "Накопленные AND-гейты по всем задачам. Стадии 1-3 — настоящая воронка "
        "выживаемости (технический контракт). Стадии 4-5 — решение (state) и контекст. "
        "Усы = 95% интервал Уилсона. Цвет = уровень (синий·1, оранжевый·2, зелёный·3)."
    )
    tab_funnel, tab_drop = st.tabs(["Воронка", "Потери по стадиям"])
    with tab_funnel:
        st.plotly_chart(funnel_chart(f), use_container_width=True)
    with tab_drop:
        st.plotly_chart(waterfall_chart(f), use_container_width=True)
        biggest = max(f[1:], key=lambda s: s["drop_from_prev"])
        st.info(
            f"Наибольшая потеря: **{stage_label(biggest)}** "
            f"(−{biggest['dropped_count']} задач, "
            f"{biggest['drop_from_prev'] * 100:.0f}% когорты). "
            f"Это уровень {biggest['tier']} — {TIER_NAMES[biggest['tier']]}. Сюда смотреть в первую очередь."
        )

    st.divider()
    render_handoff_segmentation(report)


def render_tier1(report: fm.Report) -> None:
    t1 = report.tier1
    st.markdown(f"### {TIER_NAMES[1]}")
    st.caption("Ответил ли сервис корректным объектом по контракту? Эти гейты "
               "перемножаются: падение здесь обнуляет всё дальше по воронке.")
    c = st.columns(4)
    c[0].metric("Сервис ответил", fmt(t1["api_rate"]), help=H["api"])
    c[1].metric("Валидный JSON", fmt(t1["json_rate"]), help=H["json"])
    c[2].metric("Контракт валиден", fmt(t1["schema_rate"]), help=H["schema"])
    c[3].metric("Без выдуманных id", fmt(t1["vocab_rate"]), help=H["vocab"])

    st.markdown("#### Соответствие словарю semantic layer по полям")
    st.caption("Доля задач, где id поля целиком из semantic layer (нет галлюцинаций имён).")
    vb = t1["vocab_breakdown"]
    st.plotly_chart(grouped_bars(list(vb.keys()), {"валидно": list(vb.values())}),
                    use_container_width=True)

    if t1["failures"]:
        st.markdown("#### Задачи, не прошедшие технический гейт / контракт")
        st.dataframe(t1["failures"], use_container_width=True, hide_index=True)
    else:
        st.success("Все задачи прошли технические гейты и контракт.")


def render_tier2(report: fm.Report) -> None:
    s = report.state
    st.markdown(f"### {TIER_NAMES[2]} — классификация state")
    st.caption("Точка решения (статус готовности). Главный риск — ложный `ready_for_sql` "
               "на gap/неясности (см. AGENT_FLOW_GUIDE).")
    c = st.columns(3)
    c[0].metric("Точность state", fmt(s["accuracy"]), help=H["state_acc_tier"])
    c[1].metric("reason_code: главный", fmt(s["reason_primary_accuracy"]),
                help=H["reason_primary"] + f" Считается на {s['reason_primary_n']} задачах.")
    c[2].metric("reason_codes: F1 множества", fmt(s["reason_f1"]), help=H["reason_f1"])

    # P/R таблица слева, матрица ошибок справа (поменяны местами).
    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("#### Precision / recall по классам", help=H["per_class"])
        classes = list(fm.STATES)
        pc = s["per_class"]
        st.plotly_chart(
            grouped_bars(classes, {
                "precision": [pc[c]["precision"] or 0 for c in classes],
                "recall": [pc[c]["recall"] or 0 for c in classes],
                "F1": [pc[c]["f1"] or 0 for c in classes],
            }),
            use_container_width=True,
        )
        rows = []
        for cls in classes:
            v = pc[cls]
            low = v["gold_n"] < fm.MIN_RELIABLE_N
            rows.append({
                "state": cls + (" ⚠" if low else ""),
                "эталон n": v["gold_n"],
                "precision": fmt(v["precision"]),
                "recall": fmt(v["recall"]),
                "f1": fmt(v["f1"]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"⚠ = меньше {fm.MIN_RELIABLE_N} эталонных задач; читать осторожно.")
    with right:
        st.markdown("#### Матрица ошибок", help=H["confusion"])
        st.plotly_chart(confusion_heatmap(s), use_container_width=True)
        st.caption("Диагональ (верные): зелёный = высокая полнота, красный = низкая. "
                   "Вне диагонали (ошибки): чем краснее, тем большая доля строки ушла "
                   "в неверный класс.")


def render_tier3(report: fm.Report, mode: str) -> None:
    ctx = report.context_by_mode[mode]
    st.markdown(f"### {TIER_NAMES[3]} — корректность полей")
    st.caption(f"Режим знаменателя: **{MODE_LABEL[mode]}** — {MODE_HELP[mode]}")
    low_n = ctx["n"] < fm.MIN_RELIABLE_N
    c = st.columns(4)
    c[0].metric("Размер популяции n", ctx["n"],
                help="Сколько задач в выбранном знаменателе." +
                     (" ⚠ маленькая выборка" if low_n else ""))
    c[1].metric("SQL-контекст верен", fmt(ctx["strict_context_rate"]), help=H["strict_ctx_pop"])
    c[2].metric("Полнота обязат. фильтров", fmt(ctx["mandatory_filter_recall"]),
                help=H["mand_filter"] + f" Считается на {ctx['mandatory_filter_n']} задачах.")
    c[3].metric("Нарушение пустоты", len(ctx["invariant_violations"]), help=H["violations"])

    st.markdown("#### Метрики по полям", help=H["field_vector"])
    fields = ctx["fields"]
    set_fields = [f for f in fields if f["type"] == "set"]
    st.plotly_chart(
        grouped_bars(
            [f["field"] for f in set_fields],
            {
                "exact": [f["exact"] for f in set_fields],
                "precision": [f["precision"] for f in set_fields],
                "recall": [f["recall"] for f in set_fields],
            },
        ),
        use_container_width=True,
    )
    table = [{
        "поле": f["field"], "тип": f["type"],
        "exact": fmt(f["exact"]), "precision": fmt(f["precision"]),
        "recall": fmt(f["recall"]), "f1": fmt(f["f1"]),
    } for f in fields]
    st.dataframe(table, use_container_width=True, hide_index=True)

    if ctx["invariant_violations"]:
        st.warning("gap / out_of_scope с выдуманным контекстом: "
                   + ", ".join(ctx["invariant_violations"]))
        with st.expander("Показать сырые поля по нарушениям", expanded=False):
            _render_raw(report, ctx["invariant_violations"])


def render_handoff_segmentation(report: fm.Report) -> None:
    seg = report.handoff_by_state
    states = list(fm.STATES)
    st.markdown("#### Полностью корректный handoff в разрезе эталонного state",
                help="Сегментация конечной точки воронки по типам задач. Считается по всем "
                     "задачам каждого класса (не зависит от режима знаменателя выше). "
                     "Видно, на каком типе задач handoff разваливается.")
    bar_left, _ = st.columns([1, 0.001])
    with bar_left:
        st.plotly_chart(
            grouped_bars(states, {"handoff верен": [seg[s]["endpoint_rate"] for s in states]}),
            use_container_width=True,
        )
    rows = []
    for s in states:
        d = seg[s]
        low = d["n"] < fm.MIN_RELIABLE_N
        rows.append({
            "эталон state": s + (" ⚠" if low else ""),
            "n": d["n"],
            "handoff верен": d["endpoint_passed"],
            "доля": fmt(d["endpoint_rate"]),
            "95% Wilson": f"{d['ci_low'] * 100:.0f}–{d['ci_high'] * 100:.0f}%",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Воронка по сегментам (стадии × эталонный state)", expanded=False):
        st.caption("Накопленная доля прохождения каждой стадии внутри каждого класса state. "
                   "Видно, на какой стадии сегмент теряет задачи.")
        stage_labels = [f"{i + 1}. {STAGE_LABELS[k]}" for i, (k, _l, _t) in enumerate(fm.FUNNEL_STAGES)]
        series = {s: seg[s]["stage_rates"] for s in states}
        st.plotly_chart(grouped_bars(stage_labels, series), use_container_width=True)


def render_per_task(report: fm.Report) -> None:
    st.markdown("### Детализация по задачам")
    st.caption("Одна строка = одна эталонная задача. Фильтруйте по state, чтобы углубиться.")
    rows = report.rows
    states = ["(все)"] + list(fm.STATES)
    col1, col2 = st.columns(2)
    gold_filter = col1.selectbox("Эталонный state", states, index=0)
    only_fail = col2.checkbox("Только задачи с неверным контекстом", value=False)
    filtered = rows
    if gold_filter != "(все)":
        filtered = [r for r in filtered if r["gold_state"] == gold_filter]
    if only_fail:
        filtered = [r for r in filtered if not r["context_correct"]]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    with st.expander("Показать сырой ответ агента и эталон", expanded=False):
        ids = [r["id"] for r in filtered] or [r["id"] for r in rows]
        chosen = st.multiselect("Задачи", ids, default=ids[:1])
        if chosen:
            _render_raw(report, chosen)


def _render_raw(report: fm.Report, ids: list[str]) -> None:
    questions = load_questions()
    for tid in ids:
        st.markdown(f"**{tid}**")
        if questions.get(tid):
            st.caption(questions[tid])
        gc, pc = st.columns(2)
        gc.markdown("эталон")
        gc.json(report.raw_gold.get(tid, {}), expanded=False)
        pc.markdown("ответ агента")
        pc.json(report.raw_pred.get(tid, {"(нет ответа)": True}), expanded=False)
        st.divider()


# ---------------------------------------------------------------------------
# Режим сравнения двух прогонов (A — кандидат, B — бейзлайн)
# ---------------------------------------------------------------------------

NAV_BY_SECTION = {"overview": "Обзор", "tier1": TIER_NAMES[1],
                  "tier2": TIER_NAMES[2], "tier3": TIER_NAMES[3]}
COMPARE_NAV = ["Различия", "Обзор", TIER_NAMES[1], TIER_NAMES[2], TIER_NAMES[3]]


def _goto(section: str) -> None:
    st.session_state["cmp_nav"] = NAV_BY_SECTION.get(section, "Различия")


def _diff_delta(d: "compare.MetricDiff") -> str:
    if d.from_zero:
        return f"{d.abs_diff * 100:+.0f} п.п. · c ~0"
    if d.rel_diff is not None:
        return f"{d.rel_diff * 100:+.0f}%"
    return f"{d.abs_diff * 100:+.0f} п.п."


def _stat_lines(d: "compare.MetricDiff") -> list[str]:
    parts = [
        f"Бейзлайн B: {fmt(d.b)}",
        f"Абс. Δ (A−B): {d.abs_diff * 100:+.1f} п.п.",
    ]
    if d.rel_diff is not None:
        parts.append(f"Отн. Δ к бейзлайну B: {d.rel_diff * 100:+.0f}%")
    parts.append(f"95% bootstrap CI Δ: [{d.ci_low * 100:+.1f}; {d.ci_high * 100:+.1f}] п.п.")
    parts.append(f"p = {d.p:.3f}")
    parts.append(f"q (BH-FDR) = {d.q:.3f} → "
                 f"{'значимо' if d.significant else 'незначимо'} (порог q<{compare.SIG_Q})")
    if d.discordant is not None:
        parts.append(f"несогласные пары A прав / B прав: {d.discordant[0]} / {d.discordant[1]}")
    parts.append(f"n = {d.n}, тип: {d.kind}")
    if d.low_power:
        parts.append("⚠ мало данных — мощности теста может не хватать")
    return parts


def _stat_badge(d: "compare.MetricDiff") -> str:
    """Отдельный значок с математикой различия (p/q/CI) под нативным hover браузера."""
    title = "\n".join(_stat_lines(d)).replace('"', "'").replace("<", "&lt;")
    return (
        f'<span title="{title}" style="font-size:0.8em;color:#8a8a8a;cursor:help;'
        f'border-bottom:1px dotted #8a8a8a;">🔬 матстат: p · q · CI</span>'
    )


def _metric_card(container, d: "compare.MetricDiff") -> None:
    flags = (" ✦" if d.significant else "") + (" ⚠" if d.low_power else "")
    color = "normal" if d.polarity == compare.HIGHER else "inverse"
    # «?» — определение метрики (идентично одиночному режиму); математика — на значке 🔬.
    container.metric(d.name + flags, fmt(d.a), delta=_diff_delta(d),
                     delta_color=color, help=METRIC_DEF.get(d.id, ""))
    container.markdown(_stat_badge(d), unsafe_allow_html=True)


def metric_grid(diffs: list, *, cols: int = 3, jump: bool = False) -> None:
    for i in range(0, len(diffs), cols):
        row = st.columns(cols)
        for j, d in enumerate(diffs[i:i + cols]):
            with row[j]:
                box = st.container(border=d.significant)
                _metric_card(box, d)
                if jump:
                    st.button("→ к месту", key=f"jump_{d.id}",
                              on_click=_goto, args=(d.section,))


def compare_funnel_chart(fa: list[dict], fb: list[dict], name_a: str, name_b: str) -> go.Figure:
    labels = [f"{i + 1}. {stage_label(s)}" for i, s in enumerate(fa)]
    fig = go.Figure()
    fig.add_bar(name=name_a, x=labels, y=[s["rate"] for s in fa], marker_color="#4C78A8")
    fig.add_bar(name=name_b, x=labels, y=[s["rate"] for s in fb], marker_color="#cccccc")
    fig.update_layout(
        barmode="group", height=420, margin=dict(t=40, b=10, l=10, r=10),
        yaxis=dict(range=[0, 1.08], tickformat=".0%", title="Накопленная доля"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_diff_tab(diffs: list, name_a: str, name_b: str) -> None:
    st.markdown("### Значимые различия")
    st.caption(
        f"**A = {name_a}** (кандидат) против **B = {name_b}** (бейзлайн). "
        "Зелёный — в пользу A, красный — в пользу B. Дельта показана относительно бейзлайна B. "
        "Значок **?** объясняет, как считается метрика; значок **🔬 матстат** скрывает "
        "математику различия (`p`, `q`, доверительный интервал, несогласные пары). "
        "Значимость: парный bootstrap + поправка BH-FDR (`✦` = q<0.05, `⚠` = мало данных)."
    )
    sig = compare.rank_significant(diffs)
    favor_a = sum(1 for d in sig if d.uplift > 0)
    favor_b = sum(1 for d in sig if d.uplift < 0)
    notable = [d for d in diffs if not d.significant and d.low_power and abs(d.abs_diff) >= 0.1]
    c = st.columns(3)
    c[0].metric("Значимых различий", len(sig))
    c[1].metric("В пользу A / B", f"{favor_a} / {favor_b}")
    c[2].metric("Заметных, но мало данных", len(notable))

    only_sig = st.toggle("Только значимые (q<0.05)", value=True)
    shown = sig if only_sig else sorted(
        diffs, key=lambda d: (not d.from_zero, -d.sort_magnitude))
    if not shown:
        st.success("Значимых различий не найдено (q<0.05).")
        return
    metric_grid(shown, jump=True)


def render_compare_section(section: str, diffs: list, report_a: fm.Report,
                           report_b: fm.Report, name_a: str, name_b: str) -> None:
    sub = [d for d in diffs if d.section == section]
    if section == "overview":
        st.markdown("### Обзор — A против B")
        st.plotly_chart(
            compare_funnel_chart(report_a.funnel, report_b.funnel, name_a, name_b),
            use_container_width=True)
    elif section == "tier1":
        st.markdown(f"### {TIER_NAMES[1]} — A против B")
    elif section == "tier2":
        st.markdown(f"### {TIER_NAMES[2]} — A против B")
    elif section == "tier3":
        st.markdown(f"### {TIER_NAMES[3]} — A против B")
    st.caption("Значение — у A; дельта — относительно бейзлайна B. **?** — как считается метрика, "
               "**🔬 матстат** — математика различия (p / q / CI).")
    metric_grid(sub, jump=False)


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------


def _sidebar_meta(report: fm.Report, prefix: str) -> None:
    meta = report.meta
    if meta:
        st.caption(
            f"{prefix} промпт: **{meta.get('system_prompt', '?')}** · "
            f"модель: {meta.get('model', '?')} · "
            f"{meta.get('saved_predictions', '?')}/{meta.get('total_tasks', '?')}"
        )


def main() -> None:
    st.set_page_config(page_title="Context Discovery · воронка", layout="wide")
    st.title("Context Discovery — воронка качества handoff")
    st.caption("Диагностика поверх Context Handoff Score (CHS). Термины — как в README и "
               "AGENT_FLOW_GUIDE. Главное логическое допущение: для gap / out_of_scope "
               "пустые поля = верный ответ.")

    files = list_prediction_files()
    if not files:
        st.error(f"В {OUTPUTS} нет предсказаний. Сначала запустите src/pipeline.py.")
        return

    labels = [p.name for p in files]
    with st.sidebar:
        st.header("Прогон")
        choice = st.selectbox("Файл A (кандидат)", labels, index=0)
        pred_path = OUTPUTS / choice
        report = load_report(str(pred_path), pred_path.stat().st_mtime)
        _sidebar_meta(report, "A:")

        st.divider()
        st.header("Сравнение")
        baseline = st.selectbox("Файл B (бейзлайн)", ["— нет —", *labels], index=0)
        compare_mode = baseline != "— нет —" and baseline != choice
        if baseline == choice and baseline != "— нет —":
            st.warning("B совпадает с A — выберите другой файл.")
        report_b = None
        if compare_mode:
            b_path = OUTPUTS / baseline
            report_b = load_report(str(b_path), b_path.stat().st_mtime)
            _sidebar_meta(report_b, "B:")

        st.divider()
        st.header(f"Знаменатель · {TIER_NAMES[3]}")
        mode = st.radio("Условие", CONDITION_MODES, index=0, format_func=lambda m: MODE_LABEL[m])
        st.caption(MODE_HELP[mode])

    if not compare_mode:
        overview, tier1, tier2, tier3, per_task = st.tabs(
            ["Обзор", TIER_NAMES[1], TIER_NAMES[2], TIER_NAMES[3], "По задачам"]
        )
        with overview:
            render_overview(report)
        with tier1:
            render_tier1(report)
        with tier2:
            render_tier2(report)
        with tier3:
            render_tier3(report, mode)
        with per_task:
            render_per_task(report)
        return

    # --- Режим сравнения ---
    diffs = load_diffs(str(pred_path), pred_path.stat().st_mtime,
                       str(OUTPUTS / baseline), (OUTPUTS / baseline).stat().st_mtime)
    st.session_state.setdefault("cmp_nav", "Различия")
    nav = st.segmented_control("Раздел", COMPARE_NAV, key="cmp_nav")
    nav = nav or "Различия"
    if nav == "Различия":
        render_diff_tab(diffs, choice, baseline)
    else:
        section = {v: k for k, v in NAV_BY_SECTION.items()}[nav]
        render_compare_section(section, diffs, report, report_b, choice, baseline)


if __name__ == "__main__":
    main()
