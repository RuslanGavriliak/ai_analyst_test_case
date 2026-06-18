"""MVP Context Discovery agent flow для AI Analyst.

Стартовая версия пайплайна для тестового задания: подбор контекста из Semantic Layer.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "data" / "tasks.jsonl"
SL_PATH = ROOT / "data" / "semantic_layer.json"
OUTPUT_DIR = ROOT / "outputs"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3
RESPONSE_FORMAT = {"type": "json_object"}

SYSTEM_PROMPT_V1 = """Ты AI-аналитик. По вопросу пользователя подбери метрики и таблицы из semantic layer.

Верни JSON с полями:
- domain_id (строка)
- metric_ids (массив строк)
- sources (массив строк)
- required_filters (массив строк, можно пустой)
- state (строка: ready_for_sql)
- reason_codes (массив строк)
- evidence (массив из 1-2 коротких строк)

Старайся помочь пользователю: если вопрос похож на SQL-задачу, ставь state ready_for_sql.
Выбирай метрики по совпадению слов в вопросе."""

SYSTEM_PROMPT_V2 = """Ты AI-аналитик. По вопросу пользователя подбери метрики и таблицы из semantic layer.

Верни только валидный JSON-объект без markdown, без ```json, без комментариев и без текста вокруг.

JSON поля:
- domain_id (строка)
- metric_ids (массив строк)
- sources (массив строк)
- required_filters (массив строк, можно пустой)
- state (строка: ready_for_sql или needs_clarification или semantic_gap или out_of_scope)
- reason_codes (массив строк)
- evidence (массив из 1-2 коротких строк)

Как выбирать state:
- ready_for_sql: домен, метрики, источники и критичные фильтры определены из semantic layer; контекст можно передавать аналитику для черновика SQL.
- needs_clarification: задача относится к semantic layer, но не хватает обязательного параметра для честного контекста: периода, категории, grain, списка значений или однозначной метрики.
- semantic_gap: запрос аналитический и понятный, но нужной метрики, домена, источника или бизнес-объекта нет в semantic layer. Не выдумывай метрики и источники.
- out_of_scope: запрос вне контура analyst-first продукта: PII, персональные данные, рассылки, обход ревью человека, автономное исполнение SQL/действий в проде или неаналитическая задача.

Правила:
- Используй только domain_id, metric_ids и sources, которые есть в semantic layer.
- Для semantic_gap и out_of_scope возвращай пустые metric_ids и sources, если в semantic layer нет честного контекста.
- Для needs_clarification не выдумывай недостающие параметры; укажи, чего не хватает, в evidence.
- Не ставь ready_for_sql при явном semantic gap, policy gap или критичной неоднозначности.
- reason_codes не должен быть пустым: используй good_context_match, ambiguous_request, semantic_gap или out_of_scope как основной код.
"""

SYSTEM_PROMPT_V3 = """Ты AI-аналитик этапа Context Discovery.

Твоя задача: по бизнес-вопросу аналитика подготовить JSON-пакет контекста из переданного semantic layer для следующего шага — черновика SQL под ревью человека. Ты не отвечаешь на бизнес-вопрос, не пишешь SQL и не исполняешь запросы. Ты только выбираешь домен, метрики, источники, сущности, обязательные фильтры, state, reason_codes и evidence.

Верни только валидный JSON-объект без markdown, без ```json, без комментариев и без текста вокруг.

JSON contract:
- domain_id: строка или null. Если выбран домен, используй только id домена из semantic layer.
- metric_ids: массив строк. Используй только id метрик из semantic layer; если честного контекста нет, верни [].
- sources: массив строк. Используй только id источников из semantic layer; source должен соответствовать выбранным метрикам.
- entity_ids: массив строк. Используй только эти entity id: sku_category, sku_group, segment, number, sku, city, coupon, store, session, user.
- required_filters: массив строк. Укажи обязательные фильтры метрик/источников и критичные фильтры из вопроса; если фильтров нет, верни [].
- state: одна строка из списка: ready_for_sql, needs_clarification, semantic_gap, out_of_scope.
- reason_codes: непустой массив строк. Можно вернуть несколько кодов; первый код — главная причина итогового state.
- evidence: массив из 1-2 коротких строк с опорой на вопрос и semantic layer.

State definitions:
- ready_for_sql: домен, метрики, источник и критичные фильтры определены; неоднозначности не блокируют черновик SQL.
- needs_clarification: не хватает параметра, без которого нельзя честно собрать контекст.
- semantic_gap: запрос аналитический, но нужного объекта, метрики, домена или источника нет в semantic layer.
- out_of_scope: запрос вне политики продукта: PII, персональные данные, маркетинговые рассылки, обход ревью человека, автономное исполнение SQL/действий в проде или неаналитическая задача.

Decision priority:
1. Policy violation -> out_of_scope.
2. Нужного объекта нет в semantic layer -> semantic_gap.
3. Не хватает обязательных параметров -> needs_clarification.
4. Иначе выбери домен, метрики, источники, entity_ids и required_filters.
5. Только после этого выставь state.

Запрещено ставить ready_for_sql при явном gap, неясном периоде или догадке без опоры на semantic layer.

Allowed reason_codes:
- good_context_match: контекст согласован с вопросом и semantic layer.
- partial_context: контекст полезен, но намеренно неполон; используй как дополнительный код.
- wrong_domain: домен в вопросе/подсказке не по смыслу задачи.
- wrong_metric: метрика не из того домена или не та сущность.
- wrong_source: источник не соответствует метрике или вопросу.
- missing_required_filter: не указаны обязательные фильтры из semantic layer.
- ambiguous_request: в вопросе не хватает параметров.
- semantic_gap: нужного объекта нет в semantic layer.
- out_of_scope: запрос вне контура продукта.

Semantic layer rules:
- Один вопрос — один ведущий домен, если в semantic layer не описан явный кросс-доменный сценарий. Составные вопросы («сессии приложения и конверсия поиска») — несколько метрик/источников с явным разделением в evidence.
- Слова в вопросе ≠ id метрики. Сначала намерение и домен, потом поиск по name, synonyms и primary_source в semantic layer.
- Источник в вопросе может быть неверным. Если аналитик назвал таблицу, которая не подходит метрике, в контекст кладётся правильный источник из semantic layer; в reason_codes — wrong_source, если это было важно для решения.
- Предагрегированные витрины — всегда проверяй required_filters метрики и правила источника: без них возможна двойная агрегация или неверный grain.
- Похожие метрики в разных доменах — выбирай по domain_id, primary_source и описанию в semantic layer, не по совпадению одного слова.
"""

SYSTEM_PROMPTS = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
    "v3": SYSTEM_PROMPT_V3,
}

USER_PROMPT = """Вопрос аналитика:
{question}

Semantic layer (JSON):
{semantic_layer}
"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_semantic_layer(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-..."):
        raise SystemExit(
            "Не задан OPENAI_API_KEY. Проверьте файл .env в корне репозитория "
            "(должен быть после git clone) или скопируйте .env.example."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def build_user_prompt(question: str, sl: dict) -> str:
    return USER_PROMPT.format(
        question=question,
        semantic_layer=json.dumps(sl, ensure_ascii=False),
    )


def parse_response(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {raw[:200]}") from exc
    if not isinstance(value, dict):
        raise ValueError("Response is not a JSON object")
    return value


def run_agent(client: OpenAI, question: str, sl: dict, system_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format=RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(question, sl)},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    return parse_response(raw)


def default_predictions_file_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"predictions{timestamp}.jsonl"


def meta_file_path(predictions_file_path: Path) -> Path:
    return predictions_file_path.with_suffix(".meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline context discovery pipeline.")
    parser.add_argument("--input", type=Path, default=TASKS_PATH)
    parser.add_argument("--semantic-layer", type=Path, default=SL_PATH)
    parser.add_argument(
        "--predictions-file-path",
        "--output",
        dest="predictions_file_path",
        type=Path,
        default=None,
        help="Path for predictions JSONL. Defaults to outputs/predictions{timestamp}.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--system-prompt",
        choices=sorted(SYSTEM_PROMPTS),
        default="v3",
        help="System prompt version to use: v1 is baseline, v2 is improved, v3 is full contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now().astimezone().isoformat()
    predictions_file_path = args.predictions_file_path or default_predictions_file_path()
    run_meta_file_path = meta_file_path(predictions_file_path)
    predictions_file_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_jsonl(args.input)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    sl = load_semantic_layer(args.semantic_layer)
    system_prompt = SYSTEM_PROMPTS[args.system_prompt]
    client = build_client()
    total = len(tasks)
    saved_predictions = 0
    errors: list[dict[str, str]] = []
    print(f"Загружено {total} задач", flush=True)
    print(f"System prompt: {args.system_prompt}", flush=True)

    with predictions_file_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(tasks, 1):
            print(f"Кейс {i}/{total}: {row['id']}", flush=True)
            try:
                result = run_agent(client, row["question"], sl, system_prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"  ошибка: {exc}", flush=True)
                errors.append(
                    {
                        "id": row["id"],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                continue
            out = {"id": row["id"], **result}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            saved_predictions += 1
            print(f"  state={result.get('state')}", flush=True)

    print(f"Готово: {predictions_file_path}", flush=True)
    run_meta = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "input_path": str(args.input),
        "semantic_layer_path": str(args.semantic_layer),
        "predictions_file_path": str(predictions_file_path),
        "meta_file_path": str(run_meta_file_path),
        "limit": args.limit,
        "total_tasks": total,
        "saved_predictions": saved_predictions,
        "failed_predictions": len(errors),
        "errors": errors,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "response_format": RESPONSE_FORMAT,
        "system_prompt": args.system_prompt,
        "system_prompt_text": system_prompt,
    }
    with run_meta_file_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Meta: {run_meta_file_path}", flush=True)

    from evaluate import DEFAULT_GOLDEN, run_evaluation

    if DEFAULT_GOLDEN.exists():
        print(flush=True)
        run_evaluation(predictions_path=predictions_file_path)


if __name__ == "__main__":
    main()
