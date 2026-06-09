"""MVP Context Discovery agent flow для AI Analyst.

Стартовая версия пайплайна для тестового задания: подбор контекста из Semantic Layer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "data" / "tasks.jsonl"
SL_PATH = ROOT / "data" / "semantic_layer.json"
OUTPUT_PATH = ROOT / "outputs" / "predictions.jsonl"

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """Ты AI-аналитик. По вопросу пользователя подбери метрики и таблицы из semantic layer.

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
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
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


def normalize_output(value: dict[str, Any]) -> dict[str, Any]:
    """Минимальная техническая нормализация без оценочной логики."""
    state = value.get("state") or "ready_for_sql"
    return {
        "domain_id": value.get("domain_id"),
        "metric_ids": value.get("metric_ids") if isinstance(value.get("metric_ids"), list) else [],
        "sources": value.get("sources") if isinstance(value.get("sources"), list) else [],
        "required_filters": value.get("required_filters")
        if isinstance(value.get("required_filters"), list)
        else [],
        "state": state,
        "reason_codes": value.get("reason_codes")
        if isinstance(value.get("reason_codes"), list)
        else ["good_context_match"],
        "evidence": value.get("evidence") if isinstance(value.get("evidence"), list) else [],
    }


def run_agent(client: OpenAI, question: str, sl: dict) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, sl)},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    return normalize_output(parse_response(raw))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline context discovery pipeline.")
    parser.add_argument("--input", type=Path, default=TASKS_PATH)
    parser.add_argument("--semantic-layer", type=Path, default=SL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_jsonl(args.input)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    sl = load_semantic_layer(args.semantic_layer)
    client = build_client()
    total = len(tasks)
    print(f"Загружено {total} задач", flush=True)

    with args.output.open("w", encoding="utf-8") as f:
        for i, row in enumerate(tasks, 1):
            print(f"Кейс {i}/{total}: {row['id']}", flush=True)
            try:
                result = run_agent(client, row["question"], sl)
            except Exception as exc:  # noqa: BLE001
                print(f"  ошибка: {exc}", flush=True)
                result = {
                    "domain_id": None,
                    "metric_ids": [],
                    "sources": [],
                    "required_filters": [],
                    "state": "ready_for_sql",
                    "reason_codes": ["good_context_match"],
                    "evidence": [str(exc)],
                }
            out = {"id": row["id"], **result}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"  state={result.get('state')}", flush=True)

    print(f"Готово: {args.output}", flush=True)

    from evaluate import DEFAULT_GOLDEN, run_evaluation

    if DEFAULT_GOLDEN.exists():
        print(flush=True)
        run_evaluation(predictions_path=args.output)


if __name__ == "__main__":
    main()
