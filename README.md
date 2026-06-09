# ai_analyst_test_case

Тестовое задание на роль **аналитика-разработчика** в проекте `AI Analyst` (live, 90 минут).

## Контекст

`AI Analyst` — analyst-first продукт: аналитик задаёт бизнес-вопрос, система готовит контекст из `Semantic Layer` (домен, метрики, источники, фильтры, `state`) для черновика SQL под ревью человека.

В репозитории — MVP **Context Discovery** на staging. Основной KPI релизов — **Context Handoff Score (CHS)**. Текущий пайплайн формально укладывается в порог, но поступают жалобы аналитиков на отдельные сценарии. Нужно разобраться и улучшить handoff.

## Главная метрика: Context Handoff Score (CHS)

Единственный KPI прогона `evaluate.py` — **CHS** (`CONTEXT_HANDOFF_SCORE.md`).

| Компонент | Вес | Смысл |
|-----------|-----|--------|
| `operational_reach` | 0.25 | Пайплайн отвечает на задачи |
| `confirmed_path_alignment` | 0.25 | На SQL-ready сценариях state = `ready_for_sql` |
| `undefined_task_conservatism` | 0.25 | На неопределённых задачах корректный `state` |
| `mcp_contract_compliance` | 0.25 | Ответ по контракту (`state`, `reason_codes`, `domain_id`) |

**Порог staging: CHS ≥ 0.95.** После каждого прогона смотрите `context_handoff_score` и наименьший компонент («Фокус улучшения»).

`pipeline.py` в конце **сам печатает CHS**. Отдельно: `python3 src/evaluate.py`.

## Задача

### 1. Baseline (≈15 мин)

```bash
python3 src/pipeline.py --limit 15
```

Зафиксируйте CHS и четыре компонента. Откройте несколько строк `outputs/predictions.jsonl` и сопоставьте с вопросами из `data/tasks.jsonl`.

### 2. Диагностика (≈15 мин)

- Какой компонент CHS слабее остальных?
- Почему аналитики недовольны при текущем CHS?
- Какие типовые ошибки в ответах (state, домен, фильтры куба)?

Ориентир по процедуре агента — `AGENT_FLOW_GUIDE.md`.

### 3. Правки pipeline (≈40 мин)

Улучшите `src/pipeline.py` (промпт, JSON, подача SL). Цель:

1. **Поднять CHS ≥ 0.95** (или закрыть упавшие компоненты).
2. Закрыть реальные проблемы контекста из жалоб аналитиков.

Нельзя: хардкод по `task_id`, keyword-routing в Python.

### 4. Рассказ (≈20 мин)

- CHS до / после по компонентам;
- что изменили в потоке агента;
- trade-offs;
- план на +2 часа.

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 src/pipeline.py
python3 src/evaluate.py
```

## Доступ к LLM API

`OPENAI_API_KEY` и `OPENAI_BASE_URL` уже лежат в `.env` в репозитории (proxy API, модель `gpt-4o-mini`). После клонирования ничего дополнительно настраивать не нужно.

Если `.env` отсутствует — скопируйте `.env.example` и запросите ключ у интервьюера.

## Репозиторий

```text
.env, .env.example
data/tasks.jsonl, data/golden.jsonl, data/semantic_layer.json
src/pipeline.py, src/evaluate.py
CONTEXT_HANDOFF_SCORE.md, AGENT_FLOW_GUIDE.md
```

## Модель и бюджет

`gpt-4o-mini`. Полный прогон 60 задач: ~150–300 ₽.

## Время

**90 минут** live.
