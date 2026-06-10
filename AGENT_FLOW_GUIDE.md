# Инструкция контекстного агента (Context Discovery)

Справочник по процедуре агента и доменным правилам для тестового задания.

## Словарь формулировок аналитика

| Как пишут в вопросе | Что искать в SL |
|---------------------|-----------------|
| «динамика карт / по месяцам» | метрика `client_cnt` (или другая из вопроса) + grain по времени в SQL, не отдельная метрика |
| «амбассадоры / АСР» | поле `segment` в кубе чеков (`assortment`), не домен `loyalty` |
| «маржа / выручка в кубе» | `base_amt_sum` из `mart.check_aggregated_by_cube`; наценка SKU — `pricing` / `shelf_markup_offline` |
| «средний чек» | `avg_check_amt` из куба, не `mart_mixed.sku_price` |
| «бонусы / купоны» | домен `loyalty`, `mart.loyalty_events` |
| «списания / OOS» | домен `supply`, `mart.inventory_daily`, не куб чеков |
| «поисковые сессии» | `all_searches` в `search`, не `session_cnt` приложения |

## Цель этапа

Агент **не** выдаёт финальный бизнес-ответ и **не** исполняет SQL в DWH. Он готовит **пригодный контекст** для аналитика:

- домен (`domain_id`);
- метрики (`metric_ids`);
- источники (`sources`);
- сущности (`entity_ids`), если важны для grain;
- обязательные фильтры (`required_filters`);
- решение о готовности (`state`);
- причины (`reason_codes`) и краткие доказательства (`evidence`).

## Процедура (порядок обязателен)

1. **Намерение** — что хочет аналитик: выгрузка, сравнение, динамика, поиск метрики, проверка гипотезы.
2. **Policy** — нет ли PII, рассылок, обхода ревью → `out_of_scope`.
3. **Домен** — по смыслу задачи, не по одному слову в вопросе.
4. **Покрытие SL** — есть ли метрика и источник в переданном semantic layer → иначе `semantic_gap`.
5. **Неоднозначность** — не заданы период, категория, grain, список слов → `needs_clarification`.
6. **Метрики и источник** — из SL домена; `primary_source` метрики согласован с `sources`.
7. **Обязательные фильтры** — из SL метрики и доменных правил (куб Total, экран поиска и т.д.).
8. **State** — только после шагов 1–7.

## Домены в кейсе (7)

| `domain_id` | Когда выбирать | Типовой источник |
|-------------|----------------|------------------|
| `assortment` | Продажи, АСР, куб чеков, жалобы, средний чек | `mart.check_aggregated_by_cube`, `report.v_sku_complaints` |
| `search` | Поиск, конверсия в добавление, пустая выдача | `logs.tovs_search` |
| `pricing` | Цены SKU, наценка, себестоимость, переоценки | `mart_mixed.sku_price` |
| `delivery` | Заказы и отмены доставки (**не** NPS/CSAT) | `mart.delivery_orders` |
| `loyalty` | Бонусы, купоны (**не** сегменты АСР в кубе) | `mart.loyalty_events` |
| `supply` | OOS, списания, потери склада | `mart.inventory_daily` |
| `app_engagement` | DAU, сессии МП (**не** конверсия поиска) | `logs.app_events` |

## Состояния

| State | Условие |
|-------|---------|
| `ready_for_sql` | Домен, метрики, источник и критичные фильтры определены; неоднозначности не блокируют черновик |
| `needs_clarification` | Не хватает периода, категории, списка, grain или метрика неоднозначна |
| `semantic_gap` | Задача аналитическая, но объекта нет в SL |
| `out_of_scope` | PII, маркетинг, не аналитическая задача |

**Запрещено:** ставить `ready_for_sql` при `semantic_gap` или явной неоднозначности «сам догадаюсь».

## Приоритеты ошибок (сверху вниз)

1. `out_of_scope` / policy
2. `semantic_gap`
3. `ambiguous_request` → `needs_clarification`
4. `wrong_domain`
5. `wrong_source` / `wrong_metric`
6. `missing_required_filter` (куб Total и др.)
7. Ложный `ready_for_sql` при gap / clarification

## Домен assortment + куб

- Источник: `mart.check_aggregated_by_cube`.
- Метрики: `client_cnt`, `check_cnt`, `base_amt_sum`, `quantity_sum`, `avg_check_amt`; жалобы — `complaints_count` + `report.v_sku_complaints`.
- «Подгруппа (2 уровень)» → `sku_group`.
- «Амбассадоры / АСР» → поле `segment` в кубе (фильтр или разрез), не события лояльности.
- Без фильтров `Total` на измерениях куба — риск неверного grain.

## Домен search

- Источник: `logs.tovs_search`.
- Метрики: `all_searches`, `a2c_count_from_search`, `zero_result_rate`.
- Обязательно: экран `catalog_search`, непустой `id_search`; для A/B — список `search_bar`.

## Домен pricing

- Источник: `mart_mixed.sku_price`.
- Метрики: `shelf_markup_offline`, `cost_offline`, `price_change_cnt`.
- Наценка offline: `is_offline = true`, `price_offline <> 0`.

## Домен delivery

- Источник: `mart.delivery_orders`.
- Метрики: `delivery_order_cnt`, `delivery_cancel_rate`.
- Фильтр: `order_status != 'test'`.
- **Нет в SL:** NPS, CSAT, самовывоз → `semantic_gap`.

## Домен loyalty

- Источник: `mart.loyalty_events`.
- Метрики: `bonus_points_accrued` (`event_type = accrual`), `coupon_redemption_cnt` (`event_type = redemption`).
- Бонусы и купоны **не** берутся из куба чеков, даже если в вопросе «амбассадоры» или «кулинария».

## Домен supply

- Источник: `mart.inventory_daily`.
- Метрики: `stockout_sku_days` (`on_hand_qty = 0`), `shrinkage_amount` (`movement_type` shrinkage/writeoff).
- Списания и OOS **не** из `sku_price` и **не** из куба чеков.

## Домен app_engagement

- Источник: `logs.app_events`.
- Метрики: `session_cnt`, `dau` (`event_name = app_open`).
- Фильтр: `platform in (ios, android)`.
- DAU/сессии **не** путать с `all_searches` / конверсией поиска.

## Как диагностировать ошибки

В датасете **60 задач** и **22 метрики**. Полный разбор всех кейсов на 90 минут нецелесообразен.

Рабочий порядок:

1. `python3 src/pipeline.py` — в конце прогона смотреть **CHS** и слабый компонент.
2. `python3 src/evaluate.py` — повторить CHS при необходимости.
3. Выборочно открыть строки `outputs/predictions.jsonl` и сопоставить с вопросами из `tasks.jsonl`.
4. Для парных `family` в `tasks.jsonl` — сравнить два похожих вопроса и сформулировать правило.

Типовые проблемы: ложный `ready_for_sql`, путаница доменов («сессии», «маржа»), неверный источник, пропуск `Total` в кубе, gap при отсутствии метрики в SL.

## Коды причин

- `good_context_match` — главная причина при `ready_for_sql`
- `partial_context` — полезно, но неполно (дополнительный код)
- `wrong_domain`, `wrong_metric`, `wrong_source`
- `missing_required_filter`
- `ambiguous_request`
- `semantic_gap`
- `out_of_scope`

Первый `reason_code` — главная причина итогового `state`.

## Похожие метрики в SL

В semantic layer есть метрики с близкими названиями из разных доменов: `shelf_revenue_offline`, `search_session_cnt`, `loyalty_card_cnt`. Выбор только по совпадению слов в вопросе часто даёт неверный домен или источник — проверяйте `primary_source` и доменные правила выше.
