# Минимальный верификатор графа

Пакет `src/nir_project/verifier/`. Формат отношения и эталона — [ADR 0001](adr/0001-relation-format-and-reference-partial-order.md).

## Что проверяется

Две точки входа. `check_family` проверяет сам эталон (по каждому экземпляру семейства отдельно),
`check_graph` / `verify_graph` — граф, в том числе ответ модели. Каждая проблема — `Issue` со
стабильным `code`.

| Проверка | code | Где |
|---|---|---|
| Цикл в обязательных связях | `cycle` | эталон, ответ |
| Операция словаря не вошла в граф | `uncovered_operation` | ответ; в эталоне — операция без единой связи |
| Операция вне словаря (порождать новые нельзя) | `unknown_operation` | ответ |
| Обязательная связь без основания / источника | `no_basis`, `no_source` | эталон, ответ |
| Обязательная связь только на `statistical_repeat` | `weak_basis` | эталон, ответ |
| Связь между подтверждённо независимыми операциями | `non_relation_violated` | эталон, ответ |
| `requires` без связи и связь без `requires` | `requires_without_relation`, `relation_without_requires` (и `*_interface`) | эталон |
| Несколько производителей состояния, нет производителя | `multiple_producers`, `no_producer` | эталон |
| Независимость операции с самой собой | `invalid_pair` | эталон |
| Внешняя связь, не соединяющая операцию с внешним входом или выходом | `invalid_interface` | эталон |
| Формат: типы, основания, статусы, условия, реестр источников | `invalid_*`, `unknown_*`, `missing_condition` и др. | эталон |
| Операция без связей | `isolated_operation` (предупреждение) | ответ |

## Сравнение с эталоном

Замыкание строится только по `mandatory` (в экземпляре — и по активным `conditional`). TP / FP / FN
считаются по парам замыкания, поэтому транзитивно избыточная дуга ничего не меняет. `unknown` в ответе
не считается ложным и отчитывается отдельно (`unknown_pairs`, `fn_declared_unknown`).

Основная метрика — `recall` при `fp_rate = FP / (TP + FP) <= 0.02` (`FP_BUDGET`). Дополнительно:
`fp_on_non_relation`, `fp_on_preferred` — разбор ложных пар.

В метрику идут только рёбра, которые принимает `check_graph`: ребро к операции, не указанной
в `operations` ответа, отмечается ошибкой и ничего не приносит. Пара, которую ответ и утверждает
как `mandatory`, и помечает `unknown`, воздержанием не считается.

Бюджет считается по парам замыкания, поэтому одно ложное ребро в замыкании из ~200 пар укладывается в 2%.
Такое ребро ловят структурные проверки (`weak_basis`, `non_relation_violated`), а не бюджет.

## Запуск

```
make verify                                    # эталон всех семейств
uv run python -m nir_project.verifier family data/families/flat_roof data/families/monolithic_floor
uv run python -m nir_project.verifier graph data/families/flat_roof answer.yaml --param screed=1
uv run python -m nir_project.verifier graph data/families/flat_roof answer.yaml --json
```

Коды завершения: 0 — ошибок нет, 1 — найдены ошибки верификации, 2 — плохой вход (файл не найден,
битый YAML, неверные аргументы). С `--json` в stdout выводится только JSON
(`instance`, `ok`, `issues`, `metrics`). `--param` принимает `NAME=0|1` (или `true`/`false`,
`yes`/`no`); неизвестное имя параметра — ошибка, а не молча выключенный параметр. Реестр
`data/sources.yaml` ищется относительно текущей папки; если его нет, проверки реестра пропускаются
с предупреждением в stderr, а явно указанный `--sources` обязан существовать.

Формат ответа: `{operations: [...], relations: [{from, to, type, basis, sources}]}`. Списки
должны быть списками: строка на месте списка (`basis: object_state`) — `FormatError`, а не
разбор по буквам; `type` —
`mandatory` / `preferred` / `unknown` (`conditional` в графе экземпляра недопустим). Параметры
задают экземпляр семейства; `make_instance` / `enumerate_instances` в Python.

## Тесты

`tests/test_verifier.py`. Для каждого кода ошибки есть случай, который ломает реальное семейство
`flat_roof` именно так, как описывает код (`REFERENCE_DEFECTS`, `ANSWER_DEFECTS`).
`test_every_issue_code_has_a_defect_case` падает, если в `checks.py` появился код без такого случая.

## Ограничения

- Только FS-связи; внешние связи (`interface_relations`) в метрику не входят, но проверяются на покрытие.
- Статусы `needs_check` и `expert_needed` входят в эталон; как их считать в метрике — вопрос к руководителю.
