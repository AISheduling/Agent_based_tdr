# data/

Ignored by Git except this file, `sources.yaml` and `families/`. Raw data, weights and large files
live in MLflow, MinIO or the cluster storage; the project README says how to fetch them.

## Корпус шаблонов (эталон)

Маленький текстовый корпус, который служит эталоном для верификатора и baselines. Формат и правила —
[ADR 0001](../docs/adr/0001-relation-format-and-reference-partial-order.md).

```
data/
  sources.yaml                         реестр источников: формат, объём, доступ, лицензия
  families/<family>/
    <family>.yaml                      словарь операций: requires / produces, параметры семейства
    <family>_relations.yaml            relations, interface_relations, non_relations
```

| Семейство | Источники | Операций | Связей | Параметры |
|---|---|---|---|---|
| `monolithic_floor` — монолитный этаж в крупнощитовой опалубке | ТТК ЦНИИОМТП 2004, СП 435 | 39 | 45 | — |
| `flat_roof` — плоская кровля из наплавляемого материала | ТТК «Филизол» 2001, СП 71, ТТК 1991 | 23 | 31 | `screed`, `bitumen_vapor_barrier`, `expansion_joint` |

- `relations` — связи между операциями семейства с типом, основанием, источником, локатором и статусом.
- `interface_relations` — связи с внешними входами и выходами (готовность основания, поставки).
- `non_relations` — пары операций, порядок между которыми не технологический (одна бригада, поставка,
  параллельные работы).

Полные тексты ТТК и СП в репозиторий не копируются: хранятся только извлечённые связи со ссылкой
на пункт документа.
