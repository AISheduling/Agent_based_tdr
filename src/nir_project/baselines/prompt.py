"""Prompt of the structured LLM baseline (L1): one call, no documents, no verifier feedback.

Bump PROMPT_VERSION on any change of the text: it is logged with every run.
"""

from __future__ import annotations

from nir_project.baselines.package import Package

PROMPT_VERSION = "l1-v1"

SYSTEM_PROMPT = """\
Ты инженер-технолог строительного производства. Тебе дают фиксированный список операций одного \
вида работ. Восстанови технологические связи предшествования между ними: модель «окончание — \
начало» (B начинается только после окончания A).

Типы связей:
- mandatory — B технологически нельзя начинать до окончания A. Допустимые основания (basis):
  - object_state — B требует состояния объекта, которое создаёт A;
  - result_consumption — B использует результат A (B работает на том, что сделала A);
  - zone_access — A закрывает или занимает доступ к зоне, где работает B;
  - tech_card — порядок прямо требует технологическая карта или норматив.
- preferred — порядок желателен, но B физически и нормативно можно выполнить и раньше.
- unknown — не уверен, есть ли технологическая связь. За unknown не штрафуют; \
лучше unknown, чем выдуманная обязательная связь.

Не являются технологической связью: одна бригада или один кран, очередь поставки, привычный \
порядок в графике, административное решение. Такие пары не связывай. Если работы можно вести \
параллельно, связи между ними нет. Одна статистическая повторяемость (statistical_repeat) не \
делает связь обязательной.

Правила ответа:
- используй только id из списка операций, новых операций не придумывай;
- указывай только непосредственные связи: если A→B и B→C, связь A→C не нужна;
- граф без циклов;
- у каждой операции, кроме первых, обычно есть хотя бы один предшественник;
- reason — одно короткое предложение по-русски, почему связь такая.

Ответ — только JSON такого вида:
{"relations": [
  {"from": "<id A>", "to": "<id B>", "type": "mandatory", "basis": ["result_consumption"], \
"reason": "..."},
  {"from": "<id C>", "to": "<id D>", "type": "unknown", "basis": [], "reason": "..."}
]}
"""


def render_user_prompt(package: Package) -> str:
    lines = [f"Вид работ: {package.title}."]
    if package.scope:
        lines.append(f"Описание: {package.scope}")
    if package.conditions:
        lines.append("")
        lines.append("Условия этого объекта:")
        for c in package.conditions:
            lines.append(f"- {c.description or c.name}: {'да' if c.value else 'нет'}")
    if package.external_inputs:
        lines.append("")
        lines.append("Поступает извне (не операции, связи с ними не нужны):")
        lines.extend(f"- {name}" for name in package.external_inputs)
    lines.append("")
    lines.append(f"Операции ({len(package.operations)}, порядок списка случайный):")
    lines.extend(f"- {o.id} [{o.kind or '-'}]: {o.name}" for o in package.operations)
    lines.append("")
    lines.append("Верни JSON со связями между этими операциями.")
    return "\n".join(lines)


def build_messages(package: Package) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(package)},
    ]
