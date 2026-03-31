"""
Conditional question filtering based on visible_if expressions.

visible_if format (from zepbound_question_set.json):
  "{key} = value"
  "{key} = value and {key2} = value2"

Conditions are ANDed together. Keys are wrapped in curly braces.
Values are plain strings (typically "true" or "false").
"""

import re

from app.models import Question


def _parse_condition(condition: str) -> tuple[str, str]:
    """Parse a single condition like '{key} = value' into (key, value)."""
    match = re.match(r"\{(\w+)\}\s*=\s*(\S+)", condition.strip())
    if not match:
        raise ValueError(f"Unparseable visible_if condition: {condition!r}")
    return match.group(1), match.group(2)


def _evaluate_visible_if(visible_if: str, current_answers: dict[str, str]) -> bool:
    """Return True if all conditions in the visible_if expression are satisfied."""
    conditions = [c.strip() for c in visible_if.split(" and ")]
    for condition in conditions:
        try:
            key, expected_value = _parse_condition(condition)
        except ValueError:
            return False
        actual = current_answers.get(key, "")
        if actual.lower() != expected_value.lower():
            return False
    return True


def filter_visible_questions(
    questions: list[Question], current_answers: dict[str, str]
) -> list[Question]:
    """
    Return questions that are visible given the current set of answers.

    A question with no visible_if is always included.
    A question with visible_if is included only when all its conditions are met.
    """
    visible: list[Question] = []
    for q in questions:
        if q.visible_if is None:
            visible.append(q)
        elif _evaluate_visible_if(q.visible_if, current_answers):
            visible.append(q)
    return visible
