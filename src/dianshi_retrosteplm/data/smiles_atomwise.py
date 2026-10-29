#!/usr/bin/env python3
"""Convert recognized SMILES spans to a space-separated atom-wise form."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable


SMILES_ATOM_PATTERN = r"""
    \[[^\[\]\r\n]+\]
  | Br | Cl
  | B | C | N | O | S | P | F | I
  | b | c | n | o | s | p
  | \( | \) | \.
  | >>? | -> | <-
  | = | \# | - | \+ | \\ | /
  | : | ~ | @ | \?
  | \* | \$
  | %[0-9]{2}
  | [0-9]
"""

SMILES_ATOM_RE = re.compile(SMILES_ATOM_PATTERN, re.VERBOSE)


class SmilesTokenizationError(ValueError):
    """Raised when the atom-wise expression cannot consume the whole string."""


def atomwise_tokens(smiles: str) -> list[str]:
    if not isinstance(smiles, str):
        raise TypeError(f"smiles must be str, got {type(smiles).__name__}")
    if not smiles:
        raise SmilesTokenizationError("empty SMILES")
    tokens = []
    cursor = 0
    for match in SMILES_ATOM_RE.finditer(smiles):
        if match.start() != cursor:
            raise SmilesTokenizationError(
                f"unmatched text at offset {cursor}: {smiles[cursor:match.start()]!r}"
            )
        tokens.append(match.group(0))
        cursor = match.end()
    if cursor != len(smiles):
        raise SmilesTokenizationError(
            f"unmatched text at offset {cursor}: {smiles[cursor:]!r}"
        )
    if not tokens:
        raise SmilesTokenizationError(f"no SMILES token found in {smiles!r}")
    return tokens


def to_atomwise_smiles(smiles: str) -> str:
    stripped = smiles.strip()
    try:
        return " ".join(atomwise_tokens(stripped))
    except SmilesTokenizationError:
        if not any(character.isspace() for character in stripped):
            raise
        return " ".join(atomwise_tokens("".join(stripped.split())))


def try_to_atomwise_smiles(value: str) -> str | None:
    try:
        return to_atomwise_smiles(value)
    except (SmilesTokenizationError, TypeError):
        return None


@dataclass
class TransformResult:
    text: str
    transformed_spans: int


ANSWER_RE = re.compile(r"(<answer>)(.*?)(</answer>)", re.DOTALL | re.IGNORECASE)
S2S_SYSTEM_MARKERS = (
    "input consists of a single product smiles",
    "output only the corresponding reactant smiles",
)
SENTENCE_SMILES_RE = re.compile(
    r"(?P<prefix>\bThe\s+(?:product|reactants?)\s+SMILES\s+is\s+)"
    r"(?P<value>\S+?)"
    r"(?P<suffix>\.(?=\s+(?:Below|Based|Using|Please|Predict|Given|The|By)\b)|"
    r"(?=\s*$))",
    re.IGNORECASE,
)
LINE_SMILES_RE = re.compile(
    r"(?im)^(?P<prefix>[ \t]*(?:"
    r"reactants|products|reactant|product|SMILES|product_smiles|reactant_smiles|"
    r"product\s+SMILES|reactant\s+SMILES|predicted\s+reactant(?:\s+\d+)?"
    r"):[ \t]*)(?P<value>\S+?)(?P<suffix>[ \t]*)$"
)
SYNTHON_LIST_RE = re.compile(
    r"(?im)^(?P<prefix>[ \t]*synthons_smiles:[ \t]*)(?P<value>\[.*\])"
    r"(?P<suffix>[ \t]*)$"
)
INLINE_SMILES_LIST_RE = re.compile(
    r"(?P<prefix>\b(?:"
    r"the\s+synthons\s+involved\s+in\s+this\s+reaction\s+are\s+identified\s+as|"
    r"the\s+corresponding\s+leaving\s+groups\s+for\s+this\s+reaction\s+are\s+"
    r"deduced\s+to\s+be"
    r")\s+)"
    r"(?P<value>\[[^\r\n]*\])(?P<suffix>)",
    re.IGNORECASE,
)
INFERRED_REACTANTS_RE = re.compile(
    r"(?P<prefix>\bthe\s+final\s+reactant(?:\(s\)|s)?\s+are\s+inferred\s+as:"
    r"[ \t]*)(?P<value>\S+?)(?P<suffix>(?=</think>|\r?\n|\s*$))",
    re.IGNORECASE,
)


def replace_valid_smiles(
    text: str,
    pattern: re.Pattern[str],
    formatter: Callable[[str], str | None],
) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        formatted = formatter(match.group("value"))
        if formatted is None:
            return match.group(0)
        count += 1
        return f"{match.group('prefix')}{formatted}{match.group('suffix')}"

    return pattern.sub(replace, text), count


def format_smiles_list(value: str) -> str | None:
    try:
        items = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        return None
    output = []
    changed = False
    for item in items:
        formatted = try_to_atomwise_smiles(item)
        if formatted is None:
            output.append(item)
        else:
            output.append(formatted)
            changed = changed or formatted != item
    return json.dumps(output, ensure_ascii=False) if changed else None


def transform_message_content(text: str) -> TransformResult:
    if not isinstance(text, str) or not text:
        return TransformResult(text=text, transformed_spans=0)
    total = 0
    transformations = (
        (SYNTHON_LIST_RE, format_smiles_list),
        (INLINE_SMILES_LIST_RE, format_smiles_list),
        (SENTENCE_SMILES_RE, try_to_atomwise_smiles),
        (LINE_SMILES_RE, try_to_atomwise_smiles),
        (INFERRED_REACTANTS_RE, try_to_atomwise_smiles),
    )
    for pattern, formatter in transformations:
        text, count = replace_valid_smiles(text, pattern, formatter)
        total += count

    def replace_answer(match: re.Match[str]) -> str:
        nonlocal total
        formatted = try_to_atomwise_smiles(match.group(2).strip())
        if formatted is None:
            return match.group(0)
        total += 1
        return f"{match.group(1)}{formatted}{match.group(3)}"

    return TransformResult(ANSWER_RE.sub(replace_answer, text), total)


def is_standalone_s2s_prompt(messages: Iterable[dict]) -> bool:
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            normalized = content.lower()
            if all(marker in normalized for marker in S2S_SYSTEM_MARKERS):
                return True
    return False


def transform_standalone_smiles(text: str) -> TransformResult:
    if not isinstance(text, str) or not text.strip():
        return TransformResult(text=text, transformed_spans=0)
    formatted = try_to_atomwise_smiles(text.strip())
    if formatted is None:
        return TransformResult(text=text, transformed_spans=0)
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    return TransformResult(f"{text[:start]}{formatted}{text[end:]}", 1)


def transform_messages(messages: Iterable[dict]) -> tuple[list[dict], int]:
    messages = list(messages)
    standalone_s2s = is_standalone_s2s_prompt(messages)
    output = []
    total = 0
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, str):
            result = transform_message_content(content)
            if (
                standalone_s2s
                and copied.get("role") in {"user", "assistant"}
                and result.transformed_spans == 0
            ):
                result = transform_standalone_smiles(result.text)
            copied["content"] = result.text
            total += result.transformed_spans
        output.append(copied)
    return output, total
