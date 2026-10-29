#!/usr/bin/env python3
"""SMILES atom-wise formatting for mixed natural-language/SMILES messages.

This module deliberately does not replace Qwen3's tokenizer.  It rewrites only
well-defined SMILES spans to a space-separated atom-wise representation; the
result is then encoded by the unmodified Qwen3 tokenizer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable


# Kept compatible with the regex used in
# strategy_todo/smiles_atomwise_tokenization_comparison.html.
# The alternatives are ordered from longest/most specific to shortest.
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
    """Raised when the atom-wise regex cannot consume the complete string."""


def atomwise_tokens(smiles: str) -> list[str]:
    """Split a SMILES/reaction-SMILES string with full-coverage validation."""
    if not isinstance(smiles, str):
        raise TypeError(f"smiles must be str, got {type(smiles).__name__}")
    if not smiles:
        raise SmilesTokenizationError("empty SMILES")

    tokens: list[str] = []
    cursor = 0
    for match in SMILES_ATOM_RE.finditer(smiles):
        if match.start() != cursor:
            bad = smiles[cursor : match.start()]
            raise SmilesTokenizationError(
                f"unmatched text at offset {cursor}: {bad!r} in {smiles!r}"
            )
        tokens.append(match.group(0))
        cursor = match.end()
    if cursor != len(smiles):
        raise SmilesTokenizationError(
            f"unmatched text at offset {cursor}: {smiles[cursor:]!r} in {smiles!r}"
        )
    if not tokens:
        raise SmilesTokenizationError(f"no SMILES token found in {smiles!r}")
    return tokens


def to_atomwise_smiles(smiles: str) -> str:
    """Return canonical single-space atom-wise text.

    The function is idempotent: an already formatted atom-wise string is
    accepted and returned in canonical spacing.
    """
    stripped = smiles.strip()
    try:
        return " ".join(atomwise_tokens(stripped))
    except SmilesTokenizationError:
        if not any(ch.isspace() for ch in stripped):
            raise
        compact = "".join(stripped.split())
        tokens = atomwise_tokens(compact)
        formatted = " ".join(tokens)
        # Only accept whitespace as preprocessing when every compacted
        # character forms a valid SMILES token sequence.
        return formatted


def from_atomwise_smiles(spaced_smiles: str) -> str:
    """Remove atom-wise spaces and validate that the compact text is tokenizable."""
    compact = "".join(spaced_smiles.split())
    atomwise_tokens(compact)
    return compact


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

# Current training/test templates use sentences such as:
# "The product SMILES is ... . Below are similar reactions ..."
SENTENCE_SMILES_RE = re.compile(
    r"(?P<prefix>\bThe\s+(?:product|reactants?)\s+SMILES\s+is\s+)"
    r"(?P<value>\S+?)"
    r"(?P<suffix>\.(?=\s+(?:Below|Based|Using|Please|Predict|Given|The|By)\b)|"
    r"(?=\s*$))",
    re.IGNORECASE,
)

# RAG example lines.  Deliberately excludes reaction_template/SMARTS.
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

# Chain-of-thought templates also embed SMILES in prose instead of dedicated
# fields.  Keep these rules deliberately tied to the known template wording so
# ordinary JSON lists and natural-language text are left untouched.
INLINE_SMILES_LIST_RE = re.compile(
    r"(?P<prefix>\b(?:"
    r"the\s+synthons\s+involved\s+in\s+this\s+reaction\s+are\s+identified\s+as|"
    r"the\s+corresponding\s+leaving\s+groups\s+for\s+this\s+reaction\s+are\s+"
    r"deduced\s+to\s+be"
    r")\s+)"
    r"(?P<value>\[[^\r\n]*\])"
    r"(?P<suffix>)",
    re.IGNORECASE,
)

INFERRED_REACTANTS_RE = re.compile(
    r"(?P<prefix>\bthe\s+final\s+reactant(?:\(s\)|s)?\s+are\s+inferred\s+as:"
    r"[ \t]*)"
    r"(?P<value>\S+?)"
    r"(?P<suffix>(?=</think>|\r?\n|\s*$))",
    re.IGNORECASE,
)


def _replace_valid_smiles(
    text: str,
    pattern: re.Pattern[str],
    formatter: Callable[[str], str | None],
) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        formatted = formatter(match.group("value"))
        if formatted is None:
            return match.group(0)
        count += 1
        return f"{match.group('prefix')}{formatted}{match.group('suffix')}"

    return pattern.sub(repl, text), count


def _format_synthon_list(value: str) -> str | None:
    try:
        items = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        return None

    changed = False
    output: list[str] = []
    for item in items:
        formatted = try_to_atomwise_smiles(item)
        if formatted is None:
            output.append(item)
        else:
            output.append(formatted)
            changed = changed or formatted != item
    if not changed:
        return None
    return json.dumps(output, ensure_ascii=False)


def transform_message_content(text: str) -> TransformResult:
    """Atom-wise-format known SMILES spans while preserving all other text."""
    if not isinstance(text, str) or not text:
        return TransformResult(text=text, transformed_spans=0)

    total = 0

    # JSON lists must be handled before generic line rules.
    text, n = _replace_valid_smiles(text, SYNTHON_LIST_RE, _format_synthon_list)
    total += n
    text, n = _replace_valid_smiles(
        text, INLINE_SMILES_LIST_RE, _format_synthon_list
    )
    total += n
    text, n = _replace_valid_smiles(text, SENTENCE_SMILES_RE, try_to_atomwise_smiles)
    total += n
    text, n = _replace_valid_smiles(text, LINE_SMILES_RE, try_to_atomwise_smiles)
    total += n
    text, n = _replace_valid_smiles(
        text, INFERRED_REACTANTS_RE, try_to_atomwise_smiles
    )
    total += n

    def answer_repl(match: re.Match[str]) -> str:
        nonlocal total
        formatted = try_to_atomwise_smiles(match.group(2).strip())
        if formatted is None:
            return match.group(0)
        total += 1
        return f"{match.group(1)}{formatted}{match.group(3)}"

    text = ANSWER_RE.sub(answer_repl, text)
    return TransformResult(text=text, transformed_spans=total)


def transform_messages(messages: Iterable[dict]) -> tuple[list[dict], int]:
    """Return copied messages with only recognized SMILES spans transformed."""
    output: list[dict] = []
    total = 0
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, str):
            result = transform_message_content(content)
            copied["content"] = result.text
            total += result.transformed_spans
        output.append(copied)
    return output, total


def restore_answer_spaces(text: str) -> TransformResult:
    """Compact atom-wise SMILES inside <answer> tags after generation."""
    if not isinstance(text, str) or not text:
        return TransformResult(text=text, transformed_spans=0)

    total = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal total
        try:
            compact = from_atomwise_smiles(match.group(2).strip())
        except SmilesTokenizationError:
            return match.group(0)
        total += 1
        return f"{match.group(1)}{compact}{match.group(3)}"

    return TransformResult(text=ANSWER_RE.sub(repl, text), transformed_spans=total)
