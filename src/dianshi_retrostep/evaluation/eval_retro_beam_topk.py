#!/usr/bin/env python3
"""Evaluate exact-match Top-k accuracy from retrosynthesis beam output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def canonicalize_reactants(smiles: Any) -> Optional[str]:
    """Canonicalize a dot-separated reactant mixture independently of order."""
    if not isinstance(smiles, str):
        return None
    smiles = re.sub(r"\s+", "", smiles)
    if not smiles:
        return None
    canonical_parts = []
    for part in smiles.split("."):
        if not part:
            return None
        molecule = Chem.MolFromSmiles(part)
        if molecule is None:
            return None
        canonical_parts.append(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
    return ".".join(sorted(canonical_parts))


def answer_tag_candidate(text: Any) -> Optional[str]:
    """Return the last non-empty complete answer-tag value."""
    if not isinstance(text, str):
        return None
    for match in reversed(ANSWER_RE.findall(text)):
        answer = re.sub(r"\s+", "", match)
        if answer:
            return answer
    return None


def candidate_fragments(text: str) -> list[str]:
    """Extract likely answer fragments from markdown and plain text."""
    fragments = []
    fragments.extend(
        match.strip()
        for match in re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    )
    fragments.extend(
        match.strip() for match in re.findall(r"\*\*(.*?)\*\*", text, re.DOTALL)
    )
    fragments.extend(match.strip() for match in re.findall(r"`([^`]+)`", text))
    fragments.append(text.strip())
    return fragments


def canonicalize_generated_text(text: Any) -> Optional[str]:
    """Parse tagged, reasoning, markdown, or bare-SMILES model output."""
    if not isinstance(text, str):
        return None
    tagged = answer_tag_candidate(text)
    if tagged is not None:
        canonical = canonicalize_reactants(tagged)
        if canonical is not None:
            return canonical

    cleaned = text.split("<|im_end|>")[0]
    cleaned = re.sub(r"<\|[^>]*\|>", "", cleaned)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]

    for fragment in reversed(candidate_fragments(cleaned)):
        fragment = re.sub(r"[*`]", "", fragment).strip()
        candidates = [part.strip() for part in re.split(r"[\n;]", fragment)]
        for candidate in reversed(candidates):
            if not candidate:
                continue
            candidate = re.sub(
                r"^.*?(?:answer|reactants?|smiles)\s*:\s*",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
            canonical = canonicalize_reactants(candidate)
            if canonical is not None:
                return canonical
        canonical = canonicalize_reactants(fragment)
        if canonical is not None:
            return canonical
    return None


def assistant_beams(messages: Any) -> Optional[list[Any]]:
    """Find the final assistant message containing ranked beam results."""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        return content if isinstance(content, list) else None
    return None


def beam_text(beam: Any) -> Optional[str]:
    """Read beam text without changing its original rank."""
    if isinstance(beam, str):
        return beam
    if isinstance(beam, (list, tuple)) and beam and isinstance(beam[0], str):
        return beam[0]
    if isinstance(beam, dict):
        for key in ("text", "content", "response"):
            if isinstance(beam.get(key), str):
                return beam[key]
    return None


def evaluate(path: Path, ks: Sequence[int]) -> dict[str, Any]:
    max_k = max(ks)
    correct = {k: 0 for k in ks}
    stats = {
        "nonempty_lines": 0,
        "json_errors": 0,
        "evaluated_samples": 0,
        "invalid_ground_truth": 0,
        "missing_assistant_beams": 0,
        "unparseable_predictions": 0,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            stats["nonempty_lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["json_errors"] += 1
                print(
                    f"Warning: line {line_number} contains invalid JSON: {exc}",
                    file=sys.stderr,
                )
                continue

            ground_truth = canonicalize_reactants(row.get("reactant"))
            if ground_truth is None:
                stats["invalid_ground_truth"] += 1
                continue
            stats["evaluated_samples"] += 1

            beams = assistant_beams(row.get("messages"))
            predictions: list[Optional[str]] = []
            if beams is None:
                stats["missing_assistant_beams"] += 1
            else:
                for beam in beams[:max_k]:
                    prediction = canonicalize_generated_text(beam_text(beam))
                    if prediction is None:
                        stats["unparseable_predictions"] += 1
                    predictions.append(prediction)

            for k in ks:
                if ground_truth in predictions[:k]:
                    correct[k] += 1

    total = stats["evaluated_samples"]
    return {
        "input_file": str(path),
        **stats,
        "topk_correct": {str(k): correct[k] for k in ks},
        "topk_accuracy": {
            str(k): correct[k] / total if total else 0.0 for k in ks
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[1, 3, 5, 10], help="Top-k values."
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not args.ks or any(k < 1 for k in args.ks):
        parser.error("--ks must contain positive integers")
    args.ks = sorted(set(args.ks))
    if not args.input_jsonl.is_file():
        parser.error(f"input file does not exist: {args.input_jsonl}")
    return args


def main() -> int:
    args = parse_args()
    result = evaluate(args.input_jsonl, args.ks)
    total = result["evaluated_samples"]
    print(f"Input: {result['input_file']}")
    print(f"Evaluated samples: {total}")
    for k in args.ks:
        correct = result["topk_correct"][str(k)]
        accuracy = result["topk_accuracy"][str(k)]
        print(f"Top-{k:<2}: {accuracy * 100:.2f}% ({correct}/{total})")
    print(
        "Diagnostics: "
        f"json_errors={result['json_errors']}, "
        f"invalid_ground_truth={result['invalid_ground_truth']}, "
        f"missing_assistant_beams={result['missing_assistant_beams']}, "
        f"unparseable_predictions={result['unparseable_predictions']}"
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Metrics JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
