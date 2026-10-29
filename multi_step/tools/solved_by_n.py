#!/usr/bin/env python3
"""Solve rate at N=50..500 for DianShi-RetroPlanSystem-v0 / RetroAgent / Retro-R1 results.

Usage:
    python solved_by_n.py RESULT_FILE
    python solved_by_n.py RESULT_FILE --framework bench|retroagent|r1

Supported inputs:
  - DianShi-RetroPlanSystem-v0 JSONL (``--framework bench``): one line per target, with
    ``metrics.single_step_calls`` and ``success``.
  - Retro-R1 JSONL (result.jsonl / result_ordered.jsonl): one line per
    target, with ``onestep_calls`` and ``success``.
  - RetroAgent summary.json: ``per_experiment`` entries with
    ``solved`` / ``solved_at_budget`` (uses the first experiment).

The script only computes metrics and warns about results that look like
infrastructure failures (endpoint / timeout / service exhaustion).
It does not check dataset completeness or scan logs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

NS = (50, 100, 200, 300, 400, 500)

_INFRA_RE = re.compile(
    r"(\bexhausted\b|\btimeout\b|timed\s*out|\bunreachable\b|\bconnection\b|"
    r"http\s*5\d\d|\berror\b|\bexception\b|\bcancelled\b|invalid\s+smiles)",
    re.IGNORECASE,
)


def _load(path: Path) -> tuple[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "per_experiment" in data:
        experiments = data.get("per_experiment") or []
        if not experiments:
            raise SystemExit(f"no per_experiment entries in {path}")
        return "retroagent", list(experiments[0])
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no records in {path}")
    sample = rows[0]
    metrics = sample.get("metrics") if isinstance(sample.get("metrics"), dict) else {}
    if "single_step_calls" in metrics:
        return "bench", rows
    if "onestep_calls" in sample:
        return "r1", rows
    if "solved_at_budget" in sample or "solved" in sample:
        return "retroagent", rows
    raise SystemExit(f"unrecognized result schema in {path}")


def _record(kind: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    if kind == "bench":
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        budget = metrics.get("single_step_calls")
        solved = bool(row.get("success"))
        reason = str(row.get("termination_reason") or "")
        target = row.get("idx", index)
        abnormal = None
        if budget is None:
            abnormal = "missing_budget"
        elif _INFRA_RE.search(reason):
            abnormal = reason or "infrastructure_error"
        elif not solved and int(budget) == 0:
            abnormal = "zero_single_step_calls"
        return {"target": target, "solved": solved, "budget": budget, "abnormal": abnormal}

    if kind == "r1":
        budget = row.get("onestep_calls")
        solved = bool(row.get("success"))
        turns = row.get("turns")
        elapsed = row.get("elapsed_s")
        target = row.get("idx", index)
        abnormal = None
        if budget is None:
            abnormal = "missing_budget"
        elif not solved and (int(budget) == 0 or int(turns or 0) == 0):
            abnormal = f"zero_calls(turns={turns}, onestep_calls={budget})"
        elif not solved and isinstance(elapsed, (int, float)) and elapsed < 1:
            abnormal = f"near_zero_elapsed({elapsed}s)"
        return {"target": target, "solved": solved, "budget": budget, "abnormal": abnormal}

    # RetroAgent summary entries
    solved = bool(row.get("solved"))
    budget = row.get("solved_at_budget")
    target = row.get("target_index", index)
    abnormal = None
    if solved and budget is None:
        abnormal = "solved_without_budget"
    elif not solved and (int(row.get("num_rollouts") or 0) == 0 or int(row.get("total_budget") or 0) == 0):
        abnormal = (
            f"zero_budget(num_rollouts={row.get('num_rollouts')}, "
            f"total_budget={row.get('total_budget')})"
        )
    return {"target": target, "solved": solved, "budget": budget, "abnormal": abnormal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="result .jsonl or RetroAgent summary.json")
    parser.add_argument(
        "--framework",
        choices=("auto", "bench", "retroagent", "r1"),
        default="auto",
        help="input schema (default: auto-detect)",
    )
    args = parser.parse_args()

    kind, rows = _load(args.result)
    if args.framework != "auto":
        kind = args.framework
    records = [_record(kind, row, i) for i, row in enumerate(rows)]
    total = len(records)

    budget_name = {
        "bench": "metrics.single_step_calls",
        "r1": "onestep_calls",
        "retroagent": "solved_at_budget",
    }[kind]
    print(f"framework: {kind}")
    print(f"budget:    {budget_name}")
    print(f"targets:   {total}")
    print()
    print(f"{'N':>5}  {'solved':>7}  {'solve_rate':>10}")
    for n in NS:
        solved = sum(
            1
            for record in records
            if record["solved"]
            and isinstance(record["budget"], (int, float))
            and record["budget"] <= n
        )
        print(f"{n:>5}  {solved:>7}  {100.0 * solved / total:>9.2f}%")

    abnormal = [record for record in records if record["abnormal"]]
    print()
    if abnormal:
        print(
            f"WARNING: {len(abnormal)} abnormal result(s) detected "
            "(possible infrastructure failure; rerun these targets):"
        )
        for record in abnormal[:20]:
            print(f"  target {record['target']}: {record['abnormal']}")
        if len(abnormal) > 20:
            print(f"  ... and {len(abnormal) - 20} more")
        print()
        print("Abnormal results are counted as unsolved in the rates above.")
    else:
        print("No abnormal results detected.")


if __name__ == "__main__":
    main()
