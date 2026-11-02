#!/usr/bin/env python3
"""Build the intermediate-label evaluation datasets under ``model_tests``.

The directory and file names intentionally match the established
``model_tests`` layout.  All repository paths are derived from this file, so
the project can be moved or downloaded without editing paths in the code.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dianshi_retrosteplm.data.smiles_atomwise import transform_messages


PACKAGE_DIR = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_DIR.parents[1]
PROMPT_DIR = PACKAGE_DIR / "prompts"
OUT_ROOT = ROOT / "model_tests"

PATENT_DIR = ROOT / "datasets/DianShi-PatentRxnStep-Dataset"
SOURCE_SPECS = (
    ("test_2001_2024", "test", PATENT_DIR / "DianShi-PatentRxnStep-Test-2001-2024.jsonl"),
    ("test_2001_2024", "val", PATENT_DIR / "DianShi-PatentRxnStep-Val-2001-2024.jsonl"),
    ("test_2025", "test", PATENT_DIR / "DianShi-PatentRxnStep-Test-2025.jsonl"),
    ("test_2025", "val", PATENT_DIR / "DianShi-PatentRxnStep-Val-2025.jsonl"),
    ("uspto_50k", "test", ROOT / "datasets/USPTO-50K/USPTO_50K_Test.jsonl"),
    ("uspto_50k", "val", ROOT / "datasets/USPTO-50K/USPTO_50K_Val.jsonl"),
)

PROMPTS = {
    "intermediate_label": PROMPT_DIR / "prompt_sft_intermediate_label.txt",
}

SPLIT_PREFIX = {"test": "testset", "val": "validationset"}
FORMAT_KEY = re.compile(r"\{(\w+)")


@dataclass(frozen=True)
class Variant:
    key: str
    subfolder: str
    suffix: str
    prompt_key: str


@dataclass(frozen=True)
class Unit:
    dataset: str
    split: str
    source: Path
    variant: Variant

    @property
    def name(self) -> str:
        return f"{SPLIT_PREFIX[self.split]}_{self.variant.suffix}"

    @property
    def base(self) -> Path:
        return OUT_ROOT / self.dataset / self.variant.subfolder

    @property
    def jsonl_dir(self) -> Path:
        return self.base / self.name

    @property
    def cached_dir(self) -> Path:
        return self.base / f"{self.name}_new_cached"

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.jsonl_dir, self.cached_dir)

    @property
    def prompt(self) -> Path:
        return PROMPTS[self.variant.prompt_key]

    def __str__(self) -> str:
        return f"{self.dataset}/{self.variant.subfolder}/{self.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build intermediate-label model-test JSONL and Hugging Face cached datasets."
        )
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names: test_2001_2024,test_2025,uspto_50k.",
    )
    parser.add_argument(
        "--splits", default="", help="Comma-separated splits: test,val."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the build plan only.")
    return parser.parse_args()


def csv_values(value: str) -> list[str] | None:
    values = [part.strip() for part in value.split(",") if part.strip()]
    return values or None


def display_path(path: Path) -> str:
    """Prefer a repository-relative path while remaining safe for test paths."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def make_variants() -> dict[str, Variant]:
    key = "intermediate_label"
    return {key: Variant(key, key, key, key)}


def discover_sources() -> dict[str, dict[str, Path]]:
    sources: dict[str, dict[str, Path]] = {}
    missing = []
    for dataset, split, path in SOURCE_SPECS:
        if path.is_file():
            sources.setdefault(dataset, {})[split] = path
        else:
            missing.append(path.relative_to(ROOT))
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required source datasets are missing: {joined}")
    return sources


def make_units(
    sources: dict[str, dict[str, Path]],
    variants: dict[str, Variant],
    datasets_filter: list[str] | None,
    splits_filter: list[str] | None,
) -> list[Unit]:
    unknown_datasets = sorted(set(datasets_filter or ()) - set(sources))
    unknown_splits = sorted(set(splits_filter or ()) - set(SPLIT_PREFIX))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    if unknown_splits:
        raise ValueError(f"Unknown splits: {unknown_splits}")

    units = []
    for dataset, by_split in sources.items():
        if datasets_filter and dataset not in datasets_filter:
            continue
        for split, source in by_split.items():
            if splits_filter and split not in splits_filter:
                continue
            for variant in variants.values():
                units.append(Unit(dataset, split, source, variant))
    return units


def record_payload(record: dict) -> dict:
    payload = record.get("preprocess_for_train")
    if isinstance(payload, dict) and payload:
        return payload
    keys = (
        "hash_id",
        "reactants",
        "products",
        "localtemp_res",
        "rdchiral_res",
        "synthons_smiles",
        "leaving_groups",
        "product_bond",
        "reactant_fragments",
        "rxn_insight_type",
        "rxn_smiles_modified",
    )
    return {key: record[key] for key in keys if key in record}


def joined(value) -> str:
    if isinstance(value, list):
        return ".".join(str(item) for item in value)
    return str(value or "").strip()


def get_reaction_template(value: dict) -> str:
    local = value.get("localtemp_res") if isinstance(value, dict) else None
    return str(local.get("rxn_temp", "")) if isinstance(local, dict) else ""


def load_prompt(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    sections = {"system": "", "user": ""}
    current = None
    buffers: dict[str, list[str]] = {"system": [], "user": []}
    for line in text.splitlines():
        marker = line.strip()
        if marker == "---SYSTEM---":
            current = "system"
        elif marker == "---USER---":
            current = "user"
        elif marker == "---ASSISTANT---":
            current = None
        elif current is not None:
            buffers[current].append(line)
    for key in sections:
        sections[key] = "\n".join(buffers[key]).strip()
    if not sections["system"] or not sections["user"]:
        raise ValueError(f"Prompt must contain SYSTEM and USER sections: {path}")
    return sections


def template_values(payload: dict) -> dict[str, str]:
    products = joined(payload.get("products"))
    reactants = joined(payload.get("reactants"))
    if not products or not reactants:
        raise ValueError("The record is missing products or reactants.")
    synthons = payload.get("synthons_smiles", [])
    leaving_groups = payload.get("leaving_groups", [])
    if not isinstance(synthons, list):
        synthons = []
    if not isinstance(leaving_groups, list):
        leaving_groups = []
    values = {
        "products": products,
        "reactants": reactants,
        "product_smiles": products,
        "reactant_smiles": reactants,
        "synthons_smiles": json.dumps(synthons, ensure_ascii=False),
        "leaving_groups": json.dumps(leaving_groups, ensure_ascii=False),
        "synthons_str": json.dumps(synthons, ensure_ascii=False),
        "leaving_groups_str": json.dumps(leaving_groups, ensure_ascii=False),
        "reaction_template": get_reaction_template(payload),
        "product_bond_str": json.dumps(payload.get("product_bond", []), ensure_ascii=False),
        "reactant_fragments_str": json.dumps(payload.get("reactant_fragments", []), ensure_ascii=False),
        "error_res": str(payload.get("error_res", "")),
        "rxn_insight_type": str(payload.get("rxn_insight_type", "")),
        "products_i": "",
        "reactants_i": "",
        "synthons_smiles_i": "",
        "leaving_groups_i": "",
        "similarity_i": "",
    }
    return values


def convert_unit(unit: Unit) -> int:
    prompt = load_prompt(unit.prompt)
    required = set().union(*(FORMAT_KEY.findall(value) for value in prompt.values()))

    unit.jsonl_dir.mkdir(parents=True, exist_ok=True)
    output = unit.jsonl_dir / "all.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    success = spans = 0
    line_number = 0
    try:
        with unit.source.open("r", encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                payload = record_payload(record)
                values = template_values(payload)
                missing = required - set(values)
                if missing:
                    raise KeyError(f"Prompt keys are unavailable: {sorted(missing)}")
                messages = [
                    {"role": "system", "content": prompt["system"].format(**values)},
                    {"role": "user", "content": prompt["user"].format(**values)},
                ]
                messages, count = transform_messages(messages)
                sample = {
                    "messages": messages,
                    "metadata": {
                        "preprocess_for_train": payload,
                        "meta_info": record.get("meta_info", {}),
                    },
                    "path": unit.source.relative_to(ROOT).as_posix(),
                }
                destination.write(json.dumps(sample, ensure_ascii=False) + "\n")
                success += 1
                spans += count
        temporary.replace(output)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed while converting {unit} near input line {line_number}: {exc}") from exc
    if success == 0:
        raise RuntimeError(f"No records were generated for {unit}.")
    print(
        f"  [convert] {success} records, {spans} atom-wise SMILES spans "
        f"-> {display_path(output)}"
    )
    return success


def cache_jsonl(source_dir: Path, destination: Path) -> None:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Install the dependencies in requirements.txt first.") from exc
    if not hasattr(datasets, "Dataset"):
        raise RuntimeError(
            "The Hugging Face 'datasets' package is not installed. "
            "Install the dependencies in requirements.txt first."
        )

    records = []
    for path in sorted(source_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                messages = data.get("messages", [])
                if not messages:
                    continue
                metadata = data.get("metadata", {})
                payload = metadata.get("preprocess_for_train", metadata)
                records.append(
                    {
                        "messages": messages,
                        "path": data.get("path", display_path(path)),
                        "meta_info": {
                            "product": joined(payload.get("products")),
                            "reactant": joined(payload.get("reactants")),
                            "hash_id": str(payload.get("hash_id", metadata.get("hash_id", ""))),
                        },
                        "length": len(messages),
                    }
                )
    if not records:
        raise RuntimeError(f"No valid records found in {source_dir}")
    if destination.exists():
        shutil.rmtree(destination)
    dataset = datasets.Dataset.from_list(records)
    dataset.save_to_disk(str(destination))
    print(f"  [cache] {len(dataset)} records -> {display_path(destination)}")


def main() -> int:
    args = parse_args()
    try:
        variants = make_variants()
        for prompt in PROMPTS.values():
            if not prompt.is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt}")
        sources = discover_sources()
        units = make_units(
            sources,
            variants,
            csv_values(args.datasets),
            csv_values(args.splits),
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"[plan] Variants: {list(variants)}")
    print(f"[plan] Units: {len(units)}")
    for unit in units:
        done = all(path.exists() for path in unit.outputs)
        status = "existing" if done and not args.force else "build"
        print(f"    [{status}] {unit}")
    if args.dry_run:
        return 0

    pending = [
        unit for unit in units if args.force or not all(path.exists() for path in unit.outputs)
    ]
    if not pending:
        print("[main] Every requested output already exists.")
        return 0

    if args.force:
        for unit in pending:
            for path in unit.outputs:
                if path.exists():
                    shutil.rmtree(path)

    try:
        completed = 0
        started = time.time()
        for index, unit in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {unit}")
            convert_unit(unit)
            cache_jsonl(unit.jsonl_dir, unit.cached_dir)
            completed += 1
        print(f"[main] Completed {completed} unit(s) in {time.time() - started:.1f}s.")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
