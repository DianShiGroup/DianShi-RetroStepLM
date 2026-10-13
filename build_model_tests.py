#!/usr/bin/env python3
"""Build all local evaluation datasets under ``model_tests``.

The directory and file names intentionally match the established NMI
``model_tests`` layout.  All repository paths are derived from this file, so
the project can be moved or downloaded without editing paths in the code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from smiles_atomwise import transform_messages


ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts" / "nmi"
OUT_ROOT = ROOT / "model_tests"
RAG_CACHE = ROOT / "rag_cache"
RAG_SCRIPT = ROOT / "insert_rag_into_test.py"

SOURCE_SPECS = (
    ("test_in", "test", ROOT / "datasets/DianShi-RxnStep-dataset/DianShi-RxnStep-Test-IN.jsonl"),
    ("test_in", "val", ROOT / "datasets/DianShi-RxnStep-dataset/DianShi-RxnStep-Val-IN.jsonl"),
    ("test_out", "test", ROOT / "datasets/DianShi-RxnStep-dataset/DianShi-RxnStep-Test-OUT.jsonl"),
    ("test_out", "val", ROOT / "datasets/DianShi-RxnStep-dataset/DianShi-RxnStep-Val-OUT.jsonl"),
    ("uspto_50k", "test", ROOT / "datasets/USPTO-50K/raw_test.jsonl"),
    ("uspto_50k", "val", ROOT / "datasets/USPTO-50K/raw_val.jsonl"),
)

PROMPTS = {
    "s2s": PROMPT_DIR / "prompt_sft_s2s.txt",
    "s2s_rag": PROMPT_DIR / "prompt_enhancer_s2s_rag.txt",
    "sl_cot": PROMPT_DIR / "prompt_sft_sl_cot.txt",
    "sl_cot_rag": PROMPT_DIR / "prompt_sft_sl_cot_rag.txt",
}

SPLIT_PREFIX = {"test": "testset", "val": "validationset"}
SAFE_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")
FORMAT_KEY = re.compile(r"\{(\w+)")


@dataclass(frozen=True)
class Database:
    name: str
    path: Path


@dataclass(frozen=True)
class Variant:
    key: str
    subfolder: str
    suffix: str
    prompt_key: str
    db: Database | None = None


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
    def atomwise_dir(self) -> Path:
        return self.base / f"{self.name}_atomwise"

    @property
    def cached_dir(self) -> Path:
        return self.base / f"{self.name}_new_cached"

    @property
    def atomwise_cached_dir(self) -> Path:
        return self.base / f"{self.name}_atomwise_new_cached"

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.jsonl_dir,
            self.atomwise_dir,
            self.cached_dir,
            self.atomwise_cached_dir,
        )

    @property
    def prompt(self) -> Path:
        return PROMPTS[self.variant.prompt_key]

    @property
    def rag_input(self) -> Path:
        if self.variant.db is None:
            return self.source
        return (
            RAG_CACHE
            / self.variant.db.name
            / f"{self.dataset}__{self.split}_with_rag.jsonl"
        )

    def __str__(self) -> str:
        return f"{self.dataset}/{self.variant.subfolder}/{self.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NMI model-test JSONL and Hugging Face cached datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_test_config.json"),
        help="Configuration file, relative to the repository root by default.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names: test_in,test_out,uspto_50k.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant keys. The default builds every variant.",
    )
    parser.add_argument(
        "--splits", default="", help="Comma-separated splits: test,val."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the build plan only.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retrieve-topk", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=0)
    parser.add_argument("--faiss-candidates", type=int, default=200)
    parser.add_argument("--prompt-rag-topk", type=int, default=5)
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Reuse existing files under rag_cache instead of running retrieval.",
    )
    return parser.parse_args()


def csv_values(value: str) -> list[str] | None:
    values = [part.strip() for part in value.split(",") if part.strip()]
    return values or None


def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    """Prefer a repository-relative path while remaining safe for test paths."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_databases(config_path: Path) -> list[Database]:
    path = resolve_from_root(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if "db" not in config:
        raise ValueError("The required 'db' field is missing from the configuration.")
    entries = config["db"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "The required 'db' field must be a non-empty list. "
            "See README.md for the supported format."
        )

    databases: list[Database] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"db[{index}] must be an object with name and path fields.")
        name = entry.get("name")
        raw_path = entry.get("path")
        if not isinstance(name, str) or not SAFE_DB_NAME.fullmatch(name):
            raise ValueError(
                f"db[{index}].name must contain only ASCII letters, digits, and underscores."
            )
        if name in seen:
            raise ValueError(f"Duplicate database name: {name}")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"db[{index}].path is required.")
        configured_path = Path(raw_path)
        if configured_path.is_absolute():
            raise ValueError(
                f"db[{index}].path must be relative to the repository root."
            )
        db_path = resolve_from_root(configured_path)
        if not db_path.is_dir():
            raise FileNotFoundError(f"Database directory not found: {db_path}")
        required_files = (
            "config.json",
            "metadata.jsonl",
            "product_fps.npy",
            "line_offsets.npy",
            "faiss_product.index",
        )
        missing_files = [name for name in required_files if not (db_path / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Database {name!r} is missing required files: {missing_files}"
            )
        databases.append(Database(name=name, path=db_path))
        seen.add(name)
    return databases


def make_variants(databases: list[Database]) -> dict[str, Variant]:
    variants = {
        "s2s": Variant("s2s", "s2s", "s2s", "s2s"),
        "sl_cot": Variant("sl_cot", "sl_cot", "sl_cot", "sl_cot"),
    }
    for db in databases:
        s2s_key = f"s2s_rag_{db.name}"
        cot_key = f"sl_cot_rag_{db.name}"
        variants[s2s_key] = Variant(
            s2s_key, "s2s", f"s2s_{db.name}_rag", "s2s_rag", db
        )
        variants[cot_key] = Variant(
            cot_key, "sl_cot_rag", f"sl_cot_{db.name}_rag", "sl_cot_rag", db
        )
    return variants


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
    variants_filter: list[str] | None,
    splits_filter: list[str] | None,
) -> list[Unit]:
    unknown_datasets = sorted(set(datasets_filter or ()) - set(sources))
    unknown_variants = sorted(set(variants_filter or ()) - set(variants))
    unknown_splits = sorted(set(splits_filter or ()) - set(SPLIT_PREFIX))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    if unknown_splits:
        raise ValueError(f"Unknown splits: {unknown_splits}")

    units = []
    for dataset, by_split in sources.items():
        if datasets_filter and dataset not in datasets_filter:
            continue
        for split, source in by_split.items():
            if splits_filter and split not in splits_filter:
                continue
            for key, variant in variants.items():
                if variants_filter and key not in variants_filter:
                    continue
                units.append(Unit(dataset, split, source, variant))
    return units


def relative_symlink(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def run_rag(
    units: list[Unit],
    num_workers: int,
    retrieve_topk: int,
    bottom_k: int,
    faiss_candidates: int,
    force: bool,
) -> None:
    databases = {unit.variant.db for unit in units if unit.variant.db is not None}
    for db in sorted(databases, key=lambda item: item.name):
        group = [unit for unit in units if unit.variant.db == db]
        cache_dir = RAG_CACHE / db.name
        cache_dir.mkdir(parents=True, exist_ok=True)

        unique_sources = {(unit.dataset, unit.split): unit.source for unit in group}
        pending: list[Path] = []
        for (dataset, split), source in sorted(unique_sources.items()):
            link = cache_dir / f"{dataset}__{split}.jsonl"
            output = link.with_name(f"{link.stem}_with_rag.jsonl")
            if output.exists() and not force:
                print(f"[rag:{db.name}] Reusing {output.relative_to(ROOT)}")
                continue
            relative_symlink(source, link)
            pending.append(link)

        if not pending:
            continue
        command = [
            sys.executable,
            str(RAG_SCRIPT),
            "--input-jsonl",
            ",".join(str(path) for path in pending),
            "--db-dir",
            str(db.path),
            "--num-workers",
            str(num_workers),
            "--top-k",
            str(retrieve_topk),
            "--bottom-k",
            str(bottom_k),
            "--faiss-candidates",
            str(faiss_candidates),
            "--suffix",
            "_with_rag",
        ]
        print(f"[rag:{db.name}] Retrieving for {len(pending)} source file(s).")
        subprocess.run(command, check=True, cwd=ROOT)
        for source in pending:
            output = source.with_name(f"{source.stem}_with_rag.jsonl")
            if not output.is_file():
                raise RuntimeError(f"Expected RAG output was not created: {output}")


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


def rag_examples(payload: dict, limit: int, pair_only: bool) -> str:
    examples = payload.get("prepared_trained_for_rag_sft", [])
    if not isinstance(examples, list):
        return ""
    rendered = []
    for index, example in enumerate(examples[:limit], 1):
        if not isinstance(example, dict):
            continue
        products = joined(example.get("products"))
        reactants = joined(example.get("reactants"))
        lines = [f"Example {index}", f"reactants: {reactants}", f"products: {products}"]
        if not pair_only:
            synthons = example.get("synthons_smiles", [])
            leaving_groups = example.get("leaving_groups", [])
            if not isinstance(synthons, list):
                synthons = [synthons] if synthons else []
            if not isinstance(leaving_groups, list):
                leaving_groups = [leaving_groups] if leaving_groups else []
            lines.extend(
                (
                    f"leaving_groups: {json.dumps(leaving_groups, ensure_ascii=False)}",
                    f"synthons_smiles: {json.dumps(synthons, ensure_ascii=False)}",
                )
            )
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def load_prompt(path: Path, enhancer: bool) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if enhancer:
        return {"system": text.strip(), "user": "{products}"}

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


def template_values(payload: dict, rag_topk: int, pair_only: bool) -> dict[str, str]:
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
        "rag_examples": rag_examples(payload, rag_topk, pair_only),
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


def convert_unit(unit: Unit, rag_topk: int) -> int:
    enhancer = unit.variant.prompt_key == "s2s_rag"
    prompt = load_prompt(unit.prompt, enhancer)
    required = set().union(*(FORMAT_KEY.findall(value) for value in prompt.values()))

    unit.jsonl_dir.mkdir(parents=True, exist_ok=True)
    output = unit.jsonl_dir / "all.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    success = 0
    line_number = 0
    try:
        with unit.rag_input.open("r", encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                payload = record_payload(record)
                values = template_values(payload, rag_topk, enhancer)
                missing = required - set(values)
                if missing:
                    raise KeyError(f"Prompt keys are unavailable: {sorted(missing)}")
                messages = [
                    {"role": "system", "content": prompt["system"].format(**values)},
                    {"role": "user", "content": prompt["user"].format(**values)},
                ]
                if enhancer:
                    messages.append(
                        {"role": "assistant", "content": f"<answer>{values['reactants']}</answer>"}
                    )
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
        temporary.replace(output)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed while converting {unit} near input line {line_number}: {exc}") from exc
    if success == 0:
        raise RuntimeError(f"No records were generated for {unit}.")
    print(f"  [convert] {success} records -> {display_path(output)}")
    return success


def atomwise_unit(unit: Unit) -> None:
    unit.atomwise_dir.mkdir(parents=True, exist_ok=True)
    source = unit.jsonl_dir / "all.jsonl"
    output = unit.atomwise_dir / "all.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.unlink(missing_ok=True)
    rows = spans = 0
    try:
        with source.open("r", encoding="utf-8") as src, temporary.open(
            "x", encoding="utf-8"
        ) as dst:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["messages"], count = transform_messages(record.get("messages", []))
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows += 1
                spans += count
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"  [atomwise] {rows} records, {spans} transformed SMILES spans")


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


def cache_unit(unit: Unit) -> None:
    cache_jsonl(unit.jsonl_dir, unit.cached_dir)
    cache_jsonl(unit.atomwise_dir, unit.atomwise_cached_dir)


def validate_options(args: argparse.Namespace) -> None:
    numeric = {
        "--num-workers": args.num_workers,
        "--retrieve-topk": args.retrieve_topk,
        "--faiss-candidates": args.faiss_candidates,
        "--prompt-rag-topk": args.prompt_rag_topk,
    }
    for option, value in numeric.items():
        if value < 1:
            raise ValueError(f"{option} must be at least 1.")
    if args.bottom_k < 0:
        raise ValueError("--bottom-k cannot be negative.")


def main() -> int:
    args = parse_args()
    try:
        validate_options(args)
        databases = load_databases(args.config)
        variants = make_variants(databases)
        for prompt in PROMPTS.values():
            if not prompt.is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt}")
        if not RAG_SCRIPT.is_file():
            raise FileNotFoundError(f"RAG helper not found: {RAG_SCRIPT}")
        sources = discover_sources()
        units = make_units(
            sources,
            variants,
            csv_values(args.datasets),
            csv_values(args.variants),
            csv_values(args.splits),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"[plan] Databases: {[db.name for db in databases]}")
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
        rag_units = [unit for unit in pending if unit.variant.db is not None]
        if args.skip_rag:
            missing = [unit.rag_input for unit in rag_units if not unit.rag_input.is_file()]
            if missing:
                raise FileNotFoundError(
                    "--skip-rag was requested, but cached retrieval output is missing: "
                    + ", ".join(str(path.relative_to(ROOT)) for path in missing)
                )
        else:
            run_rag(
                rag_units,
                args.num_workers,
                args.retrieve_topk,
                args.bottom_k,
                args.faiss_candidates,
                args.force,
            )

        completed = 0
        started = time.time()
        for index, unit in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {unit}")
            convert_unit(unit, args.prompt_rag_topk)
            atomwise_unit(unit)
            cache_unit(unit)
            completed += 1
        print(f"[main] Completed {completed} unit(s) in {time.time() - started:.1f}s.")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
