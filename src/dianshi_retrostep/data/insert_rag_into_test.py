#!/usr/bin/env python3
"""Insert ChemRAG retrieval results into one or more evaluation JSONL files."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import faiss
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.warning")

FIELDS_TO_EXTRACT = (
    "reactants",
    "products",
    "localtemp_res",
    "synthons_smiles",
    "leaving_groups",
    "hash_id",
    "per_local_temp_rxn_nums",
)

POPCOUNT = np.array([bin(index).count("1") for index in range(256)], dtype=np.int32)


class ChemRAGDatabase:
    """The minimal reader required for a packed-fingerprint ChemRAG database."""

    def __init__(self, directory: Path):
        self.directory = directory
        with (directory / "config.json").open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.n_records = int(self.config["n_records"])
        encoder = self.config.get("encoder", "ecfp").lower()
        if encoder != "ecfp":
            raise ValueError(f"Only ECFP ChemRAG databases are supported, not {encoder!r}.")
        params = self.config.get("encoder_params", {})
        self.fp_dim_bits = int(self.config.get("fp_dim_bits", params.get("n_bits", 2048)))
        radius = int(params.get("radius", 2))
        self.encoder = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=self.fp_dim_bits
        )
        self.product_fps_packed = np.load(directory / "product_fps.npy", mmap_mode="r")
        self.line_offsets = np.load(directory / "line_offsets.npy", mmap_mode="r")
        self.product_index = faiss.read_index_binary(str(directory / "faiss_product.index"))
        with (directory / "metadata.jsonl").open("r", encoding="utf-8") as handle:
            self.metadata = [json.loads(line) for line in handle if line.strip()]

        source_value = str(self.config.get("source_jsonl", ""))
        source_path = Path(source_value) if source_value else None
        if source_path is not None and not source_path.is_absolute():
            source_path = directory / source_path
        self.source_jsonl = source_path if source_path and source_path.is_file() else None
        print(
            f"[db] Loaded {self.n_records:,} records from {directory} "
            f"with {self.fp_dim_bits}-bit ECFP fingerprints."
        )

    def encode(self, smiles: str) -> np.ndarray | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return self.encoder.GetFingerprintAsNumPy(molecule).astype(np.uint8)


def tanimoto_packed(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    intersection = POPCOUNT[np.bitwise_and(query, candidates)].sum(axis=1)
    union = POPCOUNT[np.bitwise_or(query, candidates)].sum(axis=1)
    return np.where(union > 0, intersection / union, 0.0).astype(np.float32)


DATABASE: ChemRAGDatabase | None = None
SOURCE_HANDLE = None


def worker_init() -> None:
    global SOURCE_HANDLE
    if DATABASE is not None and DATABASE.source_jsonl is not None:
        SOURCE_HANDLE = DATABASE.source_jsonl.open("rb")


def full_record(index: int) -> dict:
    assert DATABASE is not None
    if SOURCE_HANDLE is not None:
        try:
            SOURCE_HANDLE.seek(int(DATABASE.line_offsets[index]))
            return json.loads(SOURCE_HANDLE.readline())
        except Exception:
            pass
    return DATABASE.metadata[index]


def payload(record: dict) -> dict:
    nested = record.get("preprocess_for_train")
    return nested if isinstance(nested, dict) else record


def extract_entry(record: dict, similarity: float) -> dict:
    values = payload(record)
    entry = {key: values.get(key) for key in FIELDS_TO_EXTRACT}
    entry["similarity"] = float(similarity)
    return entry


def passes_minimum(entry: dict, minimum: int) -> bool:
    if minimum <= 0:
        return True
    try:
        return int(entry.get("per_local_temp_rxn_nums")) > minimum
    except (TypeError, ValueError):
        return False


def retrieve(
    products: str,
    own_products: str,
    own_reactants: str,
    candidates_count: int,
    top_k: int,
    bottom_k: int,
    minimum_reactions: int,
    max_expand_factor: int,
) -> list[dict]:
    assert DATABASE is not None
    fingerprint = DATABASE.encode(products)
    if fingerprint is None:
        return []
    query = np.packbits(fingerprint).reshape(1, -1)

    base_search = max(candidates_count, top_k + bottom_k)
    expand = 1
    last_candidates = last_scores = last_order = None
    selected: list[dict] = []
    while True:
        search_count = min(base_search * expand, DATABASE.n_records)
        _, indices = DATABASE.product_index.search(query, search_count)
        candidates = indices[0]
        candidates = candidates[candidates >= 0]
        if not len(candidates):
            break
        scores = tanimoto_packed(
            query[0], np.array(DATABASE.product_fps_packed[candidates])
        )
        order = np.argsort(-scores)
        last_candidates, last_scores, last_order = candidates, scores, order

        selected = []
        for local_index in order:
            entry = extract_entry(
                full_record(int(candidates[local_index])), scores[local_index]
            )
            is_self = (
                entry.get("products") == own_products
                and entry.get("reactants") == own_reactants
            )
            if is_self or not passes_minimum(entry, minimum_reactions):
                continue
            selected.append(entry)
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k or search_count >= DATABASE.n_records:
            break
        if max_expand_factor > 0 and expand >= max_expand_factor:
            break
        expand *= 2
        if max_expand_factor > 0:
            expand = min(expand, max_expand_factor)

    if last_order is None:
        return []
    seen = {item.get("hash_id") for item in selected if item.get("hash_id")}
    bottom: list[dict] = []
    for local_index in reversed(last_order):
        if len(bottom) >= bottom_k:
            break
        entry = extract_entry(
            full_record(int(last_candidates[local_index])), last_scores[local_index]
        )
        is_self = (
            entry.get("products") == own_products
            and entry.get("reactants") == own_reactants
        )
        record_hash = entry.get("hash_id")
        if is_self or not passes_minimum(entry, minimum_reactions):
            continue
        if record_hash and record_hash in seen:
            continue
        bottom.append(entry)
        if record_hash:
            seen.add(record_hash)
    return selected + bottom


def normalized_payload(record: dict) -> dict:
    existing = record.get("preprocess_for_train")
    if isinstance(existing, dict):
        return existing
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
    nested = {key: record[key] for key in keys if key in record}
    record["preprocess_for_train"] = nested
    return nested


def joined(value) -> str:
    if isinstance(value, list):
        return ".".join(str(item) for item in value)
    return str(value or "").strip()


def process_chunk(arguments) -> tuple[int, list[str], int, int]:
    (
        chunk_index,
        lines,
        candidates_count,
        top_k,
        bottom_k,
        minimum_reactions,
        max_expand_factor,
    ) = arguments
    output = []
    succeeded = failed = 0
    for line in lines:
        record = json.loads(line)
        nested = normalized_payload(record)
        products = joined(nested.get("products"))
        reactants = joined(nested.get("reactants"))
        if not products:
            nested["prepared_trained_for_rag_sft"] = []
            failed += 1
        else:
            nested["prepared_trained_for_rag_sft"] = retrieve(
                products,
                products,
                reactants,
                candidates_count,
                top_k,
                bottom_k,
                minimum_reactions,
                max_expand_factor,
            )
            succeeded += 1
        output.append(json.dumps(record, ensure_ascii=False))
    return chunk_index, output, succeeded, failed


def chunks(lines: list[str], workers: int, chunk_size: int) -> list[list[str]]:
    if not lines:
        return []
    if chunk_size <= 0:
        target_count = max(1, workers * 4)
        chunk_size = max(1, (len(lines) + target_count - 1) // target_count)
    return [lines[index : index + chunk_size] for index in range(0, len(lines), chunk_size)]


def process_file(pool, source: Path, destination: Path, args) -> tuple[int, int]:
    started = time.time()
    with source.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    batches = chunks(lines, args.num_workers, args.chunk_size)
    jobs = [
        (
            index,
            batch,
            args.faiss_candidates,
            args.top_k,
            args.bottom_k,
            args.min_per_local_temp_rxn_nums,
            args.max_expand_factor,
        )
        for index, batch in enumerate(batches)
    ]
    results: list[list[str] | None] = [None] * len(jobs)
    succeeded = failed = 0
    print(f"[rag] {source} ({len(lines)} records)")
    for index, output, ok, skipped in pool.imap_unordered(process_chunk, jobs, chunksize=1):
        results[index] = output
        succeeded += ok
        failed += skipped

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for batch in results:
                for line in batch or ():
                    handle.write(line + "\n")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"[rag] Wrote {destination}: ok={succeeded}, skipped={failed}, "
        f"time={time.time() - started:.1f}s"
    )
    return succeeded, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Comma-separated input files.")
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--suffix", default="_with_rag")
    parser.add_argument("--faiss-candidates", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=0)
    parser.add_argument("--min-per-local-temp-rxn-nums", type=int, default=0)
    parser.add_argument("--max-expand-factor", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    global DATABASE
    args = parse_args()
    if args.num_workers < 1 or args.top_k < 1 or args.faiss_candidates < 1:
        raise ValueError("Worker, top-k, and candidate counts must be positive.")
    sources = [Path(item.strip()) for item in args.input_jsonl.split(",") if item.strip()]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input file not found: {missing[0]}")
    if not args.db_dir.is_dir():
        raise FileNotFoundError(f"Database directory not found: {args.db_dir}")

    DATABASE = ChemRAGDatabase(args.db_dir)
    mp.set_start_method("fork", force=True)
    with mp.Pool(processes=args.num_workers, initializer=worker_init) as pool:
        for source in sorted(sources):
            destination = source.with_name(f"{source.stem}{args.suffix}{source.suffix}")
            process_file(pool, source, destination, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
