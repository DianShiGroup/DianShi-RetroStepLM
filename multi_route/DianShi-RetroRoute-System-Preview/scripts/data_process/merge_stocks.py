#!/usr/bin/env python3
"""
Merge two stock SMILES files, canonicalize and deduplicate with multi-threading.

Processes files in batches to provide live progress without thread-level printing.
Invalid SMILES (RDKit parse error) are kept as-is.

Usage:
  python scripts/merge_stocks.py \
    --stock-a data/origin_dict.txt \
    --stock-b data/stocks.txt \
    --output data/merged_stocks.txt

  python scripts/merge_stocks.py \
    --stock-a data/origin_dict.txt \
    --stock-b data/stocks.txt \
    --output data/merged_stocks.txt \
    --workers 8 --batch-size 2000
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from rdkit import Chem


# Make progress logs visible immediately even when stdout is piped/nohup-ed.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def log(msg: str = "") -> None:
    """Print progress from the main thread only, always flushed."""
    print(msg, flush=True)


# ── Globally suppress RDKit stderr chatter ──────────────────
try:
    _dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_dn, 2)
    os.close(_dn)
except Exception:
    pass


def canonicalize_smiles_clear_map(smiles: str) -> Optional[str]:
    """Canonicalize SMILES and remove atom-map numbers. Returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def _canon_chunk(chunk: list[tuple[int, str]]) -> list[tuple[int, str, bool]]:
    """Canonicalize a chunk of (local_idx, raw_smiles). Returns (idx, result, success)."""
    results: list[tuple[int, str, bool]] = []
    for idx, text in chunk:
        norm = canonicalize_smiles_clear_map(text)
        if norm is not None:
            results.append((idx, norm, True))
        else:
            results.append((idx, text, False))
    return results


def _canonicalize_batch(
    batch_texts: list[str], workers: int
) -> list[tuple[str, bool]]:
    """Canonicalize a batch of SMILES strings using multi-threading."""
    n = len(batch_texts)
    chunk_size = max(1, n // workers)
    chunks: list[list[tuple[int, str]]] = []
    for i in range(0, n, chunk_size):
        chunk = [(j, batch_texts[j]) for j in range(i, min(i + chunk_size, n))]
        chunks.append(chunk)

    results: list[Optional[tuple[str, bool]]] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_canon_chunk, ch): id(ch) for ch in chunks}
        for future in as_completed(futures):
            for idx, norm, success in future.result():
                results[idx] = (norm, success)

    return [(r[0], r[1]) for r in results]  # type: ignore[misc]


def _process_batch(
    batch_texts: list[str],
    seen: set[str],
    output_handle,
    workers: int,
) -> dict:
    """Canonicalize + dedup + write one batch. Returns {canon_fail, written, newly_deduped}."""
    canon_results = _canonicalize_batch(batch_texts, workers)

    cf = w = nd = 0
    for norm, success in canon_results:
        if not success:
            cf += 1
        if norm in seen:
            nd += 1
        else:
            seen.add(norm)
            output_handle.write(norm + "\n")
            w += 1
    return {"canon_fail": cf, "written": w, "newly_deduped": nd}


def process_file(
    path: Path,
    seen: set[str],
    output_handle,
    label: str,
    workers: int,
    batch_size: int,
) -> dict:
    """Read and canonicalize a stock file in batches.

    For each batch: multi-threaded canonicalization → main-thread dedup + write + log.
    Progress is printed after every batch completion.
    """
    total_lines = 0
    total_canon_fail = 0
    total_written = 0
    total_newly_deduped = 0

    t_start = time.time()

    log(f"[{label}] processing in batches of {batch_size:,}, {workers} workers")

    with open(path, "r", encoding="utf-8") as fh:
        batch_texts: list[str] = []

        for raw_line in fh:
            text = raw_line.strip()
            if not text or text.startswith("#"):
                total_lines += 1
                continue

            batch_texts.append(text)
            total_lines += 1

            if len(batch_texts) >= batch_size:
                stats = _process_batch(batch_texts, seen, output_handle, workers)
                total_canon_fail += stats["canon_fail"]
                total_written += stats["written"]
                total_newly_deduped += stats["newly_deduped"]
                batch_texts.clear()

                # Print after every batch in the main thread.
                elapsed = time.time() - t_start
                rate = total_lines / elapsed if elapsed > 0 else 0
                log(f"[{label}] ... {total_lines:,} processed, "
                    f"{total_written:,} new, "
                    f"{total_newly_deduped:,} dupes, "
                    f"{total_canon_fail:,} canon_fail, "
                    f"{rate:,.0f} lines/s")

        # Final partial batch
        if batch_texts:
            stats = _process_batch(batch_texts, seen, output_handle, workers)
            total_canon_fail += stats["canon_fail"]
            total_written += stats["written"]
            total_newly_deduped += stats["newly_deduped"]

    elapsed = time.time() - t_start
    log(f"[{label}] Done — {total_written:,} new, "
        f"{total_newly_deduped:,} dupes skipped, "
        f"{total_canon_fail:,} canon_fail | {elapsed:.1f}s")

    return {"total": total_lines, "canon_fail": total_canon_fail,
            "written": total_written, "newly_deduped": total_newly_deduped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two stock SMILES files, canonicalize and deduplicate."
    )
    parser.add_argument("--stock-a", required=True,
                        help="Path to first stock file (one SMILES per line).")
    parser.add_argument("--stock-b", required=True,
                        help="Path to second stock file (one SMILES per line).")
    parser.add_argument("--output", required=True,
                        help="Path to output merged file.")
    parser.add_argument("--workers", type=int,
                        default=os.cpu_count() or 4,
                        help="Number of parallel threads (default: CPU count).")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Lines per batch (default: 10000). Progress printed after each batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_a = Path(args.stock_a)
    path_b = Path(args.stock_b)
    output_path = Path(args.output)

    for p, name in [(path_a, "stock-a"), (path_b, "stock-b")]:
        if not p.exists():
            print(f"Error: {name} not found: {p}", file=sys.stderr)
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    t_start = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        sa = process_file(path_a, seen, fout, label=path_a.name,
                          workers=args.workers, batch_size=args.batch_size)
        sb = process_file(path_b, seen, fout, label=path_b.name,
                          workers=args.workers, batch_size=args.batch_size)

    total_out = len(seen)
    total_cf = sa["canon_fail"] + sb["canon_fail"]
    total_dd = sa["newly_deduped"] + sb["newly_deduped"]
    elapsed = time.time() - t_start

    print()
    print(f"{'':=^60}")
    print(f"{'Merge Summary':^60}")
    print(f"{'':=^60}")
    print(f"  Stock A:  {path_a}")
    print(f"  Stock B:  {path_b}")
    print(f"  Output:   {output_path}")
    print(f"{'':─^60}")
    print(f"  {'':<25} {'Stock A':>12} {'Stock B':>12}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}")
    print(f"  {'Total lines':<25} {sa['total']:>12,} {sb['total']:>12,}")
    print(f"  {'Canon fail':<25} {sa['canon_fail']:>12,} {sb['canon_fail']:>12,}")
    print(f"  {'Deduped':<25} {sa['newly_deduped']:>12,} {sb['newly_deduped']:>12,}")
    print(f"  {'Written (new)':<25} {sa['written']:>12,} {sb['written']:>12,}")
    print(f"{'':─^60}")
    print(f"  Final output:             {total_out:>12,}")
    print(f"  Stock A contribution:     {sa['written']:>12,}  "
          f"({sa['written']/max(total_out,1)*100:.1f}%)")
    print(f"  Total deduped:            {total_dd:>12,}")
    print(f"  Total canon fail:         {total_cf:>12,}")
    print(f"  Elapsed:                  {elapsed:>12.1f}s")
    print(f"{'':=^60}")

    print()
    print(f"RESULT: stock_a={sa['total']:,}, "
          f"stock_b={sb['total']:,}, "
          f"dedup={total_dd:,}, "
          f"canon_fail={total_cf:,}, "
          f"final={total_out:,}")


if __name__ == "__main__":
    main()
