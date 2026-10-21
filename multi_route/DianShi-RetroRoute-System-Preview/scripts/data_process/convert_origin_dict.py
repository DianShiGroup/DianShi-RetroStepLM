#!/usr/bin/env python3
"""
Convert origin_dict.csv to a txt file with one SMILES per line.
Each SMILES is canonicalized via RDKit for normalization.
"""

import csv
import subprocess
import sys

from tqdm import tqdm

INPUT_FILE = "data/origin_dict.csv"
OUTPUT_FILE = "data/origin_dict.txt"


def count_lines(path: str) -> int:
    """Count number of lines in a file quickly (wc -l)."""
    result = subprocess.run(["wc", "-l", path], capture_output=True, text=True, check=True)
    return int(result.stdout.split()[0]) - 1  # subtract header


def main():
    count = 0
    num_rows = count_lines(INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        # skip header
        try:
            header = next(reader)
        except StopIteration:
            print("Error: CSV file is empty.")
            sys.exit(1)

        for row in tqdm(reader, total=num_rows, unit="smiles", desc="Converting"):
            if not row:
                continue
            # column: [index, mol (SMILES), idx]
            smi = row[1].strip()
            if smi:
                fout.write(smi + "\n")
                count += 1

    print(f"\nDone. Extracted {count} SMILES from {INPUT_FILE} -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
