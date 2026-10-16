#!/usr/bin/env python3
"""Convert starting molecules text/csv file to pickle for faster loading.

Usage:
    python scripts/convert_stocks_to_pkl.py --input data/stocks.txt
    python scripts/convert_stocks_to_pkl.py --input data/origin_dict.csv

Output: same path with .pkl extension (e.g. data/stocks.pkl).
"""

import argparse
import os
import pickle
import sys


def load_txt(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def load_csv(path: str) -> list[str]:
    import pandas as pd

    df = pd.read_csv(path)
    return df["mol"].dropna().astype(str).str.strip().tolist()


def main():
    parser = argparse.ArgumentParser(description="Convert stocks file to pickle")
    parser.add_argument("--input", required=True, help="Source file (.txt or .csv)")
    args = parser.parse_args()

    src = args.input
    if not os.path.exists(src):
        print(f"Error: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    suffix = os.path.splitext(src)[1].lower()
    if suffix == ".csv":
        items = load_csv(src)
    elif suffix == ".txt":
        items = load_txt(src)
    else:
        print(f"Error: unsupported format: {suffix}", file=sys.stderr)
        sys.exit(1)

    dst = os.path.splitext(src)[0] + ".pkl"
    with open(dst, "wb") as f:
        pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done: {len(items)} molecules written to {dst}")


if __name__ == "__main__":
    main()
