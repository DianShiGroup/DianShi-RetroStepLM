import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract root_smiles from a JSONL patent file into a same-name .txt file."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = input_path.with_suffix(".txt")

    smiles_lines: list[str] = []
    with open(input_path) as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue

            smiles = record.get("root_smiles")
            if not smiles:
                print(
                    f"Warning: missing or empty root_smiles on line {line_num}",
                    file=sys.stderr,
                )
                continue
            smiles_lines.append(smiles)

    output_path.write_text("\n".join(smiles_lines) + "\n")
    print(f"Wrote {len(smiles_lines)} SMILES to {output_path}")


if __name__ == "__main__":
    main()
