"""Convert SMART-based andor_tree to nested AND-OR tree (standard_json format).

Input andor_tree structure:
{
  "node": "<SMILES>",                    # current molecule
  "is_initial": true/false,              # is building block
  "derived_from": [                      # reactions that produce this molecule
    {
      "combination": [                   # reactants for this reaction
        { "node": "...", "is_initial": ..., "derived_from": [...] },
        { "node": "...", "is_initial": ..., "derived_from": [...] }
      ]
    }
  ]
}

Usage:
    python retro_app_backend/scripts/extract_perfect_data.py --input <input.jsonl>

    Reads from `<input.jsonl>`, outputs `<input>_tree.jsonl` and `<input>.txt`.

    Example:
        python retro_app_backend/scripts/extract_perfect_data.py \
            --input data/total_rxns_2025_filtered_with_templates_andor_category.jsonl
"""

import argparse
import json
import os


def convert_node(raw):
    """Convert andor_tree raw dict to nested tree with in_stock."""
    mol_smiles = raw["node"]
    is_initial = raw.get("is_initial", False)
    derived = raw.get("derived_from", [])

    mol_node = {
        "type": "mol",
        "smiles": mol_smiles,
        "in_stock": is_initial,
    }

    if not derived:
        return mol_node

    children = []
    for deriv in derived:
        combination = deriv.get("combination", [])
        if not combination:
            continue
        # reaction SMILES: reactant1.reactant2>>product
        reactant_smiles_list = []
        reactant_nodes = []
        for comb_node in combination:
            reactant_smiles_list.append(comb_node["node"])
            reactant_nodes.append(convert_node(comb_node))

        rxn_smiles = ".".join(reactant_smiles_list) + ">>" + mol_smiles

        rxn_node = {
            "type": "reaction",
            "smiles": rxn_smiles,
            "children": reactant_nodes,
        }
        children.append(rxn_node)

    mol_node["children"] = children
    return mol_node


def main():
    parser = argparse.ArgumentParser(
        description="Extract perfect AND-OR trees from jsonl data.",
    )
    parser.add_argument("--input", required=True, help="Path to input jsonl file.")
    args = parser.parse_args()

    src = args.input
    base = src.rsplit(".", 1)[0]
    out_jsonl = base + "_tree.jsonl"
    out_txt = base + ".txt"

    target_cats = {"perfect_all_common", "perfect_super"}

    records = []
    smiles_list = []

    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("tree_category") in target_cats:
                root_smiles = d["root_smiles"]
                tree = convert_node(d["andor_tree"])
                records.append(tree)
                smiles_list.append(root_smiles)

    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_txt, "w") as f:
        for s in smiles_list:
            f.write(s + "\n")

    print(f"Extracted {len(records)} records to {os.path.basename(out_jsonl)}")
    print(f"Saved {len(smiles_list)} SMILES to {os.path.basename(out_txt)}")


if __name__ == "__main__":
    main()
