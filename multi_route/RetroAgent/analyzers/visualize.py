"""
Standalone visualization for solved retrosynthesis routes.

This module translates a saved route.json (produced by
analyzers/route.save_route_json) into a Mermaid diagram,
and optionally an image if RDKit is available.

Usage:
    python -m analyzers.visualize <path/to/route.json> [--out route.mmd] [--fmt mermaid|image]
"""
import argparse
import json
import os

from utils import safe_label


def route_tree_to_mermaid(route_tree):
    """Render a solved route tree (nested molecule/reaction dict) as Mermaid.

    The tree shape matches GraphMemory.get_solved_route_tree():
        molecule {id (M#), SMILES, status, children:[reaction]}
        reaction {id (R#), status, reactants, children:[molecule]}
    Node kind is inferred from the id prefix (M = molecule, R = reaction).
    """
    lines = ["flowchart TD"]
    if not route_tree:
        lines.append('  EMPTY["No solved route"]')
        return "\n".join(lines)

    def walk(node):
        node_id = node.get("id", "")
        if node_id.startswith("R"):
            lines.append(f'  {node_id}["{node_id}"]:::reaction')
        else:
            smiles = node.get("SMILES") or node_id
            label = safe_label(smiles[:60] + "..." if len(smiles) > 60 else smiles)
            lines.append(f'  {node_id}["{label}"]:::molecule')
        for child in node.get("children") or []:
            child_id = child.get("id")
            lines.append(f"  {node_id} --> {child_id}")
            walk(child)

    walk(route_tree)
    lines.append("  classDef molecule fill:#E8EAF6,stroke:#5C6BC0,stroke-width:1px;")
    lines.append("  classDef reaction fill:#87CEEB,stroke:#4682B4,stroke-width:2px;")
    return "\n".join(lines)


def route_steps_to_image(route_steps, out_path):
    """Render each step as a reaction (reactants >> product), stacked one per row."""
    from rdkit.Chem import AllChem, Draw
    from PIL import Image, ImageDraw

    panels = []
    for i, step in enumerate(route_steps, 1):
        reactants = step.get("Reactants") or step.get("reactants") or []
        product = step.get("Product") or step.get("product")
        if not reactants or not product:
            continue
        rxn_smiles = ".".join(reactants) + ">>" + product
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        except Exception:
            rxn = None
        if rxn is None:
            continue
        img = Draw.ReactionToImage(rxn, subImgSize=(300, 300)).convert("RGB")
        panels.append((i, img))
    if not panels:
        raise ValueError("No renderable reactions in route_steps")

    label_h = 22
    width = max(img.width for _, img in panels)
    total_h = sum(img.height + label_h for _, img in panels)
    canvas = Image.new("RGB", (width, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for i, img in panels:
        draw.text((5, y + 4), f"step {i}", fill="black")
        canvas.paste(img, (0, y + label_h))
        y += img.height + label_h
    canvas.save(out_path)
    return out_path


def visualize(route_json_path, out_path=None, fmt="mermaid"):
    """Translate a route.json into a visualization.

    Args:
        route_json_path: Path to a route.json written by save_route_json.
        out_path: Output path. Defaults next to the input.
        fmt: "mermaid" (.mmd text) or "image" (per-step reaction PNG via RDKit).

    Returns:
        The output path written.
    """
    with open(route_json_path) as f:
        record = json.load(f)

    base = os.path.splitext(route_json_path)[0]
    if fmt == "mermaid":
        out_path = out_path or f"{base}.mmd"
        mmd = route_tree_to_mermaid(record.get("route_tree"))
        with open(out_path, "w") as f:
            f.write(mmd)
        return out_path
    elif fmt == "image":
        out_path = out_path or f"{base}.png"
        return route_steps_to_image(record.get("route_steps") or [], out_path)
    raise ValueError(f"Unknown fmt: {fmt} (expected 'mermaid' or 'image')")


def main():
    parser = argparse.ArgumentParser(description="Visualize a solved retrosynthesis route.")
    parser.add_argument("route_json", help="Path to route.json")
    parser.add_argument("--out", default=None, help="Output path")
    parser.add_argument("--fmt", default="mermaid", choices=["mermaid", "image"])
    args = parser.parse_args()
    out = visualize(args.route_json, args.out, args.fmt)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
