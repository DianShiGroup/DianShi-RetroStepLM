"""
Route analysis for retrosynthesis episodes.

Provides two lightweight extractors used by the training/eval pipeline
(error summary + building-block stats) and a JSON writer that persists the
solved route and per-episode diagnostics.
"""
import json
import os

from utils import parse_json_or_raw


def build_error_summary_from_tool_calls(tool_calls):
    tool_errors = {}
    invalid_molecules = []
    missing_building_blocks = []
    total_errors = 0

    for call in tool_calls:
        name = call.get("name") or "unknown"
        result = parse_json_or_raw(call.get("result"))

        if isinstance(result, dict):
            err = result.get("error")
            if isinstance(err, str) and err.strip():
                tool_errors.setdefault(name, []).append(err)
                total_errors += 1

        if name == "search_building_blocks" and isinstance(result, dict):
            results = result.get("results") or {}
            if isinstance(results, dict):
                for mol, info in results.items():
                    status = (info or {}).get("status")
                    if status == "missing" and mol not in missing_building_blocks:
                        missing_building_blocks.append(mol)
                    elif status == "invalid" and mol not in invalid_molecules:
                        invalid_molecules.append(mol)

    return {
        "tool_errors": tool_errors,
        "total_errors": total_errors,
        "invalid_molecules": invalid_molecules,
        "missing_building_blocks": missing_building_blocks,
    }


def build_bb_stats_from_graph(graph_state, error_summary=None):
    invalid_smiles_list = list((error_summary or {}).get("invalid_molecules", []))

    bb_status = {}
    final_molecules = []
    if graph_state and isinstance(graph_state, dict):
        nodes_list = graph_state.get("nodes", [])
        edges_list = graph_state.get("edges", [])

        if isinstance(edges_list, dict):
            edges_list = list(edges_list.values())
        if not isinstance(edges_list, list):
            edges_list = []

        adjacency = {}
        out_degree = {}
        for edge in edges_list:
            if not isinstance(edge, dict):
                continue
            src = edge.get("from")
            dst = edge.get("to")
            if not src or not dst:
                continue
            adjacency.setdefault(src, []).append(dst)
            out_degree[src] = out_degree.get(src, 0) + 1
            out_degree.setdefault(dst, out_degree.get(dst, 0))

        root_id = graph_state.get("root_id") or graph_state.get("root_node_id")
        reachable = set()
        if root_id:
            stack = [root_id]
            while stack:
                nid = stack.pop()
                if nid in reachable:
                    continue
                reachable.add(nid)
                stack.extend(adjacency.get(nid, []))

        if isinstance(nodes_list, list):
            for node in nodes_list:
                if not isinstance(node, dict) or node.get("type") != "molecule":
                    continue
                nid = node.get("id")
                if reachable and nid not in reachable:
                    continue
                smiles = node.get("smiles")
                if smiles:
                    bb_status[smiles] = node.get("bb_status")

            seen = set()
            for node in nodes_list:
                if not isinstance(node, dict) or node.get("type") != "molecule":
                    continue
                nid = node.get("id")
                if reachable and nid not in reachable:
                    continue
                if out_degree.get(nid, 0) != 0:
                    continue
                smiles = node.get("smiles")
                if smiles and smiles not in seen:
                    seen.add(smiles)
                    final_molecules.append(smiles)

    found_blocks = []
    missing_blocks = []
    for mol in final_molecules:
        if mol in invalid_smiles_list:
            continue
        if bb_status.get(mol) == "building_block":
            found_blocks.append((mol, mol))
        else:
            missing_blocks.append(mol)

    return {
        "total_molecules": len(final_molecules),
        "valid_smiles": len(final_molecules) - len(invalid_smiles_list),
        "found_building_blocks": len(found_blocks),
        "missing_building_blocks": len(missing_blocks),
        "invalid_smiles": len(invalid_smiles_list),
        "found_blocks": found_blocks,
        "missing_blocks": missing_blocks,
        "invalid_smiles_list": invalid_smiles_list,
    }


def save_route_json(data, output_dir):
    """Write the solved route and per-episode diagnostics as a single route.json.

    Args:
        data: Dict with target_index, target_smiles, completed, route_tree,
              route_steps, graph_state, tool_calls, and optionally
              reference_route. error_summary/bb_stats are computed if absent.
        output_dir: Directory to write route.json into.
    """
    os.makedirs(output_dir, exist_ok=True)

    graph_state = data.get("graph_state")
    tool_calls = data.get("tool_calls") or []
    route_tree = data.get("route_tree")
    route_steps = data.get("route_steps") or []

    error_summary = data.get("error_summary")
    if error_summary is None:
        error_summary = build_error_summary_from_tool_calls(tool_calls)
    bb_stats = data.get("bb_stats")
    if bb_stats is None:
        bb_stats = build_bb_stats_from_graph(graph_state, error_summary)

    # Route length = number of reaction steps in the solved route.
    route_length = len(route_steps)

    record = {
        "target_index": data.get("target_index"),
        "target_smiles": data.get("target_smiles"),
        "completed": bool(data.get("completed")),
        "route_length": route_length,
        "route_tree": route_tree,
        "route_steps": route_steps,
        "reference_route": data.get("reference_route"),
        "error_summary": error_summary,
        "bb_stats": bb_stats,
    }

    path = os.path.join(output_dir, "route.json")
    with open(path, "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    return path
