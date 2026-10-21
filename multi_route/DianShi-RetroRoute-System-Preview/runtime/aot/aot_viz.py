"""
Tree visualization utilities for AND-OR Tree debugging and display.
"""

from .aot_tree import TreeNode, NodeStatus, NodeType, AOTreeState


def visualize_tree(
    node: TreeNode, depth: int = 0, show_children: bool = True, max_depth: int = 100
) -> str:
    if depth > max_depth:
        return "  " * depth + f"[{node.node_id}] ... (max depth exceeded)"
    lines = []
    indent = "  " * depth

    status_icon = (
        "\u2713"
        if node.status == NodeStatus.SOLVED
        else (
            "\u2713"
            if node.status == NodeStatus.AVAILABLE
            else (
                "\u2717"
                if node.status in (NodeStatus.FAILED, NodeStatus.PRUNED)
                else "?"
            )
        )
    )

    type_icon = {
        NodeType.ROOT: "TARGET",
        NodeType.MOL: "MOL",
        NodeType.REACTION: "RXN",
    }.get(node.node_type, "UNK")

    status_str = node.status.value
    if node.availability == "available" and node.status == NodeStatus.UNSOLVED:
        status_str = "available"

    smiles_display = node.smiles[:50] + "..." if len(node.smiles) > 50 else node.smiles

    expandable_marker = ""
    if (
        node.node_type in (NodeType.MOL, NodeType.ROOT)
        and node.status == NodeStatus.UNSOLVED
    ):
        expandable_marker = " [CAN EXPAND]"
    elif node.node_type in (NodeType.MOL, NodeType.ROOT) and node.status in (
        NodeStatus.AVAILABLE,
        NodeStatus.SOLVED,
    ):
        expandable_marker = " [CANNOT EXPAND - already available/solved]"
    elif (
        node.node_type in (NodeType.MOL, NodeType.ROOT)
        and node.status == NodeStatus.FAILED
    ):
        expandable_marker = " [CANNOT EXPAND - failed]"
    elif (
        node.node_type in (NodeType.MOL, NodeType.ROOT)
        and node.status == NodeStatus.PRUNED
    ):
        expandable_marker = " [CANNOT EXPAND - pruned]"

    if node.node_type == NodeType.REACTION:
        lines.append(
            f"{indent}[{node.node_id}] {type_icon} ({status_str}) {status_icon}"
        )
        if node.reaction_candidates:
            lines.append(f"{indent}  Candidates: {len(node.reaction_candidates)}")
        if node.selected_reaction_indices:
            lines.append(f"{indent}  Selected: {node.selected_reaction_indices}")
    elif node.status == NodeStatus.PRUNED and node.prune_reason:
        lines.append(
            f"{indent}[{node.node_id}] {type_icon}: {smiles_display} ({status_str}: {node.prune_reason}) {status_icon}{expandable_marker}"
        )
    else:
        lines.append(
            f"{indent}[{node.node_id}] {type_icon}: {smiles_display} ({status_str}) {status_icon}{expandable_marker}"
        )

    if show_children and node.children:
        if node.node_type == NodeType.MOL:
            lines.append(
                f"{indent}  \u2514\u2500\u2500 Expanding with {len(node.children)} reactions..."
            )
        for child in node.children:
            if depth > 0:
                child_lines = visualize_tree(child, depth + 1, show_children)
                for line in child_lines.split("\n"):
                    lines.append(line)
            else:
                lines.append(visualize_tree(child, depth + 1, show_children))

    return "\n".join(lines)


def serialize_tree(state: AOTreeState, max_depth: int = 50) -> str:
    lines = []
    lines.append("=== Current AND-OR Synthesis Tree ===")
    lines.append(f"Target: {state.target_smiles}")
    lines.append(f"Total Steps: {state.total_steps}")
    lines.append(f"Completed Routes: {state.solved_path_count}")
    lines.append(f"Unsolved Nodes: {len(state.unsolved_nodes)}")
    lines.append(f"Materialized Routes: {len(state.solved_paths)}")
    lines.append("")
    lines.append("Tree Structure:")

    if state.solved_path_count > 0:
        lines.append(
            f"{state.solved_path_count} route(s) currently exist from the root - focus on solving remaining branches to reach your target_complete_paths target."
        )
    if state.root:
        tree_str = visualize_tree(state.root)
        lines.append(tree_str)

    return "\n".join(lines)
