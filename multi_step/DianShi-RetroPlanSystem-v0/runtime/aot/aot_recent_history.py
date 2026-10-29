from dataclasses import dataclass

from .aot_tree import AOTreeState, NodeType, TreeNode

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "deprotection": (
        "deprotection",
        "deprotect",
        "deboc",
        "decbz",
        "debenzyl",
        "detosyl",
        "desily",
        "ketal deprotection",
        "hydrolysis of ketal",
    ),
    "protection": (
        "protection",
        "protect",
        "boc protection",
        "cbz protection",
        "tosylation",
        "silylation",
        "benzyl protection",
        "trifluoroacetylation",
        "carbamate formation",
        "ketal formation",
    ),
    "redox": (
        "oxidation",
        "reduction",
        "reduce",
        "hydrogenation",
        "hydrogenolysis",
        "deoxygenation",
    ),
    "coupling": (
        "coupling",
        "cross-coupling",
        "n-arylation",
        "c-n coupling",
        "c-c coupling",
        "buchwald",
        "ullmann",
        "suzuki",
        "hirao",
        "snar",
        "amide coupling",
        "amidation",
    ),
    "cyclization": (
        "cyclization",
        "ring closure",
        "annulation",
        "intramolecular",
        "6-exo-tet",
        "7-endo",
    ),
    "functionalization": (
        "halogenation",
        "iodination",
        "bromination",
        "esterification",
        "hydrolysis",
        "substitution",
        "elimination",
        "isomerization",
        "acylation",
        "formylation",
    ),
}

_PRIORITY = (
    "deprotection",
    "protection",
    "redox",
    "coupling",
    "cyclization",
    "functionalization",
)


@dataclass
class ReactionHistoryLabel:
    primary: str
    secondary: list[str]


def _classify_reaction(rxn: TreeNode) -> ReactionHistoryLabel:
    text = rxn.smiles.lower()
    matches = {
        category
        for category, keywords in _CATEGORY_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    }
    if not matches:
        return ReactionHistoryLabel(primary="other", secondary=[])

    primary = next((category for category in _PRIORITY if category in matches), "other")
    secondary = [
        category for category in _PRIORITY if category in matches and category != primary
    ]
    return ReactionHistoryLabel(primary=primary, secondary=secondary)


def _collect_recent_reactions(
    state: AOTreeState, primary_node_id: str, limit: int
) -> list[TreeNode]:
    reactions: list[TreeNode] = []
    current = state.get_node(primary_node_id)
    while current is not None and len(reactions) < limit:
        if current.node_type == NodeType.REACTION:
            reactions.append(current)
        current = state.get_node(current.parent_id) if current.parent_id else None
    return reactions


def build_recent_route_history(
    state: AOTreeState,
    primary_node_id: str | None,
    limit: int = 5,
) -> str:
    lines = ["=== Recent Route History ==="]
    if not primary_node_id:
        lines.append("  Last selected reactions on locked primary route: none")
        return "\n".join(lines)

    reactions = _collect_recent_reactions(state, primary_node_id, limit)
    if not reactions:
        lines.append("  Last selected reactions on locked primary route: none")
        return "\n".join(lines)

    lines.append(f"Last {len(reactions)} selected reactions on locked primary route:")
    protection_like = 0
    redox = 0
    route_building = 0

    for idx, rxn in enumerate(reactions, start=1):
        label = _classify_reaction(rxn)
        tags = {label.primary, *label.secondary}
        if tags & {"protection", "deprotection"}:
            protection_like += 1
        if "redox" in tags:
            redox += 1
        if tags & {"coupling", "cyclization"}:
            route_building += 1

        secondary = f" (+{', +'.join(label.secondary)})" if label.secondary else ""
        lines.append(f"{idx}. [{rxn.node_id}] {label.primary}{secondary}")
        smiles = rxn.smiles[:120] + "..." if len(rxn.smiles) > 120 else rxn.smiles
        lines.append(f"   {smiles}")

    total = len(reactions)
    lines.append("")
    lines.append("Recent signals:")
    lines.append(f"- protection/deprotection: {protection_like}/{total}")
    lines.append(f"- redox: {redox}/{total}")
    lines.append(f"- coupling/cyclization: {route_building}/{total}")
    return "\n".join(lines)
