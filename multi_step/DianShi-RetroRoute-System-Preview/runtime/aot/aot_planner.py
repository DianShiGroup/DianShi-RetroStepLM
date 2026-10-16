"""
Planner Agent for parallel retrosynthesis.

Observes the AND-OR synthesis tree globally and makes strategic decisions:
- Prune clearly dead-end branches.
- Select REACTION nodes or frontier MOL nodes for idle Worker slots (expand actions).
- Signal termination when appropriate.

Actions are mutually exclusive: each response returns either prune actions OR
expand actions, never both. This prevents contradictory decisions.

NOTE: expansion of qualifying candidates is handled automatically by the
Dispatcher — Planner does NOT screen individual reactions.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .aot_tree import AOTreeState, NodeType, NodeStatus
from .aot_viz import serialize_tree
from .aot_recent_history import build_recent_route_history

logger = logging.getLogger(__name__)


def extract_reasoning_text(content: str) -> Optional[str]:
    think_match = re.search(
        r" thinking\s*(.*?)\s* response", content, re.DOTALL | re.IGNORECASE
    )
    if think_match:
        return think_match.group(1).strip()

    thinking_match = re.search(
        r"<thinking>\s*(.*?)\s*</thinking>", content, re.DOTALL | re.IGNORECASE
    )
    if thinking_match:
        return thinking_match.group(1).strip()

    tool_match = re.search(r"<tool>", content, re.IGNORECASE)
    if tool_match:
        before_tool = content[: tool_match.start()].strip()
        if before_tool:
            snippet = before_tool[-200:].strip()
            if re.search(r"[a-zA-Z]", snippet):
                return snippet

    return None


@dataclass
class PlannerAction:
    node_id: str = ""
    action: str = ""
    reason: str = ""
    option_id: int | None = None


@dataclass
class PlannerDecision:
    actions: list[PlannerAction] = field(default_factory=list)
    strategy_notes: str = ""
    reasoning: str = ""
    cancelled: bool = False
    parse_error: bool = False

    @property
    def expand_node_ids(self) -> list[str]:
        """Return legacy node_ids from expand actions."""
        return [a.node_id for a in self.actions if a.action == "expand" and a.node_id]

    @property
    def expand_actions(self) -> list[PlannerAction]:
        """Return expand actions, including option_id-based actions."""
        return [
            a
            for a in self.actions
            if a.action == "expand" and (a.node_id or a.option_id is not None)
        ]

    @property
    def is_empty(self) -> bool:
        """True when the decision has no valid actions (not cancelled, not parse error)."""
        return not self.cancelled and not self.parse_error and len(self.actions) == 0


def _is_descendant_or_self(node_id: str, ancestor_id: str) -> bool:
    return node_id == ancestor_id or node_id.startswith(f"{ancestor_id}-")


def _is_subtree_fully_resolved(node) -> bool:
    """True if every leaf under *node* is AVAILABLE or SOLVED (no UNSOLVED/FAILED/PRUNED)."""
    if not node.children:
        return node.status in (NodeStatus.AVAILABLE, NodeStatus.SOLVED)
    return all(_is_subtree_fully_resolved(c) for c in node.children)


def _build_resolved_collapse_set(state: AOTreeState) -> set[str]:
    """Compute nodes whose subtrees are fully resolved.

    Collapse propagates upward: a MOL whose entire subtree is resolved is
    collapsed; a REACTION whose *all* children are collapsed is itself
    collapsed (and likewise for ROOT).  This lets the planner see unsolved
    siblings while hiding the details of solved branches.
    """
    collapsed: set[str] = set()

    def _walk(node) -> bool:
        if _is_subtree_fully_resolved(node):
            collapsed.add(node.node_id)
            return True
        for child in node.children:
            _walk(child)
        return False

    if state.root:
        _walk(state.root)
    return collapsed


def _format_planner_tree_node(
    node,
    depth: int = 0,
    max_depth: int = 50,
    resolved_collapse_nodes: set[str] | None = None,
) -> list[str]:
    resolved_collapse_nodes = resolved_collapse_nodes or set()
    indent = "  " * depth
    node_type = {
        NodeType.ROOT: "TARGET",
        NodeType.MOL: "MOL",
        NodeType.REACTION: "RXN",
    }.get(node.node_type, "UNK")
    status = node.status.value

    if node.node_type == NodeType.REACTION:
        line = f"{indent}[{node.node_id}] {node_type} ({status})"
    else:
        smi = node.smiles or ""
        display_smi = smi[:60] + "..." if len(smi) > 60 else smi
        line = f"{indent}[{node.node_id}] {node_type}: {display_smi} ({status})"

    if depth > 0 and node.node_id in resolved_collapse_nodes:
        return [f"{line} [SUBTREE RESOLVED]"]
    # Root branches (depth=1) that are SOLVED have yielded complete routes.
    # Collapse them so the Planner does not see internal unsolved nodes and
    # try to re-explore an already-successful branch.
    if depth == 1 and node.status == NodeStatus.SOLVED:
        return [f"{line} [SUBTREE SOLVED]"]
    if node.status == NodeStatus.AVAILABLE:
        return [f"{line} [AVAILABLE]"]
    if node.status == NodeStatus.PRUNED:
        return []
    if node.status == NodeStatus.FAILED and node.fail_reason:
        return [f"{line} [FAILED: {node.fail_reason}]"]
    if depth >= max_depth:
        return [f"{line} [MAX TREE CONTEXT DEPTH]"]

    lines = [line]
    for child in node.children:
        lines.extend(
            _format_planner_tree_node(
                child,
                depth + 1,
                max_depth=max_depth,
                resolved_collapse_nodes=resolved_collapse_nodes,
            )
        )
    return lines


def build_planner_tree_context(state: AOTreeState) -> str:
    if not state.root:
        return serialize_tree(state)

    resolved_collapse_nodes = _build_resolved_collapse_set(state)
    lines = [
        "=== Current AND-OR Synthesis Tree (Planner View) ===",
        f"Target: {state.target_smiles}",
        "Resolved branches are collapsed per-MOL; unsolved siblings remain visible.",
        "Each REACTION is one concrete single-step candidate.",
    ]
    lines.extend(
        _format_planner_tree_node(
            state.root,
            resolved_collapse_nodes=resolved_collapse_nodes,
        )
    )
    return "\n".join(lines)


def _root_branch_id(node_id: str) -> str | None:
    parts = node_id.split("-")
    if len(parts) < 2 or parts[0] != "0":
        return None
    return f"{parts[0]}-{parts[1]}"


def _format_unsolved_node_line(
    depth: int,
    node_id: str,
    smiles: str,
) -> str:
    return f"    {node_id}: {smiles[:60]} (depth={depth})"


def get_legal_expand_options(
    state: AOTreeState,
    primary_node_id: str | None,
    max_depth: int | float | None = None,
    blocked_root_ids: set[str] | None = None,
    inflight_node_ids: set[str] | None = None,
) -> list[dict[str, str | int]]:
    """Return current legal expand options.

    Planner should select these by the per-round option_id shown in the
    prompt. `node_id` is retained as the internal action id and legacy
    fallback input. For REACTION choices, `dispatch_node_id` is the first
    unsolved MOL child that Dispatcher will send to a Worker.
    """
    from .aot_state import get_expandable_nodes, node_has_complete_route

    blocked_root_ids = blocked_root_ids or set()
    inflight_node_ids = inflight_node_ids or set()

    def _root_id(node_id: str) -> str | None:
        parts = node_id.split("-")
        if len(parts) < 2 or parts[0] != "0":
            return None
        return f"{parts[0]}-{parts[1]}"

    def _mol_option(node_id: str, depth: int, smiles: str) -> dict[str, str | int] | None:
        node = state.get_node(node_id)
        if node is None or node.node_type != NodeType.MOL:
            return None
        if node.status != NodeStatus.UNSOLVED or not state.is_unsolved(node_id):
            return None
        if node_has_complete_route(state, node_id):
            return None
        if node_id in inflight_node_ids:
            return None
        if _root_id(node_id) in blocked_root_ids:
            return None
        return {
            "node_id": node_id,
            "dispatch_node_id": node_id,
            "kind": "mol",
            "depth": depth,
            "smiles": smiles,
        }

    primary_node = state.get_node(primary_node_id) if primary_node_id else None
    if primary_node is not None:
        reaction_options: list[dict[str, str | int]] = []
        for rxn in primary_node.children:
            if rxn.node_type != NodeType.REACTION or rxn.status == NodeStatus.SOLVED:
                continue
            if _root_id(rxn.node_id) in blocked_root_ids:
                continue
            unsolved_mols = [
                child
                for child in rxn.children
                if child.node_type == NodeType.MOL
                and child.status == NodeStatus.UNSOLVED
                and state.is_unsolved(child.node_id)
                and not node_has_complete_route(state, child.node_id)
                and child.node_id not in inflight_node_ids
                and (max_depth is None or child.depth < max_depth)
            ]
            if not unsolved_mols:
                continue
            dispatch_node = unsolved_mols[0]
            reaction_options.append(
                {
                    "node_id": rxn.node_id,
                    "dispatch_node_id": dispatch_node.node_id,
                    "kind": "reaction",
                    "depth": dispatch_node.depth,
                    "smiles": dispatch_node.smiles,
                }
            )
        if reaction_options:
            return reaction_options

    options = []
    for depth, node_id, smiles in get_expandable_nodes(state, max_depth=max_depth):
        if primary_node_id and not _is_descendant_or_self(node_id, primary_node_id):
            continue
        option = _mol_option(node_id, depth, smiles)
        if option is not None:
            options.append(option)
    return options


def format_legal_expand_options(options: list[dict[str, str | int]]) -> str:
    if not options:
        return "  none"
    lines = []
    for idx, opt in enumerate(options):
        dispatch = opt.get("dispatch_node_id")
        dispatch_part = (
            f" -> dispatches MOL {dispatch}"
            if dispatch and dispatch != opt.get("node_id")
            else ""
        )
        lines.append(
            f"  - option_id={idx} ({opt.get('kind')}, depth={opt.get('depth')})"
            f" node_id={opt.get('node_id')}{dispatch_part}: "
            f"{str(opt.get('smiles') or '')[:60]}"
        )
    return "\n".join(lines)


def build_primary_branch_context(
    state: AOTreeState,
    primary_node_id: str | None,
    max_depth: int | float | None = None,
    blocked_root_ids: set[str] | None = None,
    inflight_node_ids: set[str] | None = None,
) -> str:
    legal_options = get_legal_expand_options(
        state,
        primary_node_id,
        max_depth=max_depth,
        blocked_root_ids=blocked_root_ids,
        inflight_node_ids=inflight_node_ids,
    )
    lines = ["=== Primary Branch Context ==="]
    lines.append("")
    lines.append("=== Legal Expand Options ===")
    lines.append(
        "For action=\"expand\", return ONLY the current option_id from this list. "
        "Do not copy the full node_id for expand. Full node_id is shown only for chemistry context and legacy compatibility."
    )
    lines.append(format_legal_expand_options(legal_options))
    lines.append("")
    if not primary_node_id:
        lines.append("  locked_primary_node: none")
        lines.append(
            "  instruction: Choose one easiest-to-decompose UNSOLVED MOL node "
            "from Legal Expand Options. Start that route as the locked primary."
        )
        return "\n".join(lines)

    primary_node = state.get_node(primary_node_id)
    lines.append(f"  locked_primary_node: {primary_node_id}")
    root_id = _root_branch_id(primary_node_id)
    if root_id:
        lines.append(f"  root_branch: {root_id}")
    if primary_node:
        lines.append(f"  depth: {primary_node.depth}")
    else:
        lines.append("  node_status: missing")
        return "\n".join(lines)

    legal_reaction_ids = {
        str(option["node_id"])
        for option in legal_options
        if option["kind"] == "reaction"
    }
    reaction_children = [
        c for c in primary_node.children
        if c.node_type == NodeType.REACTION and c.node_id in legal_reaction_ids
    ]
    has_reactions = bool(reaction_children)

    if has_reactions:
        lines.append(f"  has {len(reaction_children)} reaction options (OR-alternatives)")
        lines.append("")
        lines.append("  === Choose ONE reaction ===")
        lines.append("  (The dispatcher expands the first unsolved reactant.)")
        lines.append("")

        reaction_options = []
        for rxn in reaction_children:
            if rxn.status == NodeStatus.SOLVED:
                continue
            unsolved_mols = [
                c for c in rxn.children
                if c.node_type == NodeType.MOL
                and c.status == NodeStatus.UNSOLVED
                and state.is_unsolved(c.node_id)
            ]
            if not unsolved_mols:
                continue
            reaction_options.append((rxn, unsolved_mols))

        reaction_options.sort(key=lambda item: (len(item[1]), item[0].node_id))
        for rxn, unsolved_mols in reaction_options:
            lines.append(f"  [{rxn.node_id}] ({len(unsolved_mols)} unsolved reactant(s))")
            lines.append("    Reactants:")
            for mol in unsolved_mols:
                disp = mol.smiles or ""
                disp = disp[:60] + "..." if len(disp) > 60 else disp
                lines.append(
                    f"      - {mol.node_id}: {disp} (depth={mol.depth})"
                )
        lines.append("")
        lines.append(
            "  instruction: Select ONE REACTION by its Legal Expand Options option_id. "
            "Do NOT select multiple reactions — they are OR-alternatives."
        )
    else:
        next_unsolved_nodes = [
            (int(option["depth"]), str(option["node_id"]), str(option["smiles"]))
            for option in legal_options
            if option["kind"] == "mol"
        ]
        if next_unsolved_nodes:
            lines.append("  next_unsolved_nodes:")
            for depth, node_id, smiles in next_unsolved_nodes:
                lines.append(
                    _format_unsolved_node_line(
                        depth, node_id, smiles,
                    )
                )
        else:
            lines.append("  next_unsolved_nodes: none")
        lines.append("")
        lines.append(
            "  instruction: Continue the locked subtree. Dispatcher auto-advances "
            "the locked primary along the AND chain when this subtree is solved. "
            "Do NOT switch branches manually while this subtree still has unsolved nodes."
        )

    return "\n".join(lines)


def build_planner_prompt(
    state: AOTreeState,
    active_workers: list[dict],
    max_depth: int,
    target_complete_paths: int | float = 3,
    primary_node_id: str | None = None,
    system_prompt_text: str = "",
    last_feedback: str | None = None,
    blocked_root_ids: set[str] | None = None,
    inflight_node_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    tree_context = build_planner_tree_context(state)
    reaction_registry = (
        "=== Reaction Candidate Registry ===\n"
        "  Candidate reactions come directly from the single-step model. Use tree "
        "structure, stock availability, and route progress."
    )
    recent_route_history = build_recent_route_history(state, primary_node_id)
    primary_branch_context = build_primary_branch_context(
        state,
        primary_node_id,
        max_depth=max_depth,
        blocked_root_ids=blocked_root_ids,
        inflight_node_ids=inflight_node_ids,
    )

    active_summary = "=== Currently Active Workers ===\n"
    if active_workers:
        for w in active_workers:
            active_summary += f"  Slot {w['slot']}: processing node {w['node_id']}\n"
    else:
        active_summary += "  No active Workers.\n"

    global_summary = f"""
=== Global Tree Signals ===
  solved routes: {state.solved_path_count}
  required routes: {target_complete_paths}
  steps_used: {state.total_steps}
""".strip()

    batch_instruction = (
        "Select ONE frontier option to expand (action=expand). "
        "Use only the integer option_id from Legal Expand Options, e.g. "
        "{\"action\":\"expand\",\"option_id\":0}. "
        "Do NOT copy full node_id values for expand; they are long and error-prone. "
        "Do NOT select options that are already being processed by active expansion Workers."
    )
    advisory_notes = (
        "Do not infer feasibility or quality from display order. Select among "
        "OR-alternative REACTION nodes using stock availability, structural "
        "simplification, and observed route progress."
    )

    task_instruction = f"1. {batch_instruction}"
    action_type_instruction = (
        "- EXPAND: Return actions with action=\"expand\" entries only. Workers will be dispatched to expand those nodes."
    )
    prune_quality_rules = (
        "- If the locked primary route looks weak, continue with the best legal expand action inside the locked primary; "
        "dispatcher-managed hard checks will handle terminal branches.\n"
        "- Keep root-level and sibling branches idle as backups unless the dispatcher surfaces them naturally.\n"
    )

    feedback_section = ""
    if last_feedback:
        feedback_section = (
            "=== Last Round Feedback ===\n"
            f"{last_feedback}\n\n"
        )

    user_content = f"""
{feedback_section}{tree_context}

{reaction_registry}

{recent_route_history}

{primary_branch_context}

{active_summary}

{global_summary}

Your tasks:
{task_instruction}

CRITICAL — Return only allowed action types:
{action_type_instruction}

Quality control rules:
- If no locked primary node yet, select one frontier MOL option_id from Legal Expand Options and build the route.
- When reaction options are shown, select ONE REACTION by option_id from Legal Expand Options. Do NOT select multiple reactions — they are OR-alternatives. The dispatcher will auto-expand the next unsolved reactant.
- Continue the locked primary node's subtree until it finds a complete path, fails, or dispatcher hard rules make it terminal.
- The dispatcher auto-advances the locked primary along the AND chain when subtrees are solved. Do NOT manually switch to AND-sibling nodes.
{prune_quality_rules}- If a branch undergoes multiple decomposition steps but its structural complexity shows no significant decrease, deprioritize lateral chemistry and choose the legal expand action with the clearest core-scaffold simplification.
{advisory_notes}

Output JSON only."""

    return [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_content},
    ]


def _parse_option_id(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
    return None


def parse_planner_response(content: str) -> Optional[PlannerDecision]:
    text = content.strip()

    # Strip markdown code fences if present
    fence_pattern = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the top-level JSON object by counting braces (handles nested JSON)
    start = text.find("{")
    if start == -1:
        logger.warning("No JSON object found in Planner response: %.500s", text)
        return None

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        logger.warning("Unbalanced braces in Planner response: %.500s", text)
        return None

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in Planner response: %.500s", text[start:end])
        return None

    decision = PlannerDecision(
        strategy_notes=data.get("strategy_notes", ""),
    )

    for item in data.get("actions", []):
        action_name = item.get("action", "")
        if action_name == "expand":
            action = PlannerAction(
                node_id=item.get("node_id", ""),
                action="expand",
                option_id=_parse_option_id(item.get("option_id")),
            )
            decision.actions.append(action)

    return decision


async def call_planner(
    state: AOTreeState,
    call_llm_fn: Callable[[list[dict[str, str]]], tuple[str, str] | None],
    active_workers: list[dict] | None = None,
    max_depth: int = 15,
    target_complete_paths: int | float = 3,
    primary_node_id: str | None = None,
    system_prompt_text: str = "",
    last_feedback: str | None = None,
    blocked_root_ids: set[str] | None = None,
    inflight_node_ids: set[str] | None = None,
) -> PlannerDecision:
    messages = build_planner_prompt(
        state=state,
        active_workers=active_workers or [],
        max_depth=max_depth,
        target_complete_paths=target_complete_paths,
        primary_node_id=primary_node_id,
        system_prompt_text=system_prompt_text,
        last_feedback=last_feedback,
        blocked_root_ids=blocked_root_ids,
        inflight_node_ids=inflight_node_ids,
    )

    if last_feedback:
        logger.info(
            "Planner called with last-round feedback (%d chars): %.200s",
            len(last_feedback),
            last_feedback,
        )
    else:
        logger.info("Planner called without last-round feedback")

    llm_output = call_llm_fn(messages)
    if llm_output is None:
        logger.warning("Planner LLM call failed, using fallback")
        return PlannerDecision(cancelled=True)

    content, reasoning_from_llm = llm_output
    reasoning = reasoning_from_llm or extract_reasoning_text(content) or ""

    decision = parse_planner_response(content)
    if decision is None:
        logger.warning("Failed to parse Planner response, using fallback")
        return PlannerDecision(reasoning=reasoning, parse_error=True)

    decision.reasoning = reasoning
    return decision
