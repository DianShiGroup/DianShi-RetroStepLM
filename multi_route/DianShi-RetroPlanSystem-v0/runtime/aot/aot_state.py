"""
AND-OR Tree state management for multi-path retrosynthesis planning.
"""

import json
import logging
import random
from typing import AbstractSet, Any, Optional

from .aot_tree import (
    AOTreeState,
    ConcreteCandidate,
    MolInstance,
    NodeStatus,
    NodeType,
    TacticEdge,
    TreeNode,
)

__all__ = [
    "init_tree",
    "expand_node",
    "evaluate_solved_paths",
    "get_expandable_nodes",
    "prune_max_depth_nodes",
    "should_terminate",
    "extract_prediction_tree",
    "serialize_solved_paths_detail",
    "serialize_solved_paths_as_standard_jsonl",
    "convert_path_to_standard_json",
    "mark_node_failed",
    "mark_pruned",
    "node_has_complete_route",
    "recompute_solved_states",
]

logger = logging.getLogger(__name__)


def _canonicalize_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return smiles


def init_tree(target: str, starting_mols: Optional[AbstractSet[str]] = None) -> AOTreeState:
    canonical_target = _canonicalize_smiles(target)
    starting_mols = starting_mols or frozenset()
    root = TreeNode(
        node_id="0",
        node_type=NodeType.MOL,
        smiles=canonical_target,
        status=NodeStatus.UNSOLVED,
        depth=0,
        product_smiles=canonical_target,
        # Root has no leaving-group context; its abstract identity IS
        # its canonical SMILES. Subsequent MOL nodes inherit
        # abstract_smiles from their parent reaction's reactant_abstract.
        abstract_smiles=canonical_target,
        instances=[
            MolInstance(
                smiles=canonical_target,
                source="root",
                available=(canonical_target in starting_mols),
            )
        ],
    )

    state = AOTreeState(
        root=root,
        all_nodes={"0": root},
        unsolved_nodes=["0"],
        _unsolved_set={"0"},
        node_counter=0,
        starting_mols=starting_mols,
        target_smiles=canonical_target,
        expanded_smiles=set(),
        failed_smiles=set(),
        smiles_to_candidates={},
    )

    return state


def expand_node(
    state: AOTreeState,
    node_id: str,
    scored_candidates: list,
    force_all: bool = True,
) -> AOTreeState:
    """Expand a MOL node with its Worker-scored candidate list.

    Every cleaned single-step candidate becomes one concrete REACTION
    node. Candidates are not atom-mapped, abstracted, or merged.

    Backward-compat shim: if `scored_candidates` looks like a raw
    reactants list (legacy v1.0 input), we wrap each entry in a
    minimal scored stub so existing call sites keep working until they
    are updated.
    """
    node = state.get_node(node_id)
    if node is None:
        logger.warning(f"Node {node_id} not found in tree")
        return state

    if node.status != NodeStatus.UNSOLVED:
        logger.warning(
            f"Node {node_id} is already expanded (status={node.status.value}), skipping"
        )
        return state

    if not scored_candidates:
        state = mark_node_failed(state, node_id, reason="model:no_reactions")
        _update_metrics(state)
        logger.info(f"Node {node_id} has no valid reactions, marked as FAILED")
        return state

    scored_candidates = [_normalize_scored(c) for c in scored_candidates]

    node.reaction_candidates = [c.reactants for c in scored_candidates]
    node.status = NodeStatus.PENDING

    new_reactions: list[TreeNode] = []
    for candidate_idx, candidate in enumerate(scored_candidates):
        reaction_node = _create_reaction_node(
            state, node, candidate_idx, candidate.reactants
        )
        node.children.append(reaction_node)
        new_reactions.append(reaction_node)

    logger.info(
        "expand_node %s: candidates=%d reactions=%d",
        node_id,
        len(scored_candidates),
        len(new_reactions),
    )

    if not new_reactions:
        return mark_node_failed(state, node_id, reason="model:no_reactions")

    for rxn in list(node.children):
        state = _reconcile_upward(state, rxn.node_id)

    state.remove_from_unsolved(node_id)
    state.expanded_smiles.add(node.abstract_smiles)
    state.smiles_to_candidates[node.abstract_smiles] = node.reaction_candidates

    state = _update_ancestors(state, node_id)
    state = recompute_solved_states(state)
    _update_metrics(state)
    return state


def _normalize_scored(c: Any):
    """Accept either a worker.ScoredCandidate or a legacy raw reactants
    list. The legacy shim lets older call sites keep working until they
    pass scored objects directly."""
    if hasattr(c, "reactants") and hasattr(c, "combined_score"):
        return c
    # Legacy: c is a raw `[[smi, avail], ...]`. Wrap it in a stub.
    from .aot_worker import ScoredCandidate

    return ScoredCandidate(
        index=0,
        reactants=c,
        combined_score=0.0,
        feasibility="FEASIBLE",
        rationale="",
    )


def _get_ancestor_mol_smiles(state: AOTreeState, node: TreeNode) -> set[str]:
    ancestor_smiles = set()
    current = node
    while current and current.parent_id:
        parent = state.get_node(current.parent_id)
        if parent is None:
            break
        current = parent
        if current.node_type == NodeType.MOL and current.smiles:
            ancestor_smiles.add(current.smiles)
    return ancestor_smiles


def _create_reaction_node(
    state: AOTreeState,
    parent_mol: TreeNode,
    reaction_idx: int,
    reaction: list,
) -> TreeNode:
    ancestor_smiles = _get_ancestor_mol_smiles(state, parent_mol)
    reactant_smiles_list = []
    for r in reaction:
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            reactant_smiles_list.append((r[0], r[1]))
        else:
            reactant_smiles_list.append((str(r), "unavailable"))

    reactant_ids = []
    for r_idx, (r_smiles, r_avail) in enumerate(reactant_smiles_list):
        child_id = f"{parent_mol.node_id}-{reaction_idx}-{r_idx}"
        canonical_smiles = _canonicalize_smiles(r_smiles)

        slot_abstract = canonical_smiles

        rep_is_available = (
            r_avail == "available" or canonical_smiles in state.starting_mols
        )

        instances: list[MolInstance] = [
            MolInstance(
                smiles=canonical_smiles,
                source="candidate",
                available=rep_is_available,
            )
        ]
        slot_available = rep_is_available

        is_cycle = (
            canonical_smiles == parent_mol.smiles or canonical_smiles in ancestor_smiles
        )

        if slot_available:
            child_status = NodeStatus.AVAILABLE
        elif is_cycle:
            # Prune ancestor-smiles loops like A -> A + X before they inflate subtree counts.
            child_status = NodeStatus.FAILED
        else:
            child_status = NodeStatus.UNSOLVED

        child_node = TreeNode(
            node_id=child_id,
            node_type=NodeType.MOL,
            smiles=canonical_smiles,
            status=child_status,
            parent_id=f"{parent_mol.node_id}-{reaction_idx}",
            depth=parent_mol.depth + 1,
            availability="available" if slot_available else "unavailable",
            product_smiles=canonical_smiles,
            fail_reason="reactant:cyclic" if is_cycle and not slot_available else None,
            abstract_smiles=slot_abstract,
            instances=instances,
        )

        if is_cycle and not slot_available:
            logger.info(
                "Pruned cycle reactant %s under node %s via reaction %s",
                canonical_smiles,
                parent_mol.node_id,
                f"{parent_mol.node_id}-{reaction_idx}",
            )

        state.add_node(child_node)
        reactant_ids.append(child_id)

    reaction_smiles = _format_reaction_smiles(parent_mol.smiles, reactant_smiles_list)

    reaction_node = TreeNode(
        node_id=f"{parent_mol.node_id}-{reaction_idx}",
        node_type=NodeType.REACTION,
        smiles=reaction_smiles,
        status=NodeStatus.UNSOLVED,
        parent_id=parent_mol.node_id,
        depth=parent_mol.depth,
        children=[
            state.get_node(rid)
            for rid in reactant_ids
            if state.get_node(rid) is not None
        ],
        product_smiles=parent_mol.smiles,
        reaction_idx=reaction_idx,
    )

    state.all_nodes[reaction_node.node_id] = reaction_node

    # When a reactant is cyclic-FAILED the whole REACTION cannot succeed (AND logic).
    # The REACTION keeps the concrete child reason, then hides its reactant subtree.
    for r_idx, (r_smiles, r_avail) in enumerate(reactant_smiles_list):
        canonical_smiles = _canonicalize_smiles(r_smiles)
        is_cycle2 = (
            canonical_smiles == parent_mol.smiles or canonical_smiles in ancestor_smiles
        )
        is_available2 = (
            r_avail == "available" or canonical_smiles in state.starting_mols
        )
        if is_cycle2 and not is_available2:
            child_id = f"{parent_mol.node_id}-{reaction_idx}-{r_idx}"
            child_node = state.get_node(child_id)
            if child_node:
                _prune_reaction_due_to_child(state, reaction_node, child_node)
            break

    return reaction_node


def _update_ancestors(
    state: AOTreeState,
    expanded_mol_node_id: str,
) -> AOTreeState:
    """Trace up the parent chain from an expanded MOL node and mark the
    selected reaction index on every ancestor MOL node along the path."""
    current = state.get_node(expanded_mol_node_id)
    if current is None:
        return state

    while current and current.parent_id:
        parent = state.get_node(current.parent_id)
        if parent is None:
            break

        if parent.node_type == NodeType.REACTION:
            mol_parent = state.get_node(parent.parent_id)
            if mol_parent and mol_parent.node_type in (NodeType.MOL, NodeType.ROOT):
                rxn_idx = getattr(parent, "reaction_idx", -1)
                if rxn_idx < 0:
                    parts = parent.node_id.rsplit("-", 1)
                    if len(parts) == 2 and parts[-1].isdigit():
                        rxn_idx = int(parts[-1])
                if rxn_idx >= 0 and rxn_idx not in mol_parent.selected_reaction_indices:
                    mol_parent.selected_reaction_indices.append(rxn_idx)

            current = parent
        elif parent.node_type in (NodeType.MOL, NodeType.ROOT):
            # current is a RXN node whose parent is a MOL; keep tracing up
            current = parent
        else:
            break

    return state


def _format_reaction_smiles(product: str, reactants: list) -> str:
    reactant_strs = []
    for r in reactants:
        if isinstance(r, tuple):
            reactant_strs.append(r[0])
        else:
            reactant_strs.append(str(r))
    return f"{'.'.join(reactant_strs)}>>{product}"


def _select_best_reaction(candidates: list, starting_mols: AbstractSet[str]) -> int:
    best_score = -1
    best_indices = []

    for idx, reaction in enumerate(candidates):
        score = 0
        for r in reaction:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                r_smiles, r_avail = r[0], r[1]
            else:
                r_smiles, r_avail = str(r), "unavailable"

            canonical = _canonicalize_smiles(r_smiles)
            if r_avail == "available" or canonical in starting_mols:
                score += 2
            else:
                score += 1

        if score > best_score:
            best_score = score
            best_indices = [idx]
        elif score == best_score:
            best_indices.append(idx)

    return random.choice(best_indices)


def _update_metrics(state: AOTreeState) -> None:
    if not state.root:
        state.solved_path_count = 0
        return

    state.solved_path_count = _count_solved_paths(state.root, state.starting_mols)


def recompute_solved_states(state: AOTreeState) -> AOTreeState:
    def walk(node: TreeNode) -> NodeStatus:
        for child in node.children or []:
            walk(child)

        if node.status in (NodeStatus.FAILED, NodeStatus.PRUNED, NodeStatus.AVAILABLE):
            return node.status

        if node.node_type == NodeType.REACTION:
            if node.children and all(
                child.status in (NodeStatus.AVAILABLE, NodeStatus.SOLVED)
                for child in node.children
            ):
                node.status = NodeStatus.SOLVED
            elif node.children:
                node.status = NodeStatus.UNSOLVED
            return node.status

        if node.children:
            has_solved_child = any(
                child.node_type == NodeType.REACTION and child.status == NodeStatus.SOLVED
                for child in node.children
            )
            has_active_child = any(
                child.status not in (NodeStatus.FAILED, NodeStatus.PRUNED)
                for child in node.children
            )
            if has_solved_child:
                node.status = NodeStatus.SOLVED
            elif has_active_child:
                node.status = NodeStatus.PENDING
        return node.status

    if state.root:
        walk(state.root)

    state._unsolved_set.clear()
    state.unsolved_nodes.clear()
    for node in state.all_nodes.values():
        if node.node_type == NodeType.MOL and node.status == NodeStatus.UNSOLVED:
            state._unsolved_set.add(node.node_id)
            state.unsolved_nodes.append(node.node_id)

    return state


def _count_solved_paths(node: TreeNode, starting_mols: AbstractSet[str]) -> int:
    if node.status in (NodeStatus.FAILED, NodeStatus.PRUNED):
        return 0

    if (
        node.status == NodeStatus.AVAILABLE
        or node.availability == "available"
        or node.smiles in starting_mols
    ):
        return 1

    if not node.children:
        return 0

    if node.node_type in (NodeType.ROOT, NodeType.MOL):
        return sum(_count_solved_paths(child, starting_mols) for child in node.children)

    if node.node_type == NodeType.REACTION:
        product = 1
        for child in node.children:
            child_count = _count_solved_paths(child, starting_mols)
            if child_count <= 0:
                return 0
            product *= child_count
        return product

    return 0


def evaluate_solved_paths(
    state: AOTreeState,
    elapsed_seconds: float | None = None,
    session_id: str = "",
    max_materialized_paths: int | None = None,
) -> AOTreeState:
    _update_metrics(state)

    if state.solved_path_count > len(state.solved_paths):
        if (
            max_materialized_paths is not None
            and len(state.solved_paths) >= max_materialized_paths
        ):
            return state
        existing = {tuple(p) for p in state.solved_paths}
        remaining = None
        if max_materialized_paths is not None:
            remaining = max(0, max_materialized_paths - len(state.solved_paths))
        new_paths = _discover_all_paths(state.root, limit=remaining)
        for path in new_paths:
            path_tuple = tuple(path)
            if path_tuple not in existing:
                state.solved_paths.append(path)
                existing.add(path_tuple)
                elapsed_str = (
                    f" | took {elapsed_seconds:.1f}s"
                    if elapsed_seconds is not None
                    else ""
                )
                session_str = f" | session_id={session_id}" if session_id else ""
                logger.info(
                    f"Found completed path! Total completed: {len(state.solved_paths)}{elapsed_str}{session_str}"
                )
                if (
                    max_materialized_paths is not None
                    and len(state.solved_paths) >= max_materialized_paths
                ):
                    break

    return state


def _discover_all_paths(
    node: TreeNode,
    current: list[str] | None = None,
    limit: int | None = None,
) -> list[list[str]]:
    if limit is not None and limit <= 0:
        return []
    if current is None:
        current = []
    current = current + [node.node_id]

    if node.status == NodeStatus.AVAILABLE:
        return [current]

    if node.status == NodeStatus.FAILED or not node.children:
        return []

    if node.node_type in (NodeType.ROOT, NodeType.MOL):
        all_paths = []
        for child in node.children:
            child_limit = None if limit is None else limit - len(all_paths)
            if child_limit is not None and child_limit <= 0:
                break
            all_paths.extend(_discover_all_paths(child, current, limit=child_limit))
        return all_paths

    if node.node_type == NodeType.REACTION:
        child_results = [
            _discover_all_paths(child, current, limit=limit)
            for child in node.children
        ]
        if any(len(r) == 0 for r in child_results):
            return []

        from itertools import product

        result = []
        for combo in product(*child_results):
            full = list(combo[0])
            for child_path in combo[1:]:
                full.extend(child_path[len(current):])
            result.append(full)
            if limit is not None and len(result) >= limit:
                break
        return result

    return []


def node_has_complete_route(state: AOTreeState, node_id: str) -> bool:
    node = state.get_node(node_id)
    return node is not None and node.status in (NodeStatus.AVAILABLE, NodeStatus.SOLVED)


def get_expandable_nodes(
    state: AOTreeState, max_depth: int | float | None = None
) -> list[tuple[int, str, str]]:
    node_infos = []
    seen = set()
    for node_id in state.unsolved_nodes:
        if node_id in seen:
            continue
        seen.add(node_id)
        node = state.get_node(node_id)
        if node and node.status == NodeStatus.UNSOLVED and state.is_unsolved(node_id):
            if node_has_complete_route(state, node_id):
                continue
            if max_depth is not None and node.depth >= max_depth:
                continue
            node_infos.append((node.depth, node_id, node.smiles))

    node_infos.sort(key=lambda x: (-x[0], x[1]))
    return node_infos


def prune_max_depth_nodes(
    state: AOTreeState,
    max_depth: int | float | None,
    reach_max_depth_prune: bool = False,
) -> list[str]:
    if max_depth is None:
        return []

    pruned_node_ids: list[str] = []
    pruned_seen: set[str] = set()
    for node_id in list(state.unsolved_nodes):
        node = state.get_node(node_id)
        if node and node.status == NodeStatus.UNSOLVED and node.depth >= max_depth:
            prune_node = (
                _root_child_for_subtree(state, node)
                if reach_max_depth_prune
                else node
            )
            if prune_node is None or prune_node.node_id in pruned_seen:
                continue
            pruned_seen.add(prune_node.node_id)
            prune_type = "max_depth_root_branch" if reach_max_depth_prune else "too_deep"
            reason = (
                f"max_depth_reached:{max_depth}; pruned root branch from {node_id}"
                if reach_max_depth_prune
                else f"max_depth_reached:{max_depth}"
            )
            mark_pruned(state, prune_node.node_id, prune_type=prune_type, reason=reason)
            pruned_node_ids.append(prune_node.node_id)

    return pruned_node_ids


def _root_child_for_subtree(state: AOTreeState, node: TreeNode) -> TreeNode | None:
    current = node
    while current.parent_id:
        parent = state.get_node(current.parent_id)
        if parent is None:
            return None
        if parent.node_id == "0":
            return current
        current = parent
    return None


def _is_dead(node: TreeNode) -> bool:
    return node.status in (NodeStatus.FAILED, NodeStatus.PRUNED)


def _terminal_reason(node: TreeNode) -> str:
    if node.status == NodeStatus.FAILED:
        return node.fail_reason or "failed"
    if node.status == NodeStatus.PRUNED:
        return node.prune_reason or "pruned"
    return node.status.value


def _build_reaction_prune_reason(child: TreeNode) -> tuple[str, str]:
    prefix = (
        "reactant_failed" if child.status == NodeStatus.FAILED else "reactant_pruned"
    )
    reason_str = f"{prefix}:{child.node_id}:{_terminal_reason(child)}"
    return (prefix, reason_str)


def _classify_terminal_reason(node: TreeNode) -> str:
    """将终端节点的终止原因分类为简短类型码。"""
    if node.status == NodeStatus.FAILED:
        r = node.fail_reason or ""
        if r.startswith("reactant:cyclic") or "Cyclic" in r:
            return "cyclic"
        elif "no_reaction" in r:
            return "no_reactions"
        elif "low_score" in r:
            return "low_score"
        else:
            return "other"
    if node.status == NodeStatus.PRUNED:
        return "pruned"
    return "other"


def _build_mol_failed_reason(mol: TreeNode) -> str:
    dead_children = [c for c in mol.children if _is_dead(c)]
    if not dead_children:
        return "mol_failed:no_children"

    type_counts: dict[str, int] = {}
    for child in dead_children:
        t = _classify_terminal_reason(child)
        type_counts[t] = type_counts.get(t, 0) + 1

    # 取占比最高的类型
    dominant = max(type_counts, key=type_counts.get)

    # 如果 pruned 占多数说明被策略剪枝，否则如果是 cyclic/no_reactions/low_score
    # 则直接映射；如果多种原因混合作 mixed
    other_sum = sum(v for k, v in type_counts.items() if k != dominant)
    if other_sum == 0:
        return f"mol_failed:{dominant}"
    elif dominant == "cyclic" and type_counts["cyclic"] > other_sum:
        return "mol_failed:cyclic"
    elif dominant == "no_reactions" and type_counts["no_reactions"] > other_sum:
        return "mol_failed:no_reactions"
    elif dominant == "low_score" and type_counts["low_score"] > other_sum:
        return "mol_failed:low_score"
    elif dominant == "pruned":
        return "mol_failed:pruned"
    else:
        return "mol_failed:mixed"


def _prune_reaction_due_to_child(
    state: AOTreeState, reaction: TreeNode, child: TreeNode
) -> bool:
    if reaction.status in (NodeStatus.PRUNED, NodeStatus.FAILED):
        return False

    reaction.status = NodeStatus.PRUNED
    reaction_type, reaction_reason = _build_reaction_prune_reason(child)
    reaction.prune_type = reaction_type
    reaction.prune_reason = reaction_reason
    # A pruned reaction is an AND-branch dead end: all sibling reactant
    # subtrees are now wasted work.  Strip them fully so the Planner stops
    # dispatching their descendants and in-flight worker results are discarded.
    for sibling in reaction.children or []:
        if sibling.node_id == child.node_id:
            continue
        # Remove all descendants from state and unsolved list
        _strip_node_children(state, sibling)
        sibling.status = NodeStatus.PRUNED
        sibling.prune_type = reaction_type
        sibling.prune_reason = "sibling pruned"
        state.remove_from_unsolved(sibling.node_id)
    state.remove_from_unsolved(reaction.node_id)
    logger.info(
        "Reaction %s marked PRUNED: %s (all sibling subtrees stripped)",
        reaction.node_id,
        reaction.prune_reason,
    )
    return True


def _reconcile_reaction(state: AOTreeState, reaction: TreeNode) -> bool:
    if reaction.node_type != NodeType.REACTION:
        return False
    if reaction.status in (NodeStatus.PRUNED, NodeStatus.FAILED, NodeStatus.SOLVED):
        return False

    dead_child = next((child for child in reaction.children if _is_dead(child)), None)
    if dead_child is not None:
        return _prune_reaction_due_to_child(state, reaction, dead_child)

    if reaction.children and all(
        child.status in (NodeStatus.AVAILABLE, NodeStatus.SOLVED)
        for child in reaction.children
    ):
        reaction.status = NodeStatus.SOLVED
        state.remove_from_unsolved(reaction.node_id)
        logger.info("Reaction %s marked SOLVED", reaction.node_id)
        return True

    return False


def _reconcile_mol(state: AOTreeState, mol: TreeNode) -> bool:
    if mol.node_type not in (NodeType.MOL, NodeType.ROOT):
        return False
    if mol.status in (NodeStatus.FAILED, NodeStatus.PRUNED, NodeStatus.AVAILABLE, NodeStatus.SOLVED):
        return False
    if not mol.children:
        return False

    solved_child = next(
        (
            child
            for child in mol.children
            if child.node_type == NodeType.REACTION and child.status == NodeStatus.SOLVED
        ),
        None,
    )
    if solved_child is not None:
        mol.status = NodeStatus.SOLVED
        state.remove_from_unsolved(mol.node_id)
        logger.info("Molecule %s marked SOLVED via %s", mol.node_id, solved_child.node_id)
        return True

    if not all(_is_dead(child) for child in mol.children):
        return False

    # v2.1: every REACTION cluster under this MOL is dead. Try the next
    # unexpanded instance before declaring the abstract MOL itself dead.
    if _try_instance_fallback(state, mol, "all_reactions_dead"):
        return False

    mol.status = NodeStatus.FAILED
    mol.fail_reason = _build_mol_failed_reason(mol)
    state.remove_from_unsolved(mol.node_id)
    logger.info("Molecule %s marked FAILED: %s", mol.node_id, mol.fail_reason)
    return True


def _strip_node_children(state: AOTreeState, node: TreeNode):
    """移除 node 的所有后代节点，从 state.all_nodes 中清理，确保 planner 不可见。"""
    for child in node.children or []:
        state.remove_from_unsolved(child.node_id)
        _strip_node_children(state, child)
        state.all_nodes.pop(child.node_id, None)
    node.children = []


def _reconcile_upward(state: AOTreeState, node_id: str) -> AOTreeState:
    current = state.get_node(node_id)
    while current and current.parent_id:
        parent = state.get_node(current.parent_id)
        if parent is None:
            break

        if parent.node_type == NodeType.REACTION:
            changed = _reconcile_reaction(state, parent)
        elif parent.node_type in (NodeType.MOL, NodeType.ROOT):
            changed = _reconcile_mol(state, parent)
        else:
            changed = False

        current = parent
        if not changed and not _is_dead(parent) and parent.status != NodeStatus.SOLVED:
            break

    return state


def _try_instance_fallback(
    state: AOTreeState,
    mol: TreeNode,
    reason: str,
) -> bool:
    """v2.1 instance-level fallback (§4.4 / §6).

    When the current representative instance of an abstract MOL fails to
    produce a viable expansion (worker emitted no FEASIBLE candidates,
    all subtrees dead, etc.), we don't immediately mark the abstract
    MOL FAILED and trigger AND-OR backtrack. Instead we look for an
    unexpanded sibling instance in the pool — same abstract identity,
    different concrete LG / activation level — and rotate it into the
    primary slot so the dispatcher's next loop tick re-expands the MOL
    on that instance's SMILES.

    Returns True if a fallback was performed (caller should abort the
    pending FAILED transition); False if the pool is exhausted.
    """
    if mol.node_type != NodeType.MOL:
        return False
    if not mol.instances:
        return False

    # Mark the just-failed primary so we never retry it.
    primary = next(
        (inst for inst in mol.instances if inst.smiles == mol.smiles), None
    )
    if primary is not None:
        primary.expanded = True
        primary.failed = True

    nxt = next(
        (
            inst
            for inst in mol.instances
            if not inst.expanded and not inst.failed and inst.smiles != mol.smiles
        ),
        None,
    )
    if nxt is None:
        # Pool is exhausted — every instance has either been tried as
        # primary (expanded=True) or already failed. The MOL will now
        # genuinely be marked FAILED. Track this so we can tell, in
        # post-hoc analysis, when richer pool population (more LG
        # variants from vLLM, or stock-side expansion of the abstract
        # identity) would have given the search another chance.
        state.abstraction_metrics["fallback_pool_exhausted"] = (
            state.abstraction_metrics.get("fallback_pool_exhausted", 0) + 1
        )
        logger.info(
            "Instance fallback unavailable on MOL %s (reason=%s): pool of "
            "%d instances all expanded/failed; MOL will be marked FAILED",
            mol.node_id,
            reason,
            len(mol.instances),
        )
        return False

    # Strip everything attached to the failed primary so the rebuilt
    # subtree can't accidentally inherit dead children.
    _strip_node_children(state, mol)
    mol.children = []
    mol.reaction_candidates = []
    mol.selected_reaction_indices = []
    mol.fail_reason = None
    mol.smiles = nxt.smiles
    mol.product_smiles = nxt.smiles
    mol.status = NodeStatus.UNSOLVED
    if mol.node_id not in state._unsolved_set:
        state._unsolved_set.add(mol.node_id)
        state.unsolved_nodes.append(mol.node_id)
    state.failed_smiles.discard(mol.abstract_smiles or mol.smiles)
    state.abstraction_metrics["instance_fallback_count"] = (
        state.abstraction_metrics.get("instance_fallback_count", 0) + 1
    )
    state.pending_events.append({
        "kind": "instance_fallback",
        "mol_node_id": mol.node_id,
        "from_smiles": primary.smiles if primary else "?",
        "to_smiles": nxt.smiles,
        "remaining_unexpanded": sum(
            1 for i in mol.instances if not i.expanded and not i.failed
        ),
        "reason": reason,
    })
    logger.info(
        "Instance fallback on MOL %s (reason=%s): %s → %s "
        "(remaining unexpanded instances=%d)",
        mol.node_id,
        reason,
        primary.smiles if primary else "?",
        nxt.smiles,
        sum(1 for i in mol.instances if not i.expanded and not i.failed),
    )
    return True


def mark_node_failed(
    state: AOTreeState, node_id: str, reason: str | None = None
) -> AOTreeState:
    node = state.get_node(node_id)
    if node is None or node.status == NodeStatus.FAILED:
        return state

    # v2.1: before declaring an abstract MOL dead, try rotating to the
    # next unexpanded instance in its pool. Only worker/expansion-side
    # failure reasons trigger fallback — structural rejections (cycles,
    # depth caps) genuinely terminate the node.
    if (
        node.node_type == NodeType.MOL
        and reason in {
            "model:no_reactions",
            "worker:no_reactions",
            "worker:low_score",
            "all_self_loop_no_stock",
        }
        and _try_instance_fallback(state, node, reason or "unspecified")
    ):
        return state

    node.status = NodeStatus.FAILED
    if reason:
        node.fail_reason = reason
    state.remove_from_unsolved(node_id)
    state.failed_smiles.add(node.smiles)
    state = _reconcile_upward(state, node_id)
    state = recompute_solved_states(state)
    _update_metrics(state)
    logger.info(
        f"Node {node_id} marked as FAILED, depth={node.depth}, reason={reason or 'unspecified'}"
    )
    return state



def _prune_descendants(state: AOTreeState, node: TreeNode):
    """Mark all descendants of a pruned node as PRUNED.

    When strip_children=False, descendants are kept in all_nodes for
    frontend display, but they must also be marked PRUNED so the
    Planner tree formatter hides them and dispatch logic rejects them.
    """
    for child in node.children or []:
        if child.status not in (NodeStatus.FAILED, NodeStatus.PRUNED):
            child.status = NodeStatus.PRUNED
            child.prune_type = "ancestor_pruned"
            child.prune_reason = f"ancestor {node.node_id} pruned"
        state.remove_from_unsolved(child.node_id)
        _prune_descendants(state, child)


def mark_pruned(
    state: AOTreeState,
    node_id: str,
    prune_type: str = "",
    reason: str = "",
    strip_children: bool = True,
) -> AOTreeState:
    node = state.get_node(node_id)
    if node is None or node.status in (NodeStatus.FAILED, NodeStatus.PRUNED):
        return state
    node.status = NodeStatus.PRUNED
    node.prune_type = prune_type
    node.prune_reason = reason if reason else prune_type
    if strip_children:
        _strip_node_children(state, node)
    else:
        # Keep descendants in all_nodes (for frontend display)
        # but mark them PRUNED and remove from unsolved so the Planner
        # tree formatter hides them and dispatch rejects them.
        _prune_descendants(state, node)
    state.remove_from_unsolved(node_id)
    state = _reconcile_upward(state, node_id)
    state = recompute_solved_states(state)
    _update_metrics(state)
    logger.info(
        f"Node {node_id} marked as PRUNED (reason={reason}), depth={node.depth}, unsolved_remaining={len(state.unsolved_nodes)}, solved_paths={len(state.solved_paths)}"
    )
    if node_id == "0":
        logger.warning(
            f"ROOT node 0 was PRUNED! Tree is dead. unsolved_nodes={state.unsolved_nodes}, solved_paths={state.solved_paths}"
        )
    return state


def should_terminate(
    state: AOTreeState,
    max_iter: int | float,
    target_complete_paths: int | float,
    max_depth: int | float,
) -> tuple[bool, str]:
    if state.solved_path_count >= target_complete_paths:
        return True, "max_paths_reached"

    if state.total_steps >= max_iter:
        return True, "max_iter_reached"

    if not get_expandable_nodes(state, max_depth=max_depth):
        if state.solved_path_count > 0:
            return True, "all_explored_partial"
        return True, "all_explored_no_solution"

    return False, "continue"


def _tactic_edge_dict(edge: TacticEdge) -> dict:
    return {
        "src_smiles": edge.src_smiles,
        "dst_smiles": edge.dst_smiles,
        "rxn_smiles": edge.rxn_smiles,
        "score": edge.score,
        "rationale": edge.rationale,
    }


def _mol_instance_dict(inst: MolInstance) -> dict:
    return {
        "smiles": inst.smiles,
        "source": inst.source,
        "available": inst.available,
        "expanded": inst.expanded,
        "failed": inst.failed,
        "score": inst.score,
        "feasibility": inst.feasibility,
        "tactic_edges": [_tactic_edge_dict(e) for e in inst.tactic_edges],
    }


def _available_instance_smiles(instances) -> str | None:
    for inst in instances or []:
        if isinstance(inst, dict):
            if inst.get("available") and inst.get("smiles"):
                return inst["smiles"]
        elif getattr(inst, "available", False) and getattr(inst, "smiles", None):
            return inst.smiles
    return None


def export_smiles_for_available_mol(
    smiles: str | None, availability: str | None, instances=None
) -> str | None:
    if availability == "available":
        return _available_instance_smiles(instances) or smiles
    return smiles


def _concrete_candidate_dict(cand: ConcreteCandidate) -> dict:
    return {
        "reactants": [
            list(r) if isinstance(r, (list, tuple)) else [str(r), "unavailable"]
            for r in cand.reactants
        ],
        "rxn_smiles": cand.rxn_smiles,
        "score": cand.score,
        "feasibility": cand.feasibility,
        "rationale": cand.rationale,
        "score_source": cand.score_source,
        "abstract_rxn_key": cand.abstract_rxn_key,
        "is_representative": cand.is_representative,
        "mapping_failed": cand.mapping_failed,
    }


def _mol_extras_dict(node: TreeNode) -> dict:
    """v2.1 MOL-only payload: abstract identity + instance pool. The pool
    drives the inline-expandable view: each instance is one concrete
    realization of this abstract MOL, and `tactic_edges` materializes
    the FGI/protection/activation chain that links it to the primary."""
    extras: dict[str, Any] = {}
    if node.abstract_smiles is not None:
        extras["abstract_smiles"] = node.abstract_smiles
    if node.instances:
        extras["instances"] = [_mol_instance_dict(i) for i in node.instances]
    return extras


def _reaction_extras_dict(node: TreeNode) -> dict:
    """v2.1 REACTION-only payload: cluster identity + same-cluster
    candidate pool. The pool drives the inline-expandable view: clicking
    a REACTION reveals every concrete variant collapsed into this
    abstract step (Suzuki Br/I/OTf, etc.)."""
    extras: dict[str, Any] = {}
    if node.abstract_rxn_key is not None:
        extras["abstract_rxn_key"] = node.abstract_rxn_key
    if node.candidate_pool:
        extras["candidate_pool"] = [
            _concrete_candidate_dict(c) for c in node.candidate_pool
        ]
    return extras


def _abstraction_extras_dict(node: TreeNode) -> dict:
    """Dispatch on node type to assemble the v2.1 payload."""
    if node.node_type in (NodeType.MOL, NodeType.ROOT):
        return _mol_extras_dict(node)
    if node.node_type == NodeType.REACTION:
        return _reaction_extras_dict(node)
    return {}


def serialize_solved_paths_detail(state: AOTreeState) -> list[dict]:
    """Convert solved_paths (list of node_id arrays) into a detailed JSON-serializable
    structure with SMILES, type, and status for each node in each path.

    Paths retain their deterministic tree order.
    """
    if not state.solved_paths:
        return []

    detail_paths = []
    for path_index, node_ids in enumerate(state.solved_paths):
        steps = []
        for node_id in node_ids:
            node = state.get_node(node_id)
            if node:
                extras = _abstraction_extras_dict(node)
                step = {
                    "node_id": node.node_id,
                    "type": node.node_type.value,
                    "smiles": export_smiles_for_available_mol(
                        node.smiles, node.availability, extras.get("instances")
                    ),
                    "status": node.status.value,
                    "availability": node.availability,
                    **extras,
                }
                steps.append(step)
        detail_paths.append(
            {
                "path_index": path_index,
                "steps": steps,
                "avg_score": None,
            }
        )

    for new_index, path in enumerate(detail_paths):
        path["path_index"] = new_index

    return detail_paths


def convert_path_to_standard_json(path_steps: list[dict]) -> dict:
    """Convert a single solved path (flat steps list) into standard_json.json format.

    The flat steps follow the traversal order: mol -> reaction -> mol -> reaction -> ...
    Node IDs encode the tree hierarchy (e.g. "0", "0-0", "0-0-0", "0-0-1", "0-1", ...).
    This function reconstructs the nested tree by parsing node_id depth and parent-child
    relationships.
    """
    if not path_steps:
        return {}

    node_map: dict[str, dict] = {}
    root = None

    for step in path_steps:
        node_id = step.get("node_id", "")
        depth = node_id.count("-")
        is_mol = step["type"] in ("root", "mol")

        entry: dict[str, Any] = {}
        if is_mol:
            entry = {
                "type": "mol",
                "smiles": export_smiles_for_available_mol(
                    step["smiles"], step.get("availability"), step.get("instances")
                ),
                "in_stock": step.get("availability") == "available",
            }
            # v2.1: pass through abstract identity + instance pool so
            # the frontend can render the inline-expandable instance list.
            if "abstract_smiles" in step:
                entry["abstract_smiles"] = step["abstract_smiles"]
            if "instances" in step:
                entry["instances"] = step["instances"]
        elif step["type"] == "reaction":
            entry = {
                "type": "reaction",
                "smiles": step["smiles"],
            }
            # v2.1: pass through cluster identity + candidate pool so
            # the frontend can render the same-cluster variant list.
            if "abstract_rxn_key" in step:
                entry["abstract_rxn_key"] = step["abstract_rxn_key"]
            if "candidate_pool" in step:
                entry["candidate_pool"] = step["candidate_pool"]

        entry["_node_id"] = node_id

        node_map[node_id] = entry

        if depth == 0:
            root = entry
        else:
            parent_id = node_id.rsplit("-", 1)[0]
            parent = node_map.get(parent_id)
            if parent:
                parent.setdefault("children", []).append(entry)

    def clean(node: dict) -> None:
        node.pop("_node_id", None)
        children = node.get("children")
        if children:
            for child in children:
                clean(child)
        if "children" in node and not node["children"]:
            del node["children"]

    if root:
        clean(root)
    return root or {}


def serialize_solved_paths_as_standard_jsonl(
    solved_paths_detail: list[dict],
) -> str:
    """Convert solved_paths_detail into JSONL string in standard_json format.

    Each line is one complete route as a nested tree matching standard_json.json structure.
    """
    lines = []
    for path_entry in solved_paths_detail:
        steps = path_entry.get("steps", [])
        tree = convert_path_to_standard_json(steps)
        lines.append(json.dumps(tree, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def build_route_smiles_list(solved_paths_detail: list[dict]) -> list[list[str]]:
    """Build a list of reaction SMILES per solved path.

    Returns a list of lists, where each inner list contains the reaction SMILES
    for one complete synthesis path, ordered layer by layer.
    """
    result: list[list[str]] = []
    for path_entry in solved_paths_detail:
        steps = path_entry.get("steps", [])
        route = [s["smiles"] for s in steps if s.get("type") == "reaction"]
        result.append(route)
    return result


def extract_prediction_tree(state: AOTreeState, score_mode: bool = False) -> Optional[dict]:
    if not state.root:
        return None

    def build_dict_with_metrics(node: TreeNode) -> tuple[int, dict]:
        complete_count = _count_solved_paths(node, state.starting_mols)
        children_dicts: list[dict] = []

        if node.children:
            valid_children = [c for c in node.children if c is not None]
            child_results = [build_dict_with_metrics(c) for c in valid_children]
            children_dicts = [cr for _, cr in child_results]

        result: dict[str, Any] = {
            "type": node.node_type.value,
            "node_id": node.node_id,
            "smiles": node.smiles,
            "status": node.status.value,
            "availability": node.availability,
            "depth": node.depth,
            "solved_route_count": complete_count,
            "solved_path_count": complete_count,
            "subtree_complete_route_count": complete_count,
            "has_complete_subtree_route": complete_count > 0,
            "score": round(node.score, 4),
            "visits": node.visits,
        }

        if children_dicts:
            result["children"] = children_dicts

        if node.node_type == NodeType.REACTION:
            result["reaction_smiles"] = node.smiles
            result["prior_score"] = round(node.prior_score, 4)
            if node.strategy_notes is not None:
                result["strategy_notes"] = node.strategy_notes

        if node.status == NodeStatus.PRUNED:
            if node.prune_type:
                result["prune_type"] = node.prune_type
            if node.prune_reason:
                result["prune_reason"] = node.prune_reason
            if node.fail_reason:
                result["fail_reason"] = node.fail_reason

        if node.status == NodeStatus.FAILED and node.fail_reason:
            result["fail_reason"] = node.fail_reason

        # v2.1 abstraction-clustering payload.
        result.update(_abstraction_extras_dict(node))

        if score_mode and node.node_type in (NodeType.MOL, NodeType.ROOT):
            result["ucb_score"] = round(node.ucb_score, 4)

        if node.node_type == NodeType.MOL and node.children:
            result["selected_reaction_indices"] = node.selected_reaction_indices
            result["total_reactions"] = len(node.children)

        return complete_count, result

    _, result = build_dict_with_metrics(state.root)
    return result
