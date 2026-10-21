"""
Dispatcher for Planner-Worker retrosynthesis.

Event-driven architecture:
  - Workers run in parallel, up to MAX_CONCURRENT_WORKERS slots.
  - When a Worker finishes with qualifying candidates, Dispatcher automatically
    expands them all into the tree.  Nodes with no qualifying candidates are
    marked failed.
  - Planner is called for global decisions: prune dead-end branches, select
    REACTION nodes or frontier MOL nodes, and signal termination.
  - Planner does NOT screen individual reactions — expansion is automatic.
"""

import asyncio
import logging
import math
import time
from typing import AbstractSet, Callable, Optional

from .aot_worker import TaskRequest, TaskResult, ScoredCandidate, run_worker
from ..errors import ServiceExhaustedError
from .aot_worker_rules import canonicalize_smiles
from .aot_state import (
    expand_node,
    mark_pruned,
    mark_node_failed,
    get_expandable_nodes,
    prune_max_depth_nodes,
    node_has_complete_route,
    recompute_solved_states,
)
from .aot_tree import AOTreeState, NodeStatus, NodeType

logger = logging.getLogger(__name__)

MAX_CONCURRENT_WORKERS = 10


def _classify_result(result: TaskResult) -> str:
    if result.status == "cancelled":
        logger.info(
            "Classify %s: cancelled (worker aborted due to session cancellation)",
            result.node_id,
        )
        return "cancelled"

    if result.status in ("all_failed", "all_filtered") or result.dead_end_signal:
        logger.info(
            "Classify %s: exception (status=%s dead_end=%s)",
            result.node_id,
            result.status,
            result.dead_end_signal,
        )
        return "exception"

    if result.status == "timeout":
        logger.info(
            "Classify %s: timed_out (vLLM timeout, node stays UNSOLVED for retry)",
            result.node_id,
        )
        return "timed_out"

    if not result.candidates:
        logger.info("Classify %s: exception (no candidates)", result.node_id)
        return "exception"

    logger.info("Classify %s: success (%d candidates)", result.node_id, len(result.candidates))
    return "success"


async def _auto_expand(
    state: AOTreeState,
    tree_lock: asyncio.Lock,
    result: TaskResult,
    emit_event: Optional[Callable] = None,
) -> None:
    """Automatically expand every qualifying candidate from a Worker result.

    Every cleaned candidate is expanded. If no candidates remain, the node is
    marked failed.
    """
    if (
        result.status in ("all_failed", "all_filtered", "cancelled")
        or not result.candidates
    ):
        if result.status == "timeout":
            logger.info(
                "Auto-expand %s: vLLM timed out, node stays UNSOLVED for retry",
                result.node_id,
            )
            return

        if result.status == "cancelled":
            logger.info(
                "Auto-expand %s: worker cancelled, node stays UNSOLVED",
                result.node_id,
            )
            return

        async with tree_lock:
            fail_code = (
                "worker:no_reactions"
                if result.status == "all_failed"
                else "worker:no_candidates"
            )
            mark_node_failed(
                state,
                result.node_id,
                reason=fail_code,
            )
        logger.info(
            "Auto-expand %s: no candidates, marked failed (status=%s)",
            result.node_id,
            result.status,
        )
        return

    async with tree_lock:
        # Pass every cleaned single-step candidate directly into the tree.
        expand_node(state, result.node_id, result.candidates)
        recompute_solved_states(state)
        # P3.B: drain v2.1 SSE events accumulated by state mutators.
        pending = state.pending_events
        state.pending_events = []
    if emit_event:
        for ev in pending:
            emit_event(ev)
    logger.info(
        "Auto-expand %s: expanded %d candidates",
        result.node_id,
        len(result.candidates),
    )


async def run_colony_loop(
    target: str,
    max_iter: int | float,
    target_complete_paths: int | float,
    max_depth: int | float,
    call_llm_fn: Callable,
    single_step_fn: Callable,
    starting_mols: AbstractSet[str] | None = None,
    on_step: Optional[Callable] = None,
    emit: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    system_prompt_text: str = "",
    stats_snapshot_fn: Optional[Callable[[], dict]] = None,
    reach_max_depth_prune: bool = False,
    session_id: str = "",
    worker_timeout: float = 300.0,
) -> dict:
    logger.info(
        "run_colony_loop entered | target=%.60s max_iter=%d target_complete_paths=%d max_depth=%d reach_max_depth_prune=%s session_id=%s",
        target,
        max_iter,
        target_complete_paths,
        max_depth,
        reach_max_depth_prune,
        session_id,
    )
    from .aot_state import (
        init_tree,
        evaluate_solved_paths,
        should_terminate,
        extract_prediction_tree,
        serialize_solved_paths_detail,
        recompute_solved_states,
    )
    from .aot_planner import (
        call_planner,
        format_legal_expand_options,
        get_legal_expand_options,
    )

    state = init_tree(target, starting_mols)
    tree_lock = asyncio.Lock()
    turn = 0
    planner_no_dispatch_count = 0
    max_planner_no_dispatch = 10
    start_time = time.time()
    last_solved_count = 0
    first_path_elapsed_sec: float | None = None
    first_path_stats: dict | None = None

    def _bounded_int(value, default: int = 0) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(numeric) or numeric <= 0:
            return default
        return int(numeric)

    def _path_materialization_limit() -> int:
        target_limit = _bounded_int(target_complete_paths)
        return max(target_limit, 20) if target_limit else 20

    materialized_path_limit = _path_materialization_limit()

    def _should_terminate(_paths_detail: list[dict]) -> tuple[bool, str]:
        return should_terminate(state, max_iter, target_complete_paths, max_depth)

    def _emit_path_events(paths_detail: list[dict]):
        nonlocal last_solved_count, first_path_elapsed_sec, first_path_stats
        if state.solved_path_count > last_solved_count:
            elapsed = time.time() - start_time
            if first_path_elapsed_sec is None:
                first_path_elapsed_sec = round(elapsed, 2)
                snapshot = {}
                if stats_snapshot_fn is not None:
                    try:
                        snapshot = stats_snapshot_fn() or {}
                    except Exception:
                        logger.exception("Failed to capture first-path stats snapshot")
                        snapshot = {}
                first_path_stats = {
                    "elapsed_sec": first_path_elapsed_sec,
                    "total_steps": state.total_steps,
                    "single_step_calls": state.total_steps,
                    **snapshot,
                }
            logger.info(
                "Path completed %d/%d | elapsed=%.1fs | session_id=%s",
                state.solved_path_count,
                target_complete_paths,
                elapsed,
                session_id,
            )
            previous_total = last_solved_count
            current_total = state.solved_path_count
            new_paths = current_total - previous_total
            target_paths = _bounded_int(target_complete_paths, current_total)
            _emit(
                "path_completed",
                path_index=min(current_total, target_paths),
                previous_total_paths=previous_total,
                total_paths=current_total,
                new_paths=new_paths,
                target_paths=target_paths,
                elapsed_seconds=round(elapsed, 1),
                route_count=state.solved_path_count,
                materialized_route_count=len(paths_detail),
                run_stats_first=first_path_stats if previous_total == 0 else None,
            )
            last_solved_count = state.solved_path_count
    termination_reason = "max_iter_reached"

    _available_slots = [f"w{i}" for i in range(1, MAX_CONCURRENT_WORKERS + 1)]
    _worker_slot_map: dict[str, str] = {}
    _pending_coros: set = set()
    _inflight_node_ids: set[str] = set()
    locked_primary_node_id: str | None = None
    last_planner_feedback: str | None = None

    def _has_inflight_descendants(node_id: str) -> bool:
        return any(
            _is_descendant_or_self(nid, node_id) for nid in _inflight_node_ids
        )

    def _assign_slot(real_id: str) -> str | None:
        if real_id in _worker_slot_map:
            return _worker_slot_map[real_id]
        if _available_slots:
            slot = _available_slots.pop(0)
            _worker_slot_map[real_id] = slot
            return slot
        return None

    def _release_slot(real_id: str) -> str | None:
        slot = _worker_slot_map.pop(real_id, None)
        if slot:
            _available_slots.append(slot)
        return slot

    def _active_workers() -> list[dict]:
        nodes = []
        for task_id, slot in _worker_slot_map.items():
            node_id = task_id.split("-", 1)[1] if "-" in task_id else task_id
            nodes.append({"slot": slot, "node_id": node_id})
        return nodes

    def _prune_too_deep_nodes() -> None:
        nonlocal locked_primary_node_id
        pruned_node_ids = prune_max_depth_nodes(
            state,
            max_depth,
            reach_max_depth_prune=reach_max_depth_prune,
        )
        for pruned_id in pruned_node_ids:
            _inflight_node_ids.difference_update(
                node_id
                for node_id in list(_inflight_node_ids)
                if _is_descendant_or_self(node_id, pruned_id)
            )
            if locked_primary_node_id and _is_descendant_or_self(
                locked_primary_node_id, pruned_id
            ):
                logger.info(
                    "Clearing locked primary %s after max-depth pruning of %s",
                    locked_primary_node_id,
                    pruned_id,
                )
                locked_primary_node_id = None

    def _root_branch_id(node_id: str) -> str | None:
        parts = node_id.split("-")
        if len(parts) < 2 or parts[0] != "0":
            return None
        return f"{parts[0]}-{parts[1]}"

    def _is_descendant_or_self(node_id: str, ancestor_id: str) -> bool:
        return node_id == ancestor_id or node_id.startswith(f"{ancestor_id}-")

    def _diversity_blocked_roots() -> set[str]:
        solved_roots = {
            path[1]
            for path in state.solved_paths
            if len(path) >= 2 and state.get_node(path[1]) is not None
        }
        if not solved_roots:
            return set()

        expandable_unsolved_roots = set()
        for _, node_id, _ in get_expandable_nodes(state, max_depth=max_depth):
            root_id = _root_branch_id(node_id)
            if root_id and root_id not in solved_roots:
                expandable_unsolved_roots.add(root_id)

        return solved_roots if expandable_unsolved_roots else set()

    def _node_has_expandable_frontier(node_id: str) -> bool:
        return any(
            _is_descendant_or_self(frontier_id, node_id)
            for _, frontier_id, _ in get_expandable_nodes(state, max_depth=max_depth)
        )

    def _find_next_and_sibling(mol_node_id: str) -> str | None:
        node = state.get_node(mol_node_id)
        while node and node.parent_id:
            parent = state.get_node(node.parent_id)
            if parent is None:
                break
            if parent.node_type == NodeType.REACTION:
                for child in parent.children:
                    if (
                        child.node_type == NodeType.MOL
                        and child.node_id != node.node_id
                        and child.status == NodeStatus.UNSOLVED
                        and state.is_unsolved(child.node_id)
                        and not node_has_complete_route(state, child.node_id)
                    ):
                        return child.node_id
            node = parent
        return None

    def _nearest_primary_node_with_frontier(node_id: str | None) -> str | None:
        current = state.get_node(node_id) if node_id else None
        blocked_roots = _diversity_blocked_roots()
        root_branch = _root_branch_id(node_id) if node_id else None
        while current is not None:
            # Stop at the root branch boundary — don't walk up to root,
            # otherwise the primary would unlock to the entire tree.
            if current.node_type == NodeType.ROOT:
                break
            rid = _root_branch_id(current.node_id)
            if rid != root_branch:
                break
            if (
                rid not in blocked_roots
                and _node_has_expandable_frontier(current.node_id)
            ):
                return current.node_id
            current = state.get_node(current.parent_id) if current.parent_id else None
        return None

    def _nearest_reaction_choice_mol(node_id: str | None) -> str | None:
        current = state.get_node(node_id) if node_id else None
        while current is not None:
            if (
                current.node_type == NodeType.MOL
                and current.status != NodeStatus.SOLVED
                and _node_has_expandable_frontier(current.node_id)
            ):
                active_reactions = [
                    child
                    for child in current.children
                    if child.node_type == NodeType.REACTION
                    and child.status not in (NodeStatus.FAILED, NodeStatus.PRUNED, NodeStatus.SOLVED)
                ]
                if active_reactions:
                    return current.node_id
            current = state.get_node(current.parent_id) if current.parent_id else None
        return None

    def _advance_locked_primary(node_id: str | None) -> str | None:
        if not node_id:
            return None

        current = state.get_node(node_id)
        if current is None:
            # mark_pruned strips descendants of a pruned ancestor from
            # all_nodes; if the locked primary was inside that subtree it is
            # gone. Walk up the id-encoded path to the first surviving
            # ancestor so AND/OR backtracking resumes instead of unlocking.
            parts = node_id.split("-")
            while len(parts) > 1:
                parts.pop()
                ancestor_id = "-".join(parts)
                if state.get_node(ancestor_id) is not None:
                    return (
                        _nearest_reaction_choice_mol(ancestor_id)
                        or _nearest_primary_node_with_frontier(ancestor_id)
                        or "0"
                    )
            return "0"

        if current.status == NodeStatus.SOLVED:
            and_sibling = _find_next_and_sibling(node_id)
            if and_sibling:
                return and_sibling
            reaction_choice_mol = _nearest_reaction_choice_mol(node_id)
            if reaction_choice_mol:
                return reaction_choice_mol
            # Walk up to nearest ancestor with expandable frontier.  If none
            # found (the entire root branch is solved), lock to root "0" so
            # the Planner sees the full tree and can pick from other root
            # branches instead of looping on the solved one.
            return _nearest_primary_node_with_frontier(node_id) or "0"

        if _node_has_expandable_frontier(node_id):
            expandable = get_expandable_nodes(state, max_depth=max_depth)
            direct_mol_ids = {
                child.node_id
                for child in current.children
                if child.node_type == NodeType.MOL
                and child.status == NodeStatus.UNSOLVED
                and state.is_unsolved(child.node_id)
                and not node_has_complete_route(state, child.node_id)
            }
            if any(eid in direct_mol_ids for _, eid, _ in expandable):
                return node_id

            for child in current.children:
                if (
                    child.node_type == NodeType.MOL
                    and child.status == NodeStatus.UNSOLVED
                    and not node_has_complete_route(state, child.node_id)
                    and _node_has_expandable_frontier(child.node_id)
                ):
                    return child.node_id
            return node_id

        # FAILED / dead-frontier case: prefer OR-retry (backtrack to nearest
        # MOL ancestor with active reactions) before AND-sibling traversal.
        # AND-siblings of a failed branch live in chains that may themselves
        # become invalid if the OR-parent ultimately fails — retrying the
        # OR-parent's other reactions first matches AND-OR tree semantics.
        reaction_choice_mol = _nearest_reaction_choice_mol(node_id)
        if reaction_choice_mol:
            return reaction_choice_mol

        and_sibling = _find_next_and_sibling(node_id)
        if and_sibling and _node_has_expandable_frontier(and_sibling):
            return and_sibling

        return _nearest_primary_node_with_frontier(node_id) or "0"

    def _primary_node_from_expands(
        node_ids: list[str], current_primary: str | None
    ) -> str | None:
        for node_id in node_ids:
            node = state.get_node(node_id)
            if node is None or node.status != NodeStatus.UNSOLVED:
                continue
            if node_has_complete_route(state, node_id):
                continue
            # Resolve REACTION nodes to their first unsolved MOL child
            # so the primary lock points to the actual frontier workers expand.
            if node.node_type == NodeType.REACTION:
                for child in node.children:
                    if (
                        child.node_type == NodeType.MOL
                        and child.status == NodeStatus.UNSOLVED
                        and state.is_unsolved(child.node_id)
                    ):
                        node_id = child.node_id
                        node = child
                        break
                else:
                    continue
            if node.node_type != NodeType.MOL:
                continue
            # Reject MOL nodes inside pruned subtrees: strip_children=False
            # keeps them in all_nodes with status=UNSOLVED, but they are
            # removed from _unsolved_set and must not become the primary.
            if not state.is_unsolved(node_id):
                continue
            if current_primary and not _is_descendant_or_self(node_id, current_primary):
                continue
            if _root_branch_id(node_id) in _diversity_blocked_roots():
                continue
            return node_id
        return None

    def _legal_expand_options() -> list[dict[str, str | int]]:
        return get_legal_expand_options(
            state,
            locked_primary_node_id,
            max_depth=max_depth,
            blocked_root_ids=_diversity_blocked_roots(),
            inflight_node_ids=_inflight_node_ids,
        )

    def _legal_expand_feedback(options: list[dict[str, str | int]] | None = None) -> str:
        options = _legal_expand_options() if options is None else options
        return (
            "Current Legal Expand Options are:\n"
            f"{format_legal_expand_options(options)}"
        )

    def _emit(status: str, **kwargs):
        if emit:
            emit({"turn": turn, "status": status, **kwargs})

    def _emit_abstraction(event: dict):
        # P3.B: surface state-layer abstraction events on the SSE
        # channel so the frontend (and CLI viewers) see clustering and
        # instance-fallback as they happen, not just at end-of-run.
        if emit:
            emit({"turn": turn, "status": "abstraction_event", **event})

    _emit(
        "initializing",
        message="Starting Planner-Worker exploration...",
        workerCount=MAX_CONCURRENT_WORKERS,
    )

    def _queue_worker(
        task_prefix: str,
        node_id: str,
        strategy_notes: str | None = None,
    ) -> bool:
        node = state.get_node(node_id)
        if node is None or node.status != NodeStatus.UNSOLVED:
            return False
        if not state.is_unsolved(node_id):
            return False
        if node_has_complete_route(state, node_id):
            return False
        if node_id in _inflight_node_ids:
            return False
        if node.depth >= max_depth:
            logger.info(
                "Skip %s: depth %s reached max_depth %s",
                node_id,
                node.depth,
                max_depth,
            )
            return False
        if strategy_notes:
            node.strategy_notes = strategy_notes
        _inflight_node_ids.add(node_id)
        task = TaskRequest(
            task_id=f"{task_prefix}-{node_id}",
            node_id=node_id,
            molecule_smiles=node.smiles,
            depth=node.depth,
        )
        slot_id = _assign_slot(task.task_id)
        if not slot_id:
            _inflight_node_ids.discard(node_id)
            return False
        _pending_coros.add(
            asyncio.ensure_future(
                run_worker(
                    task,
                    single_step_fn,
                    timeout=worker_timeout,
                    should_stop=should_stop,
                )
            )
        )
        _emit(
            "worker_start",
            worker_id=slot_id,
            node_id=node_id,
            molecule_smiles=node.smiles,
        )
        return True

    def _pick_initial_node() -> str | None:
        blocked_roots = _diversity_blocked_roots()
        for _, node_id, _ in get_expandable_nodes(state, max_depth=max_depth):
            if node_id in _inflight_node_ids:
                continue
            root_id = _root_branch_id(node_id)
            if root_id in blocked_roots:
                logger.info(
                    "Skip %s: root branch %s already has solved path(s); preserving diversity",
                    node_id,
                    root_id,
                )
                continue
            node = state.get_node(node_id)
            if node is None or node.status != NodeStatus.UNSOLVED:
                continue
            return node_id
        return None

    initial_node_id = _pick_initial_node()
    if initial_node_id and _queue_worker("init", initial_node_id):
        logger.info("Initial batch: 1 task dispatched (%s)", initial_node_id)
    else:
        logger.info("No expandable nodes at start")
        termination_reason = "no_expandable_nodes"

    # --- Event loop ---
    logger.info("Entering main event loop | pending_coros=%d", len(_pending_coros))
    while turn < max_iter:
        if should_stop and should_stop():
            termination_reason = "cancelled"
            logger.info("Cancellation requested, stopping main loop")
            break

        _prune_too_deep_nodes()
        paths_detail = serialize_solved_paths_detail(state)
        should_stop_flag, reason = _should_terminate(paths_detail)
        if should_stop_flag:
            termination_reason = reason
            logger.info("Termination: %s", reason)
            break

        if not _pending_coros:
            if should_stop and should_stop():
                termination_reason = "cancelled"
                logger.info("Cancellation requested, stopping main loop")
                break
            if not state.unsolved_nodes:
                termination_reason = "no_expandable_nodes"
                logger.info("No pending workers and no unsolved nodes, terminating")
                break
            wake_planner = True
            awakened_node_ids: list[str] = []
        else:
            logger.info(
                "Event loop iteration %d | waiting on %d pending worker(s)",
                turn,
                len(_pending_coros),
            )
            t0_wait = time.time()
            done, _pending_coros = await asyncio.wait(
                _pending_coros, return_when=asyncio.FIRST_COMPLETED
            )
            logger.info(
                "Event loop iteration %d | asyncio.wait returned %d done in %.2fs",
                turn,
                len(done),
                time.time() - t0_wait,
            )

            wake_planner = False
            awakened_node_ids: list[str] = []

            service_fail = None
            for coro in done:
                try:
                    result = await coro
                except asyncio.CancelledError:
                    logger.info("Worker coroutine was cancelled before result handling")
                    continue
                except ServiceExhaustedError as exc:
                    logger.error(
                        "Single-step service exhausted, failing case: %s",
                        exc.termination_reason,
                    )
                    service_fail = exc
                    break
                except Exception:
                    logger.exception("Worker coroutine failed before result handling")
                    continue
                turn += 1
                state.total_steps += 1

                _inflight_node_ids.discard(result.node_id)
                slot_id = _release_slot(result.task_id)
                display_id = slot_id or result.task_id

                classification = _classify_result(result)
                logger.info(
                    "Worker done %s (%s, %d candidates)",
                    result.node_id,
                    classification,
                    len(result.candidates),
                )

                _emit(
                    "worker_done",
                    worker_id=display_id,
                    node_id=result.node_id,
                    result={
                        "status": result.status,
                        "candidates": [
                            {
                                "index": c.index,
                                "reactants": c.reactants,
                            }
                            for c in result.candidates
                        ],
                        "pipeline_stats": result.pipeline_stats,
                    },
                )

                await _auto_expand(state, tree_lock, result, emit_event=_emit_abstraction)
                async with tree_lock:
                    _prune_too_deep_nodes()
                wake_planner = True
                awakened_node_ids.append(result.node_id)

            if service_fail is not None:
                # Single-step service exhausted (retries/timeouts exhausted):
                # stop the search and report the case as failed instead of
                # letting the search continue with degraded expansions.
                termination_reason = service_fail.termination_reason
                break

        # Emit tree_update immediately after worker expansions so the UI
        # reflects new nodes before the (potentially slow) planner call.
        _prune_too_deep_nodes()
        recompute_solved_states(state)
        evaluate_solved_paths(
            state,
            time.time() - start_time,
            session_id=session_id,
            max_materialized_paths=materialized_path_limit,
        )
        tree_dict = extract_prediction_tree(state, score_mode=False)
        paths_detail = serialize_solved_paths_detail(state)
        _emit_path_events(paths_detail)
        _emit("tree_update", tree=tree_dict, turns=state.total_steps)

        should_stop_flag, reason = _should_terminate(paths_detail)
        if should_stop_flag:
            termination_reason = reason
            logger.info("Termination: %s", reason)
            break

        if wake_planner:
            if locked_primary_node_id:
                advanced_primary = _advance_locked_primary(
                    locked_primary_node_id
                )
                if advanced_primary != locked_primary_node_id:
                    logger.info(
                        "Advancing primary node %s -> %s",
                        locked_primary_node_id,
                        advanced_primary,
                    )
                    locked_primary_node_id = advanced_primary

            logger.info(
                "Waking Planner | %d active worker(s), %d free slot(s)",
                len(_worker_slot_map),
                len(_available_slots),
            )

            # Planner's focus is the (possibly advanced) primary, not the
            # just-finished worker nodes — those can be FAILED and mislead UI.
            focus_node_ids = (
                [locked_primary_node_id] if locked_primary_node_id else list(awakened_node_ids)
            )

            _emit(
                "planner_busy",
                batch_id="",
                workerCount=MAX_CONCURRENT_WORKERS,
                node_ids=focus_node_ids,
            )

            _emit(
                "planner_thinking",
                batch_id="",
                thinking_text="",
                node_ids=focus_node_ids,
            )

            planner_decision = await call_planner(
                state=state,
                call_llm_fn=call_llm_fn,
                active_workers=_active_workers(),
                max_depth=max_depth,
                target_complete_paths=target_complete_paths,
                primary_node_id=locked_primary_node_id,
                system_prompt_text=system_prompt_text,
                last_feedback=last_planner_feedback,
                blocked_root_ids=_diversity_blocked_roots(),
                inflight_node_ids=_inflight_node_ids,
            )

            if planner_decision.cancelled:
                logger.warning(
                    "Planner LLM call failed — skipping this turn. "
                    "Inflight workers will continue and retry planner on next wake."
                )
                _emit(
                    "planner_decision",
                    decisions=[],
                    next_batch=[],
                    strategy_notes="Planner LLM call failed, will retry",
                    workerCount=MAX_CONCURRENT_WORKERS,
                    batch_id="",
                )
                expand_actions = []
                expand_requests = []
                accepted_prune_count = 0
                feedback_parts: list[str] = []
                # LLM call failure is not the planner's fault — keep stale
                # feedback so the planner still sees it on the next wake.
            else:
                accepted_prune_count = 0
                feedback_parts = []

                if planner_decision.parse_error:
                    feedback_parts.append(
                        "Your previous response could not be parsed as valid JSON. "
                        "Ensure your response contains ONLY a valid JSON object "
                        "(no markdown code fences, no extra text outside the JSON braces)."
                    )

                expand_actions = planner_decision.expand_actions
                expand_requests = [
                    a.node_id if a.node_id else f"option_id:{a.option_id}"
                    for a in expand_actions
                ]

                if planner_decision.is_empty:
                    feedback_parts.append(
                        "Your response contained no valid actions. "
                        'Each action must have "action" set to "expand" and include '
                        "a Legal Expand Options option_id."
                    )

                _emit(
                    "planner_decision",
                    decisions=[
                        {
                            "action": a.action,
                            "node_id": a.node_id,
                            **(
                                {"option_id": a.option_id}
                                if getattr(a, "option_id", None) is not None
                                else {}
                            ),
                        }
                        for a in planner_decision.actions
                    ],
                    next_batch=expand_requests,
                    strategy_notes=planner_decision.strategy_notes,
                    workerCount=MAX_CONCURRENT_WORKERS,
                    batch_id="",
                    reasoning=planner_decision.reasoning,
                )

                accepted_prune_count = 0
                if locked_primary_node_id:
                    advanced_primary = _advance_locked_primary(
                        locked_primary_node_id
                    )
                    if advanced_primary != locked_primary_node_id:
                        logger.info(
                            "Advancing primary node after planner actions %s -> %s",
                            locked_primary_node_id,
                            advanced_primary,
                        )
                        locked_primary_node_id = advanced_primary
                logger.info(
                    "Planner done: %d actions (expand=%d)",
                    len(planner_decision.actions),
                    len(expand_actions),
                )

            # Final tree_update after planner decisions (if any) are applied.
        _prune_too_deep_nodes()
        evaluate_solved_paths(
            state,
            time.time() - start_time,
            session_id=session_id,
            max_materialized_paths=materialized_path_limit,
        )
        tree_dict = extract_prediction_tree(state, score_mode=False)
        paths_detail = serialize_solved_paths_detail(state)
        _emit("tree_update", tree=tree_dict, turns=state.total_steps)

        if on_step:
            on_step(
                {
                    "turn": turn,
                    "tree": tree_dict,
                    "solved_paths": state.solved_path_count,
                    "solved_paths_detail": paths_detail,
                }
            )

        _prune_too_deep_nodes()
        recompute_solved_states(state)
        should_stop_flag, reason = _should_terminate(paths_detail)
        if should_stop_flag:
            termination_reason = reason
            break

        # Dispatch only planner-selected options; invalid choices are checked
        # against the same legal-action list shown in the planner prompt.
        dispatched_count = 0
        selected_expand_ids: list[str] = []
        if wake_planner and expand_actions:
            if should_stop and should_stop():
                termination_reason = "cancelled"
                logger.info(
                    "Cancellation requested, stopping before expand dispatch"
                )
                break
            seen_smiles: set[str] = set()
            blocked_roots = _diversity_blocked_roots()
            legal_options = _legal_expand_options()
            legal_by_id = {str(opt["node_id"]): opt for opt in legal_options}
            legal_by_option_id = {idx: opt for idx, opt in enumerate(legal_options)}
            dispatch_node_ids: list[str] = []
            for action in expand_actions:
                option = None
                if getattr(action, "option_id", None) is not None:
                    option = legal_by_option_id.get(action.option_id)
                    if option is None:
                        feedback_parts.append(
                            f"expand option_id {action.option_id} was refused: "
                            "option_id is not in the current Legal Expand Options. "
                            f"{_legal_expand_feedback(legal_options)}"
                        )
                        continue
                else:
                    node_id = action.node_id
                    option = legal_by_id.get(node_id)
                    if option is None:
                        node = state.get_node(node_id)
                        if node is None:
                            reason = "node was not found in the current tree"
                        elif not state.is_unsolved(node_id):
                            reason = "node is no longer available (solved, failed, or pruned)"
                        else:
                            reason = "node is not legal for the current primary branch"
                        feedback_parts.append(
                            f"expand node \"{node_id}\" was refused: {reason}. "
                            "For expand, return option_id from Legal Expand Options. "
                            f"{_legal_expand_feedback(legal_options)}"
                        )
                        continue
                selected_expand_ids.append(str(option["node_id"]))
                dispatch_node_ids.append(str(option["dispatch_node_id"]))
            for node_id in dispatch_node_ids:
                if not _available_slots:
                    break
                root_id = _root_branch_id(node_id)
                if (
                    locked_primary_node_id
                    and root_id
                    and not _is_descendant_or_self(node_id, locked_primary_node_id)
                ):
                    logger.info(
                        "Skip %s: outside locked primary node %s",
                        node_id,
                        locked_primary_node_id,
                    )
                    feedback_parts.append(
                        f"expand node \"{node_id}\" was refused: "
                        f"outside locked primary branch \"{locked_primary_node_id}\". "
                        f"Select a descendant of the locked primary."
                    )
                    continue
                if root_id in blocked_roots:
                    logger.info(
                        "Skip %s: root branch %s already has solved path(s); preserving diversity",
                        node_id,
                        root_id,
                    )
                    feedback_parts.append(
                        f"expand node \"{node_id}\" was refused: "
                        f"root branch \"{root_id}\" already has a solved path. "
                        f"Select a node from an unsolved root branch."
                    )
                    continue
                node = state.get_node(node_id)
                if node is None:
                    feedback_parts.append(
                        f"expand node \"{node_id}\" was refused: "
                        f"node was not found in the current tree."
                    )
                    continue
                canon = canonicalize_smiles(node.smiles)
                if canon in seen_smiles:
                    continue
                seen_smiles.add(canon)
                if _queue_worker(
                    "q",
                    node_id,
                    strategy_notes=planner_decision.strategy_notes,
                ):
                    dispatched_count += 1

        if wake_planner:
            if dispatched_count > 0 or accepted_prune_count > 0:
                planner_no_dispatch_count = 0
            elif not _pending_coros:
                legal_options = _legal_expand_options()
                if legal_options and _available_slots:
                    fallback = legal_options[0]
                    fallback_option_id = 0
                    fallback_action_id = str(fallback["node_id"])
                    fallback_node_id = str(fallback["dispatch_node_id"])
                    logger.warning(
                        "Planner made no dispatch progress; fallback dispatch %s via legal option_id %d (node_id %s)",
                        fallback_node_id,
                        fallback_option_id,
                        fallback_action_id,
                    )
                    feedback_parts.append(
                        "No worker was dispatched from your previous decision. "
                        f"Dispatcher used legal fallback expand option_id {fallback_option_id} "
                        f"(node_id \"{fallback_action_id}\"). "
                        f"{_legal_expand_feedback(legal_options)}"
                    )
                    fallback_request = f"option_id:{fallback_option_id}"
                    if fallback_request not in expand_requests:
                        expand_requests.append(fallback_request)
                    if fallback_action_id not in selected_expand_ids:
                        selected_expand_ids.append(fallback_action_id)
                    if _queue_worker(
                        "fallback",
                        fallback_node_id,
                        strategy_notes=planner_decision.strategy_notes,
                    ):
                        dispatched_count += 1
                        planner_no_dispatch_count = 0
                    else:
                        planner_no_dispatch_count += 1
                else:
                    planner_no_dispatch_count += 1

                if planner_no_dispatch_count:
                    logger.warning(
                        "Planner made no dispatch progress (%d/%d)",
                        planner_no_dispatch_count,
                        max_planner_no_dispatch,
                    )
                if planner_no_dispatch_count >= max_planner_no_dispatch:
                    termination_reason = "planner_no_progress"
                    logger.warning(
                        "Terminating: planner made no dispatch progress after %d attempts",
                        planner_no_dispatch_count,
                    )
                    break

        # Build feedback for the next planner round — runs regardless of
        # whether expand action selection was empty (parse errors, prune+expand
        # conflicts, and other pre-dispatch rejections also need feedback).
        if wake_planner and not planner_decision.cancelled:
            if feedback_parts:
                lines = [
                    "Your previous decision was REFUSED for the following reasons:"
                ]
                for part in feedback_parts:
                    lines.append(f"  - {part}")
                lines.append(
                    "Please correct these issues in your response below."
                )
                last_planner_feedback = "\n".join(lines)
                logger.info(
                    "Planner feedback for next round (%d reason(s)): %s",
                    len(feedback_parts),
                    last_planner_feedback[:300],
                )
            else:
                last_planner_feedback = None

        # Lock primary AFTER dispatch so all Planner-selected nodes are
        # eligible (not filtered by a premature primary lock).
        next_primary_node_id = _primary_node_from_expands(
            selected_expand_ids,
            locked_primary_node_id,
        )
        if next_primary_node_id and next_primary_node_id != locked_primary_node_id:
            logger.info(
                "Locked primary node: %s -> %s",
                locked_primary_node_id,
                next_primary_node_id,
            )
            locked_primary_node_id = next_primary_node_id

    # Cancel any still-running Workers
    _prune_too_deep_nodes()
    for coro in _pending_coros:
        coro.cancel()

    full_tree_dict = extract_prediction_tree(state, score_mode=False)
    paths_detail = serialize_solved_paths_detail(state)

    return {
        "success": state.solved_path_count > 0,
        "termination_reason": termination_reason,
        "turns": state.total_steps,
        "tree": full_tree_dict,
        "solved_paths": len(paths_detail),
        "full_solved_paths": state.solved_path_count,
        "solved_paths_detail": paths_detail,
        "run_stats_first": first_path_stats,
        "run_stats": {
            "total_steps": state.total_steps,
            "first_path_elapsed_sec": first_path_elapsed_sec,
            "first_path_stats": first_path_stats,
        },
    }
