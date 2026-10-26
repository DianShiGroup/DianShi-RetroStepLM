from runtime.aot.aot_planner import build_planner_prompt, build_primary_branch_context
from runtime.aot.aot_state import expand_node, init_tree
from runtime.aot.aot_worker import ScoredCandidate


def _expanded_root():
    state = init_tree("CC")
    expand_node(
        state,
        "0",
        [
            ScoredCandidate(
                index=0,
                reactants=[["O", "unavailable"], ["N", "unavailable"]],
            )
        ],
    )
    return state


def test_primary_context_describes_first_reactant_and_filters_legal_options():
    state = _expanded_root()
    context = build_primary_branch_context(state, "0")
    assert "first unsolved reactant" in context
    assert "All unsolved reactants" not in context

    filtered_context = build_primary_branch_context(
        state,
        "0",
        blocked_root_ids={"0-0"},
    )
    assert "  none" in filtered_context
    assert "has 1 reaction options" not in filtered_context

    messages = build_planner_prompt(
        state,
        active_workers=[],
        max_depth=5,
        primary_node_id="0",
        blocked_root_ids={"0-0"},
    )
    assert filtered_context in messages[1]["content"]
