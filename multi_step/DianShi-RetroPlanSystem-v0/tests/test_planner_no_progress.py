from __future__ import annotations

import asyncio

from runtime.aot.aot_dispatcher import run_colony_loop
from runtime.aot.aot_planner import parse_planner_response
from runtime.aot.aot_state import expand_node, init_tree, prune_max_depth_nodes
from runtime.aot.aot_tree import NodeStatus
from runtime.aot.aot_worker import ScoredCandidate


def test_empty_planner_actions_fall_back_until_tree_is_explored():
    planner_calls = 0

    def call_llm(_messages):
        nonlocal planner_calls
        planner_calls += 1
        return "{}", ""

    def single_step(_smiles):
        return [[["CCO", "n/a"]]]

    result = asyncio.run(
        run_colony_loop(
            target="CCCO",
            max_iter=100,
            target_complete_paths=1,
            max_depth=5,
            call_llm_fn=call_llm,
            single_step_fn=single_step,
            starting_mols=set(),
        )
    )

    assert result["termination_reason"] == "all_explored_no_solution"
    assert result["turns"] == 2
    assert planner_calls == 1


def test_reach_max_depth_prune_removes_root_branch():
    def scored(idx, reactant):
        return ScoredCandidate(
            index=idx,
            reactants=[[reactant, "n/a"]],
            combined_score=1.0,
            feasibility="FEASIBLE",
            rationale="",
        )

    state = init_tree("CCCC", starting_mols=set())
    expand_node(
        state,
        "0",
        [
            scored(0, "CCC"),
            scored(1, "CCO"),
        ],
    )
    expand_node(state, "0-0-0", [scored(0, "CC")])

    pruned = prune_max_depth_nodes(state, 2, reach_max_depth_prune=True)

    assert pruned == ["0-0"]
    assert state.get_node("0-0").status == NodeStatus.PRUNED
    assert state.get_node("0-0-0") is None
    assert state.get_node("0-1").status != NodeStatus.PRUNED
    assert state.get_node("0-1-0").status == NodeStatus.UNSOLVED


def test_target_complete_paths_counts_concrete_aot_tree_routes():

    def call_llm(_messages):
        raise AssertionError(
            "planner should not run when solved tree has no expandable nodes"
        )

    def single_step(_smiles):
        return [
            [["Brc1ccccc1", "available"], ["OB(O)c1ccccc1", "available"]],
            [["Ic1ccccc1", "available"], ["OB(O)c1ccccc1", "available"]],
        ]

    result = asyncio.run(
        run_colony_loop(
            target="c1ccc(-c2ccccc2)cc1",
            max_iter=100,
            target_complete_paths=2,
            max_depth=5,
            call_llm_fn=call_llm,
            single_step_fn=single_step,
            starting_mols={"Brc1ccccc1", "Ic1ccccc1", "OB(O)c1ccccc1"},
        )
    )

    assert result["termination_reason"] == "max_paths_reached"
    assert result["success"] is True
    assert result["full_solved_paths"] == 2
    assert result["solved_paths"] == 2


def test_max_iter_reached_returns_success_when_any_route_exists():
    def call_llm(_messages):
        raise AssertionError("planner should not run after max_iter is reached")

    def single_step(_smiles):
        return [[["C", "available"]]]

    result = asyncio.run(
        run_colony_loop(
            target="CC",
            max_iter=1,
            target_complete_paths=5,
            max_depth=5,
            call_llm_fn=call_llm,
            single_step_fn=single_step,
            starting_mols={"C"},
        )
    )

    assert result["termination_reason"] == "max_iter_reached"
    assert result["success"] is True
    assert result["full_solved_paths"] == 1
    assert result["solved_paths"] == 1


def test_parse_planner_expand_option_id():
    decision = parse_planner_response(
        '{"actions":[{"action":"expand","option_id":0}],"strategy_notes":"pick first"}'
    )

    assert decision is not None
    assert len(decision.actions) == 1
    assert decision.actions[0].action == "expand"
    assert decision.actions[0].option_id == 0
    assert decision.actions[0].node_id == ""


def test_planner_expand_option_id_dispatches_legal_option():
    planner_prompts = []

    def call_llm(messages):
        planner_prompts.append(messages[-1]["content"])
        return (
            '{"actions":[{"action":"expand","option_id":0}],"strategy_notes":"first legal option"}',
            "",
        )

    def single_step(smiles):
        if smiles == "CCC":
            return [[["CC", "unavailable"]]]
        if smiles == "CC":
            return [[["C", "available"]]]
        return []

    result = asyncio.run(
        run_colony_loop(
            target="CCC",
            max_iter=100,
            target_complete_paths=1,
            max_depth=5,
            call_llm_fn=call_llm,
            single_step_fn=single_step,
            starting_mols={"C"},
        )
    )

    assert result["success"] is True
    assert result["termination_reason"] == "max_paths_reached"
    assert result["turns"] == 2
    assert len(planner_prompts) == 1
    assert "=== Legal Expand Options ===" in planner_prompts[0]
    assert "option_id=0" in planner_prompts[0]


def test_invalid_planner_expand_falls_back_to_legal_frontier():
    planner_prompts = []

    def call_llm(messages):
        planner_prompts.append(messages[-1]["content"])
        return (
            '{"actions":[{"action":"expand","node_id":"0-0-0-99"}],"strategy_notes":"bad id"}',
            "",
        )

    def single_step(smiles):
        if smiles == "CCC":
            return [[["CC", "unavailable"]]]
        if smiles == "CC":
            return [[["C", "available"]]]
        return []

    result = asyncio.run(
        run_colony_loop(
            target="CCC",
            max_iter=100,
            target_complete_paths=1,
            max_depth=5,
            call_llm_fn=call_llm,
            single_step_fn=single_step,
            starting_mols={"C"},
        )
    )

    assert result["success"] is True
    assert result["termination_reason"] == "max_paths_reached"
    assert result["turns"] == 2
    assert len(planner_prompts) == 1
    assert "=== Legal Expand Options ===" in planner_prompts[0]
    assert "0-0-0" in planner_prompts[0]
