from runtime.aot.aot_planner import build_planner_tree_context
from runtime.aot.aot_state import expand_node, init_tree
from runtime.aot.aot_tree import NodeType
from runtime.aot.aot_worker import ScoredCandidate


def test_each_cleaned_candidate_becomes_a_concrete_reaction_node():
    state = init_tree("c1ccc(-c2ccccc2)cc1", starting_mols=set())
    candidates = [
        ScoredCandidate(
            index=0,
            reactants=[["Brc1ccccc1", "unavailable"], ["OB(O)c1ccccc1", "unavailable"]],
        ),
        ScoredCandidate(
            index=1,
            reactants=[["Ic1ccccc1", "unavailable"], ["OB(O)c1ccccc1", "unavailable"]],
        ),
        ScoredCandidate(
            index=2,
            reactants=[["Clc1ccccc1", "unavailable"], ["OB(O)c1ccccc1", "unavailable"]],
        ),
    ]

    expand_node(state, "0", candidates)

    reactions = [node for node in state.root.children if node.node_type == NodeType.REACTION]
    assert [node.node_id for node in reactions] == ["0-0", "0-1", "0-2"]
    assert all(not node.candidate_pool for node in reactions)

    planner_view = build_planner_tree_context(state)
    assert "Each REACTION is one concrete single-step candidate." in planner_view
    assert "cluster_size" not in planner_view
    assert "abstract" not in planner_view.lower()
