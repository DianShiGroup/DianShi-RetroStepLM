"""
AND-OR Tree (AOT) module for multi-path retrosynthesis planning.

This module implements parallel Planner + Worker agent multi-path exploration.
"""

from .aot_tree import TreeNode, NodeStatus, NodeType, AOTreeState
from .aot_state import (
    init_tree,
    expand_node,
    evaluate_solved_paths,
    get_expandable_nodes,
    prune_max_depth_nodes,
    should_terminate,
    extract_prediction_tree,
    mark_node_failed,
    mark_pruned,
    build_route_smiles_list,
)
from .aot_planner import extract_reasoning_text
from .aot_dispatcher import run_colony_loop
from .aot_worker import TaskRequest, TaskResult, ScoredCandidate

__all__ = [
    "TreeNode",
    "NodeStatus",
    "NodeType",
    "AOTreeState",
    "init_tree",
    "expand_node",
    "evaluate_solved_paths",
    "get_expandable_nodes",
    "prune_max_depth_nodes",
    "should_terminate",
    "extract_prediction_tree",
    "mark_node_failed",
    "mark_pruned",
    "build_route_smiles_list",
    "extract_reasoning_text",
    "run_colony_loop",
    "TaskRequest",
    "TaskResult",
    "ScoredCandidate",
]
