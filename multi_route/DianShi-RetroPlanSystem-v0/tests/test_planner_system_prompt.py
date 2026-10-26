from pathlib import Path

from runtime.aot.aot_planner import build_planner_prompt
from runtime.aot.aot_tree import AOTreeState, NodeType, TreeNode
from runtime.retrosynthesis_service import _render_planner_system_prompt


_PROMPT_PATH = Path("runtime/prompts/planner_system_v2.md")


def test_system_prompt_has_no_placeholder_left():
    rendered = _render_planner_system_prompt(
        _PROMPT_PATH.read_text(encoding="utf-8"),
    )

    assert "{{PLANNER_PRUNE_PROMPT}}" not in rendered
    assert "Pruning Policy" not in rendered


def test_planner_request_offers_expand_actions_only():
    root = TreeNode(node_id="0", node_type=NodeType.ROOT, smiles="CC")
    state = AOTreeState(root=root, all_nodes={root.node_id: root})

    messages = build_planner_prompt(
        state,
        active_workers=[],
        max_depth=5,
    )

    assert 'action="prune"' not in messages[1]["content"]
    assert "- PRUNE:" not in messages[1]["content"]
    assert 'action="expand"' in messages[1]["content"]
