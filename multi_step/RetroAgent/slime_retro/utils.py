"""
Shared utilities for retrosynthesis agent generation loop.

Used by both slime_retro/generate.py (training) and slime_retro/eval_worker.py (eval).
No SLIME or heavy dependencies, only stdlib + json/re + config.
"""

import json
import re

from slime_retro.config import (
    FORMAT_FAILURE_PENALTY,
    REWARD_N_CAP,
    REWARD_N_RATE,
    REWARD_N_THRESHOLD,
    REWARD_TURN_CAP,
    REWARD_TURN_RATE,
    REWARD_TURN_THRESHOLD,
    REWARD_DEPTH_CAP,
    REWARD_DEPTH_RATE,
    REWARD_DEPTH_THRESHOLD,
    REWARD_TOKEN_CAP,
    REWARD_TOKEN_RATE,
    REWARD_TOKEN_THRESHOLD,
    TURN_CLOSE,
    TURN_REOPEN,
)

NO_TOOL_CALLS_ERROR = (
    TURN_CLOSE + '<tool_response>\n{"error": "Invalid tool call format. '
    "Please use the format: \\n<tool_call>\\n"
    '{\\"name\\": \\"tool_name\\", \\"arguments\\": {\\"param\\": \\"value\\"}}\\n'
    '</tool_call>"}\n</tool_response>' + TURN_REOPEN
)


def parse_tool_calls(text):
    """Parse tool calls from model output using <tool_call> XML tags."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            call = json.loads(match)
            if isinstance(call, dict) and "name" in call:
                tool_calls.append(call)
        except json.JSONDecodeError:
            continue
    return tool_calls


def build_observation(results):
    """Format tool results with proper chat template turn boundaries.

    Closes the assistant turn, wraps results in a user turn, then reopens
    the assistant turn so the model sees the correct Qwen ChatML format:
        ...<tool_call>...</tool_call><|im_end|>
        <|im_start|>user
        <tool_response>...</tool_response><|im_end|>
        <|im_start|>assistant
    """
    parts = []
    for item in results:
        result = item.get("response", item) if isinstance(item, dict) else item
        result_json = json.dumps(result, ensure_ascii=False, default=str)
        parts.append(f"<tool_response>\n{result_json}\n</tool_response>")
    return TURN_CLOSE + "\n".join(parts) + TURN_REOPEN


async def execute_tool_calls(tool_client, session_id, tool_calls):
    """Execute parsed tool calls via tool_client.

    Returns:
        Tuple of (results, status) where status is a dict with keys:
        - completed: route solved
        - any_failed: at least one tool call failed
        - dead_frontier: all frontier molecules have 0 predictions (provably unsolvable)
        - has_successful_expand: at least one expand advanced the graph
    """
    results = []
    completed = False
    any_failed = False
    dead_frontier = False
    has_successful_expand = False

    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            results.append(
                {
                    "name": name,
                    "response": {"error": "arguments must be a dict"},
                }
            )
            any_failed = True
            continue
        response = await tool_client.invoke_tool(session_id, name, args)
        results.append({"name": name, "response": response})

        if "error" in response:
            any_failed = True
        elif "errors" in response and not response.get("expansions"):
            any_failed = True
        if isinstance(response, dict) and response.get("route_solved"):
            completed = True

        # Check for successful expand
        if name == "expand":
            if "error" not in response and response.get("expansions", [True]):
                has_successful_expand = True

        if isinstance(response, dict) and response.get("dead_frontier"):
            dead_frontier = True

    status = {
        "completed": completed,
        "any_failed": any_failed,
        "dead_frontier": dead_frontier,
        "has_successful_expand": has_successful_expand,
    }
    return results, status


def compute_reward(
    completed,
    predict_templates_count,
    turns,
    tool_penalty_sum,
    max_depth=0,
    tool_call_count=0,
    trainable_tokens=0,
):
    """Compute reward with penalties for search budget (N), turns, depth, and token length.

    Primary penalty on predict_templates calls (search efficiency).
    Depth penalty discourages deep exploration when shallow routes exist.
    Token penalty discourages verbose thinking (overthinking).
    Format penalty for trajectories that never produce a valid tool call.
    """
    if not completed:
        if tool_call_count == 0:
            return FORMAT_FAILURE_PENALTY
        return -1.0
    n_penalty = min(
        REWARD_N_CAP,
        REWARD_N_RATE * max(0, predict_templates_count - REWARD_N_THRESHOLD),
    )
    turn_penalty = min(
        REWARD_TURN_CAP, REWARD_TURN_RATE * max(0, turns - REWARD_TURN_THRESHOLD)
    )
    token_penalty = min(
        REWARD_TOKEN_CAP,
        REWARD_TOKEN_RATE * max(0, (trainable_tokens - REWARD_TOKEN_THRESHOLD) / 1000),
    )
    depth_penalty = min(
        REWARD_DEPTH_CAP,
        REWARD_DEPTH_RATE * max(0, max_depth - REWARD_DEPTH_THRESHOLD),
    )
    return max(
        -1.0,
        min(1.0, 1.0 - n_penalty - turn_penalty - depth_penalty - token_penalty + tool_penalty_sum),
    )
