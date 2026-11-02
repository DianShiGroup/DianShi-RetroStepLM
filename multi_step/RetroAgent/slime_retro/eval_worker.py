"""
Per-episode eval engine for RetroAgent's iterative planning protocol.

Library module exposing eval_single_target(): one evaluation episode over an
SGLang or vLLM planner endpoint, with <tool_call> XML parsing, NO context
pruning, and identical tool execution. The iterative-planning driver in
slime_retro/eval.py orchestrates these episodes.

The tool server must be running separately (same as training).
"""

import asyncio
import logging
import time

import httpx

from slime_retro.prompts import build_system_prompt, build_user_prompt
from slime_retro.config import (
    EXPAND_MODE,
    NO_TOOL_CALLS_PENALTY,
    TOOL_FAILURE_PENALTY,
)
from slime_retro.transcript import TranscriptLogger
from slime_retro.utils import (
    NO_TOOL_CALLS_ERROR,
    build_observation,
    compute_reward,
    execute_tool_calls,
    parse_tool_calls,
)

logger = logging.getLogger(__name__)
client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=256), timeout=None, trust_env=False
)


async def post(url, payload, headers=None, max_retries=60):
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"POST {url} failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)


async def generate(
    planner_url,
    planner_backend,
    planner_model,
    planner_api_key,
    prompt,
    sampling_params,
):
    if planner_backend == "sglang":
        output = await post(
            planner_url,
            {"text": prompt, "sampling_params": sampling_params},
        )
        return output["text"], output["meta_info"]["finish_reason"]

    if planner_backend != "vllm":
        raise ValueError(f"Unsupported planner backend: {planner_backend}")

    headers = {}
    if planner_api_key:
        headers["Authorization"] = f"Bearer {planner_api_key}"
    output = await post(
        planner_url.rstrip("/") + "/completions",
        {
            "model": planner_model,
            "prompt": prompt,
            "temperature": sampling_params["temperature"],
            "top_k": sampling_params["top_k"],
            "top_p": sampling_params["top_p"],
            "max_tokens": sampling_params["max_new_tokens"],
            "skip_special_tokens": False,
        },
        headers=headers,
    )
    choices = output.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("vLLM response does not contain a completion choice")
    choice = choices[0]
    return str(choice.get("text", "")), choice.get("finish_reason")


async def eval_single_target(
    target_smiles,
    target_index,
    tokenizer,
    tool_client,
    planner_url,
    planner_backend,
    planner_model,
    planner_api_key,
    sampling_params,
    max_turns,
    max_consecutive_failures,
    max_total_tokens,
    predict_templates_top_k,
    log_dir,
    tool_schemas,
):
    """Run one eval episode, the same loop as generate.py without RL bookkeeping."""
    session_id = await tool_client.create_session(
        target_smiles,
        predict_templates_top_k=predict_templates_top_k,
    )
    tx = TranscriptLogger(target_smiles, session_id, log_dir)

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                multi_expand=(EXPAND_MODE == "multi"),
            ),
        },
        {"role": "user", "content": build_user_prompt(target_smiles)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tool_schemas,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tx.header_info = {
        "target": target_smiles,
        "prompt_tokens": len(prompt_token_ids),
        "initial_prompt": prompt_text,
    }

    response = ""
    response_token_ids = []
    completed = False
    turn = 0
    tool_penalty_sum = 0.0
    no_tool_calls_count = 0
    tool_error_count = 0
    tool_call_count = 0
    predict_templates_count = 0
    turns_since_last_expand = 0
    termination_reason = "max_turns"
    time_generation = 0.0
    time_tool_calls = 0.0
    t_start = time.time()

    for _ in range(max_turns):
        turn += 1
        total_tokens = len(prompt_token_ids) + len(response_token_ids)
        if total_tokens >= max_total_tokens:
            termination_reason = "token_limit"
            break

        t_gen = time.time()
        cur_response, finish_reason = await generate(
            planner_url=planner_url,
            planner_backend=planner_backend,
            planner_model=planner_model,
            planner_api_key=planner_api_key,
            prompt=prompt_text + response,
            sampling_params=sampling_params,
        )
        gen_elapsed = time.time() - t_gen
        time_generation += gen_elapsed

        cur_token_ids = tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_token_ids

        if (
            (isinstance(finish_reason, dict) and finish_reason.get("type") == "length")
            or finish_reason == "length"
        ):
            termination_reason = "length"
            tx.turns.append(
                {
                    "turn": turn,
                    "tool_calls": [],
                    "results": [],
                    "failed": True,
                    "gen_time": gen_elapsed,
                    "tokens": len(cur_token_ids),
                    "reward": 0,
                    "raw_response": cur_response,
                    "observation": None,
                    "finish_reason": finish_reason,
                    "length_terminated": True,
                }
            )
            break

        tool_calls = parse_tool_calls(cur_response)

        if not tool_calls:
            no_tool_calls_count += 1
            turns_since_last_expand += 1
            tool_penalty_sum += NO_TOOL_CALLS_PENALTY
            obs_text = NO_TOOL_CALLS_ERROR
            tx.turns.append(
                {
                    "turn": turn,
                    "tool_calls": [],
                    "results": [],
                    "no_tool_calls": True,
                    "failed": True,
                    "gen_time": gen_elapsed,
                    "tokens": len(cur_token_ids),
                    "reward": NO_TOOL_CALLS_PENALTY,
                    "raw_response": cur_response,
                    "observation": obs_text,
                    "finish_reason": finish_reason,
                }
            )
            if turns_since_last_expand >= max_consecutive_failures:
                termination_reason = "no_progress"
                break
            obs_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]
            response += obs_text
            response_token_ids += obs_ids
            continue

        tool_call_count += len(tool_calls)
        predict_templates_count += sum(
            1 for tc in tool_calls if tc.get("name") == "predict_templates"
        )
        t_tool = time.time()
        results, status = await execute_tool_calls(tool_client, session_id, tool_calls)
        route_completed = status["completed"]
        any_failed = status["any_failed"]
        dead_frontier = status["dead_frontier"]
        tool_elapsed = time.time() - t_tool
        time_tool_calls += tool_elapsed

        penalty = TOOL_FAILURE_PENALTY if any_failed else 0
        if any_failed:
            tool_error_count += 1
            tool_penalty_sum += penalty
        if status["has_successful_expand"]:
            turns_since_last_expand = 0
        else:
            turns_since_last_expand += 1

        obs_text = build_observation(results)
        tx.turns.append(
            {
                "turn": turn,
                "tool_calls": tool_calls,
                "results": results,
                "failed": any_failed,
                "gen_time": gen_elapsed,
                "tool_time": tool_elapsed,
                "tokens": len(cur_token_ids),
                "reward": penalty,
                "raw_response": cur_response,
                "observation": obs_text,
                "finish_reason": finish_reason,
            }
        )

        obs_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]
        response += obs_text
        response_token_ids += obs_ids

        if route_completed:
            completed = True
            termination_reason = "completed"
            break

        if dead_frontier:
            termination_reason = "dead_frontier"
            break

        if turns_since_last_expand >= max_consecutive_failures:
            termination_reason = "no_progress"
            break

    elapsed = time.time() - t_start
    total_tokens = len(prompt_token_ids) + len(response_token_ids)
    trainable_tokens = total_tokens - len(prompt_token_ids)

    # Get graph state before computing reward (need max_depth)
    resp = await tool_client.delete_session(session_id)
    graph_state = resp.get("graph_state", {})
    max_depth = max(
        (
            n.get("depth", 0)
            for n in graph_state.get("nodes", [])
            if n.get("type") == "molecule"
        ),
        default=0,
    )
    reward = compute_reward(
        completed, predict_templates_count, turn, tool_penalty_sum, max_depth,
        tool_call_count, trainable_tokens,
    )

    tx.summary_info = {
        "completed": completed,
        "reward": reward,
        "turns": turn,
        "termination": termination_reason,
        "tool_call_count": tool_call_count,
        "predict_templates_count": predict_templates_count,
        "tool_error_count": tool_error_count,
        "no_tool_calls_count": no_tool_calls_count,
        "tool_penalty_sum": tool_penalty_sum,
        "total_tokens": total_tokens,
        "prompt_tokens": len(prompt_token_ids),
        "trainable_tokens": total_tokens - len(prompt_token_ids),
        "elapsed": round(elapsed, 1),
        "time_generation": round(time_generation, 2),
        "time_tool_calls": round(time_tool_calls, 2),
        "time_total": round(elapsed, 2),
    }
    tx.completed = completed
    tx.response = response
    tx.graph_state = graph_state
    tx.tool_calls_log = resp.get("tool_calls_log", [])
    tx.route_steps = resp.get("route_steps", [])
    tx.route_tree = resp.get("route_tree")
    tx.flush()

    logger.info(
        f"[{target_index:03d}] {target_smiles[:40]} "
        f"completed={completed} termination={termination_reason} "
        f"turns={turn} reward={reward:.2f} elapsed={elapsed:.0f}s"
    )
    return {
        "target": target_smiles,
        "session_id": session_id,
        "completed": completed,
        "reward": reward,
        "turns": turn,
        "predict_templates_count": predict_templates_count,
        "termination": termination_reason,
        "depth": None,
    }
