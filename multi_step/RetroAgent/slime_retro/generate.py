"""
Custom generate and reward functions for retrosynthesis planning with SLIME.

This module provides:
- generate(): Multi-turn tool-calling generation loop for retrosynthesis
- reward_func(): Reward based on route completion quality

Reward and penalty behavior:
- No-tool-calls continuation with user message injection (-0.1 penalty)
- Tool failure penalty (-0.05 per failed call)
- Token limit and max turn termination
- Detailed per-turn logging for progress tracking

Supports off-policy training via SLIME's TIS (Truncated Importance Sampling)
when --use-tis and --use-rollout-logprobs flags are enabled.

Usage in SLIME training config:
    --custom-generate-function-path slime_retro.generate_retro.generate
    --custom-rm-path slime_retro.generate_retro.reward_func
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import wandb
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.types import Sample

# Insert project root at the front of sys.path so our `tools` package
# takes priority over /root/slime/tools/ in the SLIME Docker container.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from slime_retro import config, utils
from slime_retro.transcript import TranscriptLogger, flush_target_sessions
from tools.tool_client import ToolClient

logger = logging.getLogger(__name__)


def log_dataset_info():
    """Log prepare_data config for all dataset parquet files at startup."""
    paths = os.environ.get("RETRO_DATA_PATHS", "")
    for path in paths.split(","):
        path = path.strip()
        if not path:
            continue
        metadata = pq.read_schema(path).metadata or {}
        raw = metadata.get(b"prepare_data_config")
        if raw is None:
            logger.warning(f"Dataset {path}: no prepare_data_config in metadata")
            continue
        cfg = json.loads(raw.decode())
        logger.info(f"Dataset {path}: {json.dumps(cfg)}")


log_dataset_info()

RETRO_CONFIGS = {
    "max_turns": config.MAX_TURNS,
    "log_dir": os.environ["RETRO_LOG_DIR"],
    "tool_server_base_url": config.TOOL_SERVER_BASE_URL,
    "tool_server_base_port": config.TOOL_SERVER_BASE_PORT,
    "tool_server_num_instances": config.TOOL_SERVER_NUM_INSTANCES,
    "predict_templates_top_k": config.PREDICT_TEMPLATES_TOP_K,
    "max_total_tokens": config.MAX_TOTAL_TOKENS,
    "return_logprob": config.RETURN_LOGPROB,
}

tool_client = ToolClient(
    base_url=RETRO_CONFIGS["tool_server_base_url"],
    base_port=RETRO_CONFIGS["tool_server_base_port"],
    num_instances=RETRO_CONFIGS["tool_server_num_instances"],
)

wandb_metrics_defined = False

# Track concurrent generate() calls
active_generates = 0
total_generates = 0
batch_size_hint = 0

# Per-target session tracking for incremental flushing
# Maps target SMILES -> list of session_ids within the current rollout
target_sessions = {}


def postprocess_response(resp):
    """Truncate response after the last complete tool call.

    Only used when return_logprob=False to align text with tokenization.
    """
    last_end = resp.rfind("</tool_call>")
    if last_end != -1:
        return resp[: last_end + len("</tool_call>")]
    return resp


def inject_non_trainable(
    text, tokenizer, response_token_ids, loss_mask, rollout_log_probs
):
    """Tokenize text and append as non-trainable tokens."""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    response_token_ids += token_ids
    loss_mask += [0] * len(token_ids)
    if rollout_log_probs is not None:
        rollout_log_probs += [0.0] * len(token_ids)
    return text


async def generate(args, sample, sampling_params):
    """Multi-turn retrosynthesis planning with tool calling."""
    global active_generates, total_generates, batch_size_hint
    active_generates += 1
    total_generates += 1
    gen_id = total_generates
    batch_size_hint = max(batch_size_hint, active_generates)

    state = GenerateState(args)
    tokenizer = state.tokenizer
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    label = json.loads(sample.label)
    target = label["target"]
    sample_tag = target[:50]
    logger.info(
        f"[gen:{gen_id}] START {sample_tag} "
        f"(active={active_generates}/{batch_size_hint})"
    )
    session_id = await tool_client.create_session(
        target,
        predict_templates_top_k=RETRO_CONFIGS["predict_templates_top_k"],
    )
    tx = TranscriptLogger(target, session_id, RETRO_CONFIGS["log_dir"])

    prompt_text = sample.prompt
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tx.header_info = {
        "target": target,
        "session_id": session_id,
        "prompt_tokens": len(prompt_token_ids),
    }

    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if RETRO_CONFIGS["return_logprob"] else None

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

    for _ in range(RETRO_CONFIGS["max_turns"]):
        turn += 1

        # Check total token limit
        total_tokens = len(prompt_token_ids) + len(response_token_ids)
        if total_tokens >= RETRO_CONFIGS["max_total_tokens"]:
            termination_reason = "token_limit"
            break

        # Generate
        payload = {"text": prompt_text + response, "sampling_params": sampling_params}
        if RETRO_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        t_gen_start = time.time()
        output = await post(sglang_url, payload)
        gen_elapsed = time.time() - t_gen_start
        time_generation += gen_elapsed
        last_finish_reason = output["meta_info"]["finish_reason"]
        sample.update_from_meta_info(args, output["meta_info"])

        if last_finish_reason["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            await tool_client.delete_session(session_id)
            active_generates -= 1
            done = batch_size_hint - active_generates
            pct = done / batch_size_hint * 100 if batch_size_hint > 0 else 0
            logger.info(
                f"[gen:{gen_id}] ABORTED {sample_tag} ({done}/{batch_size_hint} {pct:.0f}%)"
            )
            return sample

        cur_response = output["text"]

        # Extract token IDs and log probs
        if RETRO_CONFIGS["return_logprob"]:
            logprobs = output["meta_info"]["output_token_logprobs"]
            cur_token_ids = [item[1] for item in logprobs]
            cur_log_probs = [item[0] for item in logprobs]
        else:
            cur_response = postprocess_response(cur_response)
            cur_token_ids = tokenizer(cur_response, add_special_tokens=False)[
                "input_ids"
            ]

        # Accumulate model-generated tokens (trainable, loss_mask=1)
        response += cur_response
        response_token_ids += cur_token_ids
        loss_mask += [1] * len(cur_token_ids)
        if RETRO_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_log_probs

        if last_finish_reason["type"] == "length":
            termination_reason = "length"
            break

        # Ensure assistant turn is closed before injecting observations.
        # Model normally generates <|im_end|> as stop token, but
        # postprocess_response (return_logprob=False) may strip it.
        if not response.endswith("<|im_end|>"):
            response += inject_non_trainable(
                "<|im_end|>",
                tokenizer,
                response_token_ids,
                loss_mask,
                rollout_log_probs,
            )

        # Parse tool calls
        tool_calls = utils.parse_tool_calls(cur_response)

        if not tool_calls:
            no_tool_calls_count += 1
            turns_since_last_expand += 1
            penalty = config.NO_TOOL_CALLS_PENALTY
            tool_penalty_sum += penalty
            tx.turns.append(
                {
                    "turn": turn,
                    "tool_calls": [],
                    "results": [],
                    "no_tool_calls": True,
                    "failed": True,
                    "model_text": cur_response.replace("<|im_end|>", "")
                    .replace("</think>", "")
                    .strip()
                    or None,
                    "gen_time": gen_elapsed,
                    "tokens": len(cur_token_ids),
                    "reward": penalty,
                }
            )
            if turns_since_last_expand >= config.MAX_CONSECUTIVE_FAILURES:
                termination_reason = "no_progress"
                break
            response += inject_non_trainable(
                utils.NO_TOOL_CALLS_ERROR,
                tokenizer,
                response_token_ids,
                loss_mask,
                rollout_log_probs,
            )
            continue

        # Execute tool calls
        tool_call_count += len(tool_calls)
        predict_templates_count += sum(
            1 for tc in tool_calls if tc.get("name") == "predict_templates"
        )
        t_tool_start = time.time()
        results, status = await utils.execute_tool_calls(
            tool_client, session_id, tool_calls
        )
        route_completed = status["completed"]
        any_failed = status["any_failed"]
        dead_frontier = status["dead_frontier"]
        tool_elapsed = time.time() - t_tool_start
        time_tool_calls += tool_elapsed
        sample.non_generation_time += tool_elapsed

        reasoning = (
            cur_response.split("<tool_call>")[0]
            .replace("<|im_end|>", "")
            .replace("</think>", "")
            .strip()
        )
        penalty = config.TOOL_FAILURE_PENALTY if any_failed else 0
        if any_failed:
            tool_error_count += 1
            tool_penalty_sum += penalty
        if status["has_successful_expand"]:
            turns_since_last_expand = 0
        else:
            turns_since_last_expand += 1
        tx.turns.append(
            {
                "turn": turn,
                "tool_calls": tool_calls,
                "results": results,
                "model_text": reasoning if reasoning else None,
                "failed": any_failed,
                "gen_time": gen_elapsed,
                "tool_time": tool_elapsed,
                "tokens": len(cur_token_ids),
                "reward": penalty,
            }
        )

        # Inject observation as non-trainable tokens
        obs_text = utils.build_observation(results)
        response += inject_non_trainable(
            obs_text,
            tokenizer,
            response_token_ids,
            loss_mask,
            rollout_log_probs,
        )

        if route_completed:
            completed = True
            termination_reason = "completed"
            break

        if dead_frontier:
            termination_reason = "dead_frontier"
            break

        if turns_since_last_expand >= config.MAX_CONSECUTIVE_FAILURES:
            termination_reason = "no_progress"
            break

    # Finalize sample
    elapsed = time.time() - t_start
    total_tokens = len(prompt_token_ids) + len(response_token_ids)
    trainable_tokens = sum(loss_mask)
    prompt_tokens = len(prompt_token_ids)

    # Get graph state and max_depth before building label
    resp = await tool_client.delete_session(session_id)
    graph_state = resp["graph_state"]
    max_depth = max(
        (
            n.get("depth", 0)
            for n in graph_state.get("nodes", [])
            if n.get("type") == "molecule"
        ),
        default=0,
    )

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    if RETRO_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    label.update(
        {
            "completed": completed,
            "turns": turn,
            "tool_penalty_sum": tool_penalty_sum,
            "no_tool_calls_count": no_tool_calls_count,
            "tool_error_count": tool_error_count,
            "tool_call_count": tool_call_count,
            "predict_templates_count": predict_templates_count,
            "max_depth": max_depth,
            "termination": termination_reason,
            "total_tokens": total_tokens,
            "trainable_tokens": trainable_tokens,
            "elapsed": round(elapsed, 1),
        }
    )
    sample.label = json.dumps(label)
    sample.metadata.update(
        {
            "time_generation": round(time_generation, 2),
            "time_tool_calls": round(time_tool_calls, 2),
            "time_total": round(elapsed, 2),
            "turns": turn,
            "completed": int(completed),
            "tool_call_count": tool_call_count,
            "predict_templates_count": predict_templates_count,
            "tool_error_count": tool_error_count,
            "no_tool_calls_count": no_tool_calls_count,
            "total_tokens": total_tokens,
            "trainable_tokens": trainable_tokens,
            "prompt_tokens": prompt_tokens,
            "observation_tokens": total_tokens - trainable_tokens - prompt_tokens,
        }
    )

    if completed:
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED
    reward = utils.compute_reward(
        completed,
        predict_templates_count,
        turn,
        tool_penalty_sum,
        max_depth,
        tool_call_count,
        trainable_tokens,
    )
    tx.summary_info = {**label, **sample.metadata, "reward": reward}
    tx.completed = completed
    tx.response = response
    tx.graph_state = graph_state
    tx.tool_calls_log = resp.get("tool_calls_log", [])
    tx.route_steps = resp.get("route_steps", [])
    tx.route_tree = resp.get("route_tree")
    tx.flush()
    target_sessions.setdefault(target, []).append(session_id)

    active_generates -= 1
    done = batch_size_hint - active_generates
    pct = done / batch_size_hint * 100 if batch_size_hint > 0 else 0
    logger.info(
        f"[gen:{gen_id}] END {sample_tag} "
        f"completed={completed} termination={termination_reason} "
        f"turns={turn} tc={tool_call_count} "
        f"tok={trainable_tokens}/{total_tokens} reward={reward:.2f} "
        f"elapsed={elapsed:.1f}s "
        f"({done}/{batch_size_hint} {pct:.0f}%)"
    )

    return sample


async def reward_func(args, sample, **kwargs):
    """SLIME reward callback."""
    label = sample.label
    if isinstance(label, str):
        label = json.loads(label)

    score = utils.compute_reward(
        label.get("completed", False),
        label.get("predict_templates_count", 0),
        label.get("turns", 0),
        label.get("tool_penalty_sum", 0.0),
        label.get("max_depth", 0),
        label.get("tool_call_count", 0),
        label.get("trainable_tokens", 0),
    )
    return {
        "score": score,
        "completed": float(label.get("completed", False)),
        "turns": float(label.get("turns", 0)),
    }


def compute_diagnostics(samples, group_size=1):
    """Compute reward stats, termination breakdown, and constant-reward group fraction.

    Returns a dict with: reward, completed_rate, n_completed, n_total,
    avg_turns, termination (dict), constant_reward_frac.
    """
    scores = []
    completed_list = []
    turns_list = []
    trainable_tokens_list = []
    termination_counts = {}
    n_format_failures = 0
    for s in samples:
        r = s.reward if isinstance(s.reward, dict) else {}
        scores.append(r.get("score", 0.0))
        completed_list.append(r.get("completed", 0.0))
        turns_list.append(r.get("turns", 0.0))
        label = json.loads(s.label) if isinstance(s.label, str) else s.label
        reason = label.get("termination", "unknown")
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
        trainable_tokens_list.append(label.get("trainable_tokens", 0))
        if label.get("tool_call_count", 1) == 0:
            n_format_failures += 1

    n_total = len(samples)

    # Constant-reward group fraction
    n_groups = n_total // group_size if group_size > 1 else 0
    n_constant = 0
    for g in range(n_groups):
        group_scores = [scores[g * group_size + i] for i in range(group_size)]
        if len(set(round(r, 4) for r in group_scores)) == 1:
            n_constant += 1

    return {
        "reward": np.mean(scores),
        "completed_rate": np.mean(completed_list),
        "n_completed": int(sum(completed_list)),
        "n_total": n_total,
        "avg_turns": np.mean(turns_list),
        "avg_trainable_tokens": np.mean(trainable_tokens_list),
        "n_format_failures": n_format_failures,
        "termination": termination_counts,
        "n_constant_groups": n_constant,
        "n_groups": n_groups,
        "constant_reward_frac": n_constant / n_groups if n_groups > 0 else 0,
    }


def format_diagnostics(d):
    """Format diagnostics dict into a log string."""
    term_str = " ".join(f"{k}={v}" for k, v in sorted(d["termination"].items()))
    parts = [
        f"reward={d['reward']:.3f}",
        f"completed={d['completed_rate']:.2%} ({d['n_completed']}/{d['n_total']})",
        f"avg_turns={d['avg_turns']:.1f}",
        f"avg_train_tok={d['avg_trainable_tokens']:.0f}",
    ]
    if d["n_format_failures"] > 0:
        parts.append(f"format_fail={d['n_format_failures']}")
    if d["n_groups"] > 0:
        parts.append(
            f"constant_groups={d['n_constant_groups']}/{d['n_groups']} ({d['constant_reward_frac']:.0%})"
        )
    parts.append(f"termination=[{term_str}]")
    return ", ".join(parts)


def diagnostics_to_metrics(d, prefix):
    """Convert diagnostics dict to flat wandb metrics with prefix."""
    n = d["n_total"]
    metrics = {
        f"{prefix}/completed_rate": d["completed_rate"],
        f"{prefix}/avg_turns": d["avg_turns"],
        f"{prefix}/avg_trainable_tokens": d["avg_trainable_tokens"],
        f"{prefix}/format_failure_rate": d["n_format_failures"] / n if n > 0 else 0,
        f"{prefix}/constant_reward_frac": d["constant_reward_frac"],
    }
    for reason in (
        "max_turns",
        "token_limit",
        "length",
        "no_progress",
        "dead_frontier",
    ):
        metrics[f"{prefix}/termination/{reason}"] = d["termination"].get(reason, 0) / n
    return metrics


def ensure_wandb_metrics():
    global wandb_metrics_defined
    if wandb_metrics_defined:
        return
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.config.update({"retro": RETRO_CONFIGS}, allow_val_change=True)
    wandb_metrics_defined = True


def log_rollout(rollout_id, args, samples, extra_metrics, rollout_time):
    """Custom rollout logging. Returns False so SLIME's default logging also runs."""
    global target_sessions
    ensure_wandb_metrics()
    d = compute_diagnostics(samples, group_size=args.n_samples_per_prompt)
    logger.info(
        f"rollout {rollout_id}: {format_diagnostics(d)}, time={rollout_time:.0f}s"
    )
    extra_metrics.update(diagnostics_to_metrics(d, "rollout"))
    extra_metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    flush_target_sessions(RETRO_CONFIGS["log_dir"], "step", rollout_id, target_sessions)
    target_sessions = {}
    return False


def log_eval(rollout_id, args, data, extra_metrics):
    """Custom eval logging. Returns False so SLIME's default logging also runs."""
    global target_sessions
    ensure_wandb_metrics()
    custom_metrics = {}
    for key, dataset_data in data.items():
        samples = dataset_data.get("samples", [])
        if not samples:
            continue
        d = compute_diagnostics(samples, group_size=args.n_samples_per_eval_prompt)
        logger.info(f"eval {rollout_id} [{key}]: {format_diagnostics(d)}")
        custom_metrics.update(diagnostics_to_metrics(d, f"eval/{key}"))

    custom_metrics["eval/step"] = compute_rollout_step(args, rollout_id)
    wandb.log(custom_metrics, commit=False)
    flush_target_sessions(RETRO_CONFIGS["log_dir"], "eval", rollout_id, target_sessions)
    target_sessions = {}
    return False
