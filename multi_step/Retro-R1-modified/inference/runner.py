#!/usr/bin/env python3
"""
API-based inference for Retro-R1 using the original framework and custom inference backends.

Tests SOTA models (GPT-4o, Claude, DeepSeek, etc.) on the retrosynthesis
task via OpenAI-compatible API. Uses vLLM for one-step retrosynthesis.

Architecture overview
---------------------
The Planner LLM acts as the planning agent; either vLLM or the local MLP serves
as the one-step retrosynthesis oracle:

  Planner LLM (via API)
    │  generates <tool_call>{"name":"single_step_retro","arguments":{"molecule":"0-0"}}</tool_call>
    ▼
  step()  ─── original agent_r1/tool/tool_env_retro_noback.py
    │  parses tool call, validates args, calls tool.execute()
    ▼
  single_step tool  ─── inference/step/{dianshi_direct,mlp}.py
    │  calls the selected one-step backend to predict reactants
    │  returns a ranked list of reactant sets with availability labels
    ▼
  observation text  (reaction state shown to LLM next turn)
    │
    └─► LLM selects a reaction via select_reaction tool
        └─► env advances synthesis tree state
            └─► loop until all molecules solved or max_turns reached

Usage:
    python cli.py \
        --data_file ./data/reaction_pathway_search/validation_retro_190.parquet \
        --llm configs/planner.yaml \
        --output_dir results/

The DianShi-Direct single-step backend defaults to ``configs/dianshi_direct.yaml`` in the
project root. Use ``--single_step_config`` to select another vLLM configuration;
pass ``--single_step`` only when selecting a local MLP checkpoint.
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from collections import deque

from agent_r1.tool.tool_env_retro_noback import ToolEnvRetroNoBack, step, process_chira
from agent_r1.src.reward_score.reaction_pathway_reward import compute_all_score
from inference.step.factory import get_tools
from .prompt import INSTRUCTION_FOLLOWING, build_txt_prompt as _build_txt_prompt
from .planner import OpenAICompatiblePlanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
from inference.support.config_loader import load_yaml

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_STEP_CONFIG = PROJECT_ROOT / "configs/dianshi_direct.yaml"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_txt_prompt(target: str, maxstep: int) -> List[Dict[str, str]]:
    """Build the original-style user prompt for a one-SMILES-per-line file."""
    return _build_txt_prompt(target, maxstep)


def load_txt_data(txt_path: str, maxstep: int = 30) -> List[Dict[str, Any]]:
    """Load one target SMILES per non-empty line from a TXT file."""
    items = []
    with open(txt_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            target = line.strip()
            if not target:
                continue
            items.append({
                "target": target,
                "prompt": build_txt_prompt(target, maxstep),
                "ground_truth": None,
                "data_source": "txt",
                "line_number": line_number,
            })
    return items


def load_validation_data(data_path: str, maxstep: int = 30) -> List[Dict[str, Any]]:
    """Load validation Parquet data or one-SMILES-per-line TXT data."""
    if os.path.splitext(data_path)[1].lower() == ".txt":
        return load_txt_data(data_path, maxstep=maxstep)

    df = pd.read_parquet(data_path)
    items = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        reward_model = row["reward_model"]
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)

        items.append({
            "target": row["target"],
            "prompt": prompt,
            "ground_truth": reward_model["ground_truth"],
            "data_source": row["data_source"],
        })
    return items


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """{tools_desc}

{instruction}"""


def build_system_message(env: ToolEnvRetroNoBack, maxstep: int) -> str:
    tools_desc = env.tools_format_func()
    instruction = INSTRUCTION_FOLLOWING.format(n=maxstep)
    return SYSTEM_TEMPLATE.format(tools_desc=tools_desc, instruction=instruction)


# ---------------------------------------------------------------------------
# Single-molecule evaluation
# ---------------------------------------------------------------------------

def is_solved(env: ToolEnvRetroNoBack) -> bool:
    """Check whether any state in unsolved_dict is fully resolved."""
    for value in env.unsolved_dict.values():
        if not value:  # empty list == all molecules solved
            return True
    return False


def extract_prediction_tree(env: ToolEnvRetroNoBack) -> Optional[Dict[str, Any]]:
    """Build a hierarchical synthesis tree for a target molecule.

    Returns a nested dict with alternating mol / reaction layers::

        {type: "mol", smiles, in_stock, children: [
            {type: "reaction", smiles: "product>>r1.r2", children: [
                {type: "mol", smiles: "r1", in_stock: true},
                {type: "mol", smiles: "r2", in_stock: false, children: [...]},
            ]}
        ]}

    For failed routes, the tree still shows the partial decomposition with
    unsolved leaf molecules appearing as ``in_stock: false`` without children.
    Returns ``None`` only when no decomposition was attempted at all.
    """
    if not env.state_relation[0]:
        return None

    # Prefer the solved state; fall back to the furthest state reached.
    terminal_state = None
    for sid, unsolved in env.unsolved_dict.items():
        if not unsolved:
            terminal_state = int(sid)
            break
    if terminal_state is None:
        terminal_state = env.current_state

    # Build adjacency list from state_relation
    adj: Dict[int, list] = {}
    for p, c, rxns in zip(
        env.state_relation[0], env.state_relation[1], env.state_relation[2]
    ):
        adj.setdefault(p, []).append((c, rxns))

    # BFS from state 0 to the terminal state
    queue = deque([(0, [])])
    visited = {0}
    path = None
    while queue:
        state, edges = queue.popleft()
        if state == terminal_state:
            path = edges
            break
        if state in adj:
            for to_state, rxns in adj[state]:
                if to_state not in visited:
                    visited.add(to_state)
                    queue.append((to_state, edges + [(state, to_state, rxns)]))

    if path is None:
        return None

    # Build decomposition map: mol_smiles -> reaction (list of [smiles, avail])
    # Register all chirality variants so lookup is robust.
    decompositions: Dict[str, list] = {}
    for from_state, _to_state, rxn_ids in path:
        rxn_id = rxn_ids[0]
        parts = rxn_id.split("-")
        sid, mol_idx, rxn_idx = parts[0], int(parts[1]), int(parts[2])
        mol_smiles = env.unsolved_dict[sid][mol_idx]
        mol_key = f"{sid}-{mol_idx}"
        reaction = env.reaction_dict[mol_key][rxn_idx]
        for variant in process_chira(mol_smiles):
            if variant not in decompositions:
                decompositions[variant] = reaction

    def _lookup(smiles: str):
        if smiles in decompositions:
            return decompositions[smiles]
        for variant in process_chira(smiles):
            if variant in decompositions:
                return decompositions[variant]
        return None

    seen: set = set()

    def _build(smiles: str, availability: str = "unavailable") -> Dict[str, Any]:
        in_stock = availability == "available"
        reaction = _lookup(smiles)

        if reaction is not None and smiles not in seen:
            seen.add(smiles)
            reactant_smiles = [r[0] for r in reaction]
            rxn_repr = f"{smiles}>>{'.'.join(reactant_smiles)}"
            rxn_node = {
                "type": "reaction",
                "smiles": rxn_repr,
                "children": [_build(r[0], r[1]) for r in reaction],
            }
            return {
                "type": "mol",
                "smiles": smiles,
                "in_stock": False,
                "children": [rxn_node],
            }

        node: Dict[str, Any] = {"type": "mol", "smiles": smiles, "in_stock": in_stock}
        return node

    return _build(env.target)


def run_single_molecule(
    planner: OpenAICompatiblePlanner,
    env_template: ToolEnvRetroNoBack,
    item: Dict[str, Any],
    maxstep: int,
    max_turns: int,
    save_trajectory: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Run the multi-turn retrosynthesis loop for one target molecule."""

    target = item["target"]
    ground_truth = item["ground_truth"]
    user_content = item["prompt"][0]["content"]

    env = env_template.copy(target)

    system_msg = build_system_message(env, maxstep)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    trajectory: List[Dict[str, str]] = []
    start_time = time.time()
    consecutive_ineffective_turns = 0
    max_consecutive_ineffective_turns = 10

    for turn in range(max_turns):
        # ---- Call the LLM ----
        try:
            assistant_content = planner.complete(messages)
        except Exception as exc:
            logger.error("Planner failed for %s: %s", target[:80], exc)
            break

        messages.append({"role": "assistant", "content": assistant_content})
        if save_trajectory:
            trajectory.append({"role": "assistant", "content": assistant_content})

        if verbose:
            preview = assistant_content.replace("\n", " ")[:200]
            logger.info("  [turn %d] assistant: %s", turn, preview)

        # If the model didn't produce a tool call, nudge it once then stop
        if "<tool_call>" not in assistant_content:
            consecutive_ineffective_turns += 1
            if consecutive_ineffective_turns >= max_consecutive_ineffective_turns:
                logger.warning(
                    "Stopping target %s after %d consecutive turns without an effective tool call",
                    target[:80],
                    consecutive_ineffective_turns,
                )
                break
            if not is_solved(env) and turn < max_turns - 1:
                nudge = (
                    "You have not called a tool yet. Please use "
                    "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> "
                    "to continue searching for the synthesis pathway."
                )
                messages.append({"role": "user", "content": nudge})
                if save_trajectory:
                    trajectory.append({"role": "user", "content": nudge})
                continue
            break

        # ---- Execute tool call via the vLLM one-step oracle ----
        # step() parses the <tool_call> from the LLM response and dispatches it:
        #
        #  • single_step_retro  →  the selected inference single-step tool
        #                          → Calls the MLP or vLLM backend
        #                          Returns a ranked list of reactant sets with
        #                          availability labels ("available"/"unavailable").
        #
        #  • select_reaction    →  SelectReactionTool.execute()
        #                          Pure state update — no model call.
        #                          Advances the synthesis tree and returns the
        #                          new "unsolved molecules" state to the LLM.
        observation, reward, done, info = step(env, assistant_content)

        if info.get("action_is_effective", False):
            consecutive_ineffective_turns = 0
        else:
            consecutive_ineffective_turns += 1

        tool_response_text = (
            f"<tool_response>\n{observation}\n</tool_response>"
        )
        messages.append({"role": "user", "content": tool_response_text})
        if save_trajectory:
            trajectory.append({"role": "user", "content": tool_response_text})

        if verbose:
            obs_preview = observation.replace("\n", " ")[:200]
            logger.info("  [turn %d] tool: %s", turn, obs_preview)

        # ---- Check termination ----
        if done:
            break
        if env.back_flag:
            break
        if is_solved(env):
            break
        if consecutive_ineffective_turns >= max_consecutive_ineffective_turns:
            logger.warning(
                "Stopping target %s after %d consecutive ineffective tool calls",
                target[:80],
                consecutive_ineffective_turns,
            )
            break

    elapsed = time.time() - start_time

    # ---- Compute scores ----
    score, end_score, answer_score, format_score = compute_all_score(
        "done", ground_truth, env
    )

    success = bool(end_score > 0)
    result: Dict[str, Any] = {
        "target": target,
        "success": success,
        "score": score,
        "end_score": end_score,
        "answer_score": answer_score,
        "format_score": format_score,
        "turns": env.steps_taken,
        "onestep_calls": env.count_onestep,
        "sibling_retries": env.count_sibling_retry,
        "elapsed_s": round(elapsed, 2),
    }
    try:
        tree = extract_prediction_tree(env)
        if tree is not None:
            result["prediction_tree"] = tree
    except Exception as e:
        logger.warning("extract_prediction_tree failed for %s: %s", target[:40], e)
    if save_trajectory:
        result["trajectory"] = trajectory

    return result


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_previous_results(streaming_jsonl: str) -> Dict[int, Dict[str, Any]]:
    """Read a streaming JSONL file and return a mapping of idx → result.

    Malformed lines are skipped with a warning so a partially-written last
    line (e.g. from a hard crash) never blocks a resume.
    """
    cache: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(streaming_jsonl):
        return cache
    with open(streaming_jsonl) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                idx = int(entry["idx"])
                res = {k: v for k, v in entry.items() if k != "idx"}
                cache[idx] = res
            except Exception as e:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, streaming_jsonl, e)
    return cache


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate(
    planner: OpenAICompatiblePlanner,
    env_template: ToolEnvRetroNoBack,
    items: List[Dict[str, Any]],
    maxstep: int,
    max_turns: int,
    workers: int,
    save_trajectory: bool,
    verbose: bool,
    streaming_jsonl: Optional[str] = None,
    cached: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a dataset of target molecules, optionally in parallel.

    If ``streaming_jsonl`` is given, each new result is appended to that file
    as it completes (crash-safe incremental saving), unordered.

    If ``cached`` is given (idx → result), those molecules are skipped and
    their stored results are used directly (resume support).
    """

    cached = cached or {}
    results: List[Dict[str, Any]] = []
    n = len(items)

    # Pre-load cached results so the final list is complete
    for idx, res in cached.items():
        results.append((idx, res))

    stream_f = None
    if streaming_jsonl:
        os.makedirs(os.path.dirname(streaming_jsonl) or ".", exist_ok=True)
        stream_f = open(streaming_jsonl, "a", buffering=1)  # line-buffered

    def _append_streaming(idx: int, res: Dict[str, Any]):
        if stream_f:
            stream_f.write(json.dumps({"idx": idx, **res}, ensure_ascii=False) + "\n")

    def _run(idx_item):
        idx, item = idx_item
        return idx, run_single_molecule(
            planner, env_template, item,
            maxstep, max_turns,
            save_trajectory, verbose,
        )

    # Only submit items that haven't been completed yet
    pending = [(i, item) for i, item in enumerate(items) if i not in cached]

    try:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run, idx_item): idx_item[0] for idx_item in pending}
                for future in as_completed(futures):
                    try:
                        idx, res = future.result()
                    except Exception as e:
                        orig_idx = futures[future]
                        logger.error("Worker failed on item %d: %s", orig_idx, e, exc_info=True)
                        continue
                    results.append((idx, res))
                    _append_streaming(idx, res)
                    done_count = len(results)
                    status = "OK" if res["success"] else "FAIL"
                    logger.info(
                        "[%d/%d] %s  target=%s  turns=%d  %.1fs",
                        done_count, n, status,
                        res["target"][:40], res["turns"], res["elapsed_s"],
                    )
        else:
            for i, item in pending:
                _, res = _run((i, item))
                results.append((i, res))
                _append_streaming(i, res)
                status = "OK" if res["success"] else "FAIL"
                logger.info(
                    "[%d/%d] %s  target=%s  turns=%d  %.1fs",
                    len(results), n, status,
                    res["target"][:40], res["turns"], res["elapsed_s"],
                )
    finally:
        if stream_f:
            stream_f.close()

    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {}
    successes = [r for r in results if r["success"]]
    return {
        "total": n,
        "success": len(successes),
        "success_rate": len(successes) / n,
        "avg_score": np.mean([r["score"] for r in results]),
        "avg_end_score": np.mean([r["end_score"] for r in results]),
        "avg_answer_score": np.mean([r["answer_score"] for r in results]),
        "avg_format_score": np.mean([r["format_score"] for r in results]),
        "avg_turns": np.mean([r["turns"] for r in results]),
        "avg_turns_success": (
            np.mean([r["turns"] for r in successes]) if successes else 0.0
        ),
        "avg_onestep_calls": np.mean([r["onestep_calls"] for r in results]),
        "avg_sibling_retries": np.mean([r.get("sibling_retries", 0) for r in results]),
        "avg_elapsed_s": np.mean([r["elapsed_s"] for r in results]),
    }


def print_report(metrics: Dict[str, float], model: str, data_file: str):
    print()
    print("=" * 64)
    print(f"  Retro-R1 API Inference — Results")
    print("=" * 64)
    print(f"  Model          : {model}")
    print(f"  Dataset        : {data_file}")
    print(f"  Total molecules: {int(metrics['total'])}")
    print("-" * 64)
    print(f"  Success rate   : {metrics['success_rate']:.1%}  "
          f"({int(metrics['success'])}/{int(metrics['total'])})")
    print(f"  Avg score      : {metrics['avg_score']:.4f}")
    print(f"  Avg end score  : {metrics['avg_end_score']:.4f}")
    print(f"  Avg answer sc. : {metrics['avg_answer_score']:.4f}")
    print(f"  Avg format sc. : {metrics['avg_format_score']:.4f}")
    print("-" * 64)
    print(f"  Avg turns      : {metrics['avg_turns']:.1f}")
    print(f"  Avg turns (OK) : {metrics['avg_turns_success']:.1f}")
    print(f"  Avg 1-step call: {metrics['avg_onestep_calls']:.1f}")
    print(f"  Avg sibling retry: {metrics['avg_sibling_retries']:.1f}")
    print(f"  Avg time / mol : {metrics['avg_elapsed_s']:.1f}s")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_llm_config(path: str) -> Dict[str, Any]:
    """Load and validate the external planner LLM configuration."""
    config = load_yaml(path) or {}

    if not isinstance(config, dict):
        raise ValueError(f"LLM config must be a YAML mapping: {path}")

    if not config.get("base_url"):
        raise ValueError(f"LLM config requires base_url: {path}")
    if not config.get("model_name"):
        raise ValueError(f"LLM config requires model_name: {path}")

    api_key_env = config.get("api_key_env")
    if api_key_env:
        config["api_key"] = os.environ.get(api_key_env, "")

    return config


def resolve_llm_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve planner settings, with explicit CLI values taking precedence."""
    config = load_llm_config(args.llm) if args.llm else {}

    args.model = args.model or config.get("model_name") or "gpt-4o"
    args.api_base = args.api_base or config.get("base_url")
    args.api_key = args.api_key or config.get("api_key")
    args.temperature = (
        args.temperature
        if args.temperature is not None
        else config.get("temperature", 0.0)
    )
    args.max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else config.get("max_tokens", 4096)
    )
    args.timeout = (
        args.timeout
        if args.timeout is not None
        else config.get("timeout")
    )
    return args


def load_single_step_config(path: str) -> Dict[str, Any]:
    """Load and validate a single-step backend configuration."""
    config = load_yaml(path) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Single-step config must be a YAML mapping: {path}")
    backend = config.get("backend", "dianshi_direct")
    if backend == "dianshi_direct":
        api_urls = config.get("api_urls", config.get("api_url"))
        if isinstance(api_urls, str):
            api_urls = [api_urls]
        if (
            not isinstance(api_urls, list)
            or not api_urls
            or any(not isinstance(url, str) or not url.strip() for url in api_urls)
        ):
            raise ValueError(f"Single-step config requires api_url or api_urls: {path}")
        config["api_urls"] = [url.strip() for url in api_urls]
    elif backend != "mlp":
        raise ValueError(f"Unsupported single-step backend {backend!r}: {path}")

    if not config.get("model_name"):
        raise ValueError(f"Single-step config requires model_name: {path}")
    config["backend"] = backend

    api_key_env = config.get("api_key_env")
    if api_key_env:
        config["api_key"] = os.environ.get(api_key_env, "")

    return config


def resolve_single_step_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve a vLLM config or a local MLP checkpoint."""
    config = {}
    args.single_step_backend = "dianshi_direct"
    args.mlp_model = None
    args.single_step_label = "dianshi_direct"

    if args.single_step and args.single_step_config:
        raise ValueError(
            "--single_step_config is only valid for the vLLM backend; "
            "do not combine it with a local --single_step checkpoint"
        )

    if args.single_step:
        suffix = os.path.splitext(args.single_step)[1].lower()
        if suffix in {".yaml", ".yml"}:
            raise ValueError(
                "Pass a vLLM YAML configuration with --single_step_config, "
                "not --single_step"
            )
        elif suffix == ".ckpt":
            args.single_step_backend = "mlp"
            args.mlp_model = args.single_step
            args.single_step_label = f"mlp_{os.path.splitext(os.path.basename(args.single_step))[0]}"
        else:
            raise ValueError(
                "--single_step must point to a .ckpt local MLP checkpoint"
            )
    else:
        args.single_step = args.single_step_config or str(DEFAULT_SINGLE_STEP_CONFIG)
        config = load_single_step_config(args.single_step)
        args.single_step_backend = config["backend"]

    if args.step_api_url:
        args.step_api_urls = [args.step_api_url]
    elif config.get("api_urls"):
        args.step_api_urls = list(config["api_urls"])
    else:
        args.step_api_urls = ["http://localhost:8000/v1/chat/completions"]
    args.step_api_url = args.step_api_urls[0]
    args.step_api_key = (
        args.step_api_key
        or config.get("api_key")
        or os.environ.get("DIANSHI_DIRECT_API_KEY")
    )
    args.step_model = args.step_model or config.get("model_name") or "default"
    args.step_temperature = (
        args.step_temperature
        if args.step_temperature is not None
        else config.get("temperature", 0.0)
    )
    args.step_max_tokens = (
        args.step_max_tokens
        if args.step_max_tokens is not None
        else config.get("max_tokens", 1024)
    )
    args.step_timeout = (
        args.step_timeout
        if args.step_timeout is not None
        else config.get("timeout", 120)
    )
    args.step_beam_width = (
        args.step_beam_width
        if args.step_beam_width is not None
        else config.get(
            "server_top_k",
            config.get("num_res", config.get("beam_width", 10)),
        )
    )
    args.step_atomwise = bool(config.get("atomwise", False))
    args.step_prompt_path_no_rag = config.get("prompt_path_no_rag")
    args.step_template_filter = config.get("template_filter") or {}
    search_config = config.get("search") or {}
    if not isinstance(search_config, dict):
        raise ValueError("Single-step config 'search' must be a mapping")
    args.sibling_retry = bool(search_config.get("sibling_retry", False))
    if config:
        args.single_step_label = f"{args.single_step_backend}_{args.step_model}"
    return args

def parse_args():
    p = argparse.ArgumentParser(
        description="Test SOTA LLMs on Retro-R1 retrosynthesis with API or local MLP one-step inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument(
        "--data_file", required=True,
        help="Path to validation parquet file or TXT with one SMILES per line",
    )
    p.add_argument("--max_samples", type=int, default=None,
                    help="Limit evaluation to first N molecules")

    # Planner LLM API (e.g., GPT-4o, Claude, etc.)
    p.add_argument("--llm", "--llm_config", dest="llm", default=None,
                    help="YAML config for the planner LLM")
    p.add_argument("--model", default=None,
                    help="Planner model name (OpenAI-compatible)")
    p.add_argument("--api_base", default=None,
                    help="API base URL (for non-OpenAI providers)")
    p.add_argument("--api_key", default=None,
                    help="API key (defaults to $OPENAI_API_KEY)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None,
                    help="Max tokens per planner LLM response")
    p.add_argument("--timeout", type=float, default=None,
                    help="Planner API request timeout in seconds")

    # Environment
    p.add_argument("--max_turns", type=int, default=100,
                    help="Max tool-calling turns per molecule")
    p.add_argument("--maxstep", type=int, default=30,
                    help="Max synthesis steps allowed")
    p.add_argument("--topk", type=int, default=10,
                    help="Top-k reactions from one-step model")
    p.add_argument("--shuffle", action="store_true",
                    help="Shuffle reaction order")
    p.add_argument("--force_noloop", action="store_true",
                    help="Prevent cycles in the search")

    # vLLM One-Step API options
    p.add_argument(
        "--single_step", dest="single_step", default=None,
        help=(
            "Local MLP .ckpt; omit this option to use the fixed project config "
            "configs/dianshi_direct.yaml"
        ),
    )
    p.add_argument(
        "--single_step_config", "--single-step-config",
        dest="single_step_config", default=None,
        help="Optional DianShi-Direct single-step YAML; ignored only when --single_step selects a local MLP checkpoint",
    )
    p.add_argument("--step_api_url", type=str, default=None,
                    help="URL for OpenAI-compatible endpoint for one-step retrosynthesis")
    p.add_argument("--step_api_key", type=str, default=None,
                    help="API key for the single-step vLLM endpoint")
    p.add_argument("--step_model", type=str, default=None,
                    help="Model name for vLLM API")
    p.add_argument("--step_temperature", type=float, default=None,
                    help="Temperature for vLLM sampling")
    p.add_argument("--step_max_tokens", type=int, default=None,
                    help="Max tokens for vLLM generation")
    p.add_argument("--step_timeout", type=int, default=None,
                    help="Timeout for single-step vLLM requests in seconds")
    p.add_argument("--step_beam_width", type=int, default=None,
                    help="Beam width for vLLM beam search (controls number of candidates)")
    p.add_argument("--starting_molecules", type=str, default="./dataset/origin_dict.csv",
                    help="Path to file containing available starting molecules (CSV or PKL)")

    # Execution
    p.add_argument("--workers", type=int, default=1,
                    help="Parallel API workers (>1 for concurrent calls)")
    p.add_argument("--save_trajectory", action="store_true",
                    help="Save full conversation trajectories")
    p.add_argument("--output_dir", default=None,
                    help="Directory for output files")
    p.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="JSONL",
        help=(
            "Resume from an existing JSONL; when used without a path, use the "
            "deterministic file under --output_dir"
        ),
    )
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    args = resolve_llm_args(args)
    args = resolve_single_step_args(args)

    # ---- Build tool environment ----
    # Creates the selected one-step retrosynthesis tool + selection tool
    logger.info("Loading one-step retrosynthesis tools...")
    tools = get_tools(
        backend=args.single_step_backend,
        step_api_url=args.step_api_url,
        step_api_urls=args.step_api_urls,
        model_name=args.step_model,
        topk=args.topk,
        temperature=args.step_temperature,
        max_tokens=args.step_max_tokens,
        beam_width=args.step_beam_width,
        starting_molecules=args.starting_molecules,
        api_key=args.step_api_key,
        timeout=args.step_timeout,
        mlp_model_dump=args.mlp_model,
        atomwise=args.step_atomwise,
        prompt_path_no_rag=args.step_prompt_path_no_rag,
        template_filter=getattr(args, "step_template_filter", None),
    )
    env_template = ToolEnvRetroNoBack(
        tools=tools,
        max_turns=args.max_turns,
        maxstep=args.maxstep,
        topk=args.topk,
        shuffle=args.shuffle,
        force_noloop=args.force_noloop,
        sibling_retry=args.sibling_retry,
    )
    # Log loaded tools info
    tool_names = [t.name for t in env_template.tools]
    logger.info(
        "Tools loaded: %s | Single-step: %s (topk=%d) | sibling_retry=%s%s",
        tool_names,
        args.single_step_label,
        args.topk,
        args.sibling_retry,
        " | rdchiral template_filter=on"
        if (getattr(args, "step_template_filter", None) or {}).get("enabled")
        else "",
    )

    # ---- Load data ----
    logger.info("Loading data from %s", args.data_file)
    items = load_validation_data(args.data_file, maxstep=args.maxstep)
    if args.max_samples:
        items = items[:args.max_samples]
    logger.info("Loaded %d molecules", len(items))

    # ---- Init Planner client ----
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error(
            "No API key provided. Set --api_key, use api_key/api_key_env in "
            "--llm, or set $OPENAI_API_KEY."
        )
        sys.exit(1)
    planner = OpenAICompatiblePlanner(
        model=args.model,
        api_key=api_key,
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout or 600.0,
    )
    if args.llm:
        logger.info("Loaded planner LLM config from %s", args.llm)
    if args.single_step:
        logger.info("Loaded single-step input from %s", args.single_step)
    logger.info("API client ready (model=%s)", args.model)

    # ---- Derive output paths ----
    # Filename pattern: {dataset}_t{temperature}_{single_step}
    dataset_name = os.path.splitext(os.path.basename(args.data_file))[0]
    # Strip common validation_ prefix for brevity
    dataset_name = dataset_name.replace("validation_", "")
    run_stem = f"{dataset_name}_t{args.temperature}_{args.single_step_label}"

    streaming_jsonl: Optional[str] = None
    ordered_jsonl: Optional[str] = None
    if args.resume:
        streaming_jsonl = os.path.abspath(args.resume)
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        streaming_jsonl = os.path.join(args.output_dir, f"{run_stem}.jsonl")
    elif args.resume is not None:
        logger.error("--resume without a path requires --output_dir")
        sys.exit(2)

    if streaming_jsonl:
        if args.resume is None and os.path.exists(streaming_jsonl):
            logger.error(
                "Output file already exists: %s. Use --resume to skip completed cases.",
                streaming_jsonl,
            )
            sys.exit(2)
        ordered_jsonl = os.path.splitext(streaming_jsonl)[0] + "_ordered.jsonl"
        logger.info("Streaming results  → %s", streaming_jsonl)
        logger.info("Ordered results    → %s", ordered_jsonl)

    # ---- Resume: load previously completed results if specified ----
    cached: Dict[int, Dict[str, Any]] = {}
    resume_file = streaming_jsonl if args.resume is not None else None
    if resume_file:
        if os.path.exists(resume_file):
            cached = load_previous_results(resume_file)
        if cached:
            logger.info(
                "Resuming: loaded %d/%d previously completed results from %s",
                len(cached), len(items), resume_file,
            )
            for idx in sorted(cached):
                res = cached[idx]
                status = "OK" if res.get("success") else "FAIL"
                logger.info(
                    "  [cached %d] %s  target=%s  turns=%s",
                    idx, status, str(res.get("target", ""))[:40], res.get("turns", "?"),
                )
        elif os.path.exists(resume_file):
            logger.warning("No valid results found in resume file, starting fresh.")
        else:
            logger.info("Resume file does not exist yet; starting a fresh run: %s", resume_file)
    else:
        logger.info("Starting fresh run (no resume file specified).")

    # ---- Run evaluation ----
    results = evaluate(
        planner=planner,
        env_template=env_template,
        items=items,
        maxstep=args.maxstep,
        max_turns=args.max_turns,
        workers=args.workers,
        save_trajectory=args.save_trajectory,
        verbose=args.verbose,
        streaming_jsonl=streaming_jsonl,
        cached=cached,
    )

    # ---- Metrics ----
    metrics = compute_metrics(results)
    print_report(metrics, args.model, args.data_file)

    # ---- Save ordered results ----
    if ordered_jsonl:
        config = {k: v for k, v in vars(args).items() if k != "api_key"}
        header = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        with open(ordered_jsonl, "w") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        logger.info("Ordered results saved to %s", ordered_jsonl)


if __name__ == "__main__":
    main()
