"""
Iterative-planning evaluation for retrosynthesis (the single eval entry point).

For each target, runs rollouts sequentially until the search budget (unique
single-step model calls) is exhausted or a route is found. This follows the
iterative planning protocol of Retro-R1: each rollout starts with a fresh
prompt and graph, but cached predictions (molecules expanded in a prior
rollout) do not count toward the budget.

If num_exp > 1, the entire iterative process is repeated num_exp times per
target, and every metric is reported as mean +- std across experiments.

Two metrics are reported (both saved to summary.json), following Retro-R1:
  - Pass@1: fraction of targets solved on the first rollout (single trajectory),
    computed automatically on every run.
  - Success rate at budget N: fraction solved within N unique single-step model
    calls, at the configured milestones.
Setting max_rollouts=1 restricts the search to a single trajectory per
experiment (the budget curve then collapses onto Pass@1).

Multi-replica: pass a comma-separated list of SGLang URLs to `sglang_url`.
All (target, exp) work items go in one shared queue; each replica runs
`concurrency` workers that pull the next pending item when free, load-balancing
hard targets across all replicas. Resumable: per-target results are persisted
to `results/expN_tgtNNN.json` and reloaded on restart. A single URL is just a
one-replica pool.

Usage:
    # Single replica, budget-500 iterative eval at temperature 0.6
    python -m slime_retro.eval \\
        sglang_url=http://localhost:30000 \\
        targets=uspto-190 max_budget=500 num_exp=10 temperature=0.6

    # Multi-replica
    python -m slime_retro.eval \\
        sglang_url=http://localhost:30000,http://localhost:30001 \\
        targets=uspto-190 max_budget=500 num_exp=10

    # pass@1
    python -m slime_retro.eval sglang_url=http://localhost:30000 \\
        targets=uspto-190 max_rollouts=1 num_exp=10
"""

import asyncio
import json
import logging
import os
import statistics
from datetime import datetime

import chz
import httpx
from transformers import AutoTokenizer

from data.loaders import load_target_data
from slime_retro.eval_worker import eval_single_target
from slime_retro.transcript import process_target_rollouts
from tools.tool_client import ToolClient

logger = logging.getLogger(__name__)
client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=256), timeout=None, trust_env=False
)




def extract_expanded_smiles(log_dir, session_id):
    """Extract unique SMILES that had predict_templates called from a transcript."""
    pending_path = os.path.join(log_dir, "pending", f"{session_id}.json")
    if not os.path.exists(pending_path):
        return []
    with open(pending_path) as f:
        transcript = json.load(f)

    graph_state = transcript.get("graph_state", {})
    id_to_smiles = {}
    for node in graph_state.get("nodes", []):
        if node.get("type") == "molecule":
            id_to_smiles[node["id"]] = node["smiles"]

    seen = set()
    ordered = []
    for call in transcript.get("tool_calls_log", []):
        if call.get("name") != "predict_templates":
            continue
        mol_id = call.get("args", {}).get("molecule_id")
        smiles = id_to_smiles.get(mol_id)
        if smiles and smiles not in seen:
            seen.add(smiles)
            ordered.append(smiles)
    return ordered


async def run_iterative_target(
    target_smiles,
    target_index,
    exp_index,
    tokenizer,
    tool_client,
    planner_url,
    sampling_params,
    args,
    tool_schemas,
    max_budget,
    semaphore,
):
    """Run iterative planning for one target in one experiment.

    Returns (result_dict, list_of_session_ids).
    Resumable: if a finalized result file exists, load and return it without recomputing.
    """
    results_dir = os.path.join(args.log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"exp{exp_index}_tgt{target_index:03d}.json")
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                cached = json.load(f)
            # A result file in this method-specific directory is finalized for
            # this target/experiment, so resume skips it unconditionally -- even
            # if the endpoints changed, because endpoint changes do not invalidate
            # an already recorded result and failures are intentionally not retried.
            if (
                cached.get("target_index") == target_index
                and cached.get("exp_index") == exp_index
            ):
                logger.info(f"[{target_index:03d}] exp={exp_index} RESUMED from file")
                return cached
            logger.warning(
                f"[{target_index:03d}] exp={exp_index} result file is for a different "
                "target/exp index; recomputing"
            )
        except Exception:
            pass
    async with semaphore:
        seen_smiles = set()
        budget = 0
        solved = False
        solved_at_budget = None
        rollout_index = 0
        session_ids = []
        zero_budget_streak = 0

        max_rollouts = args.max_rollouts or float("inf")
        while budget < max_budget and not solved and rollout_index < max_rollouts:
            result = await eval_single_target(
                target_smiles=target_smiles,
                target_index=target_index,
                tokenizer=tokenizer,
                tool_client=tool_client,
                planner_url=planner_url,
                planner_backend=args.planner_backend,
                planner_model=args.planner_model,
                planner_api_key=args.planner_api_key,
                sampling_params=sampling_params,
                max_turns=args.max_turns,
                max_consecutive_failures=args.max_consecutive_failures,
                max_total_tokens=args.max_total_tokens,
                predict_templates_top_k=args.predict_templates_top_k,
                log_dir=args.log_dir,
                tool_schemas=tool_schemas,
            )
            session_ids.append(result["session_id"])

            expanded = extract_expanded_smiles(args.log_dir, result["session_id"])
            new_budget = 0
            for smiles in expanded:
                if smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    budget += 1
                    new_budget += 1

            if result["completed"]:
                solved = True
                solved_at_budget = budget

            rollout_index += 1
            logger.info(
                f"[{target_index:03d}] exp={exp_index} rollout={rollout_index} "
                f"budget={budget}/{max_budget} solved={solved}"
            )

            # Break if two consecutive rollouts added no new budget
            # (target likely has 0 predictions or search space exhausted)
            if new_budget == 0 and not solved:
                zero_budget_streak += 1
                if zero_budget_streak >= 2:
                    logger.info(
                        f"[{target_index:03d}] {zero_budget_streak} consecutive zero-budget rollouts, stopping"
                    )
                    break
            else:
                zero_budget_streak = 0

        result_obj = {
            "target_index": target_index,
            "exp_index": exp_index,
            "solved": solved,
            "solved_at_budget": solved_at_budget,
            "total_budget": budget,
            "num_rollouts": rollout_index,
            "session_ids": session_ids,
        }
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result_obj, f)
        os.replace(tmp_path, result_path)
        return result_obj


async def run_eval(args):
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(
        args.log_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tool_client = ToolClient(
        base_url=args.tool_server_url,
        base_port=args.tool_server_port,
        num_instances=args.tool_server_instances,
    )

    schemas_resp = await tool_client.get_tool_schemas()
    tool_schemas = schemas_resp["schemas"]
    logger.info(f"Fetched {len(tool_schemas)} tool schemas from tool server")

    all_targets = load_target_data(targets=args.targets, num_targets=None)
    if args.target_offset > 0:
        all_targets = all_targets[args.target_offset :]
    if args.num_targets is not None:
        all_targets = all_targets[: args.num_targets]
    targets = all_targets
    logger.info(
        f"Loaded {len(targets)} targets from {args.targets} "
        f"(offset={args.target_offset}, num_targets={args.num_targets})"
    )

    with open(os.path.join(args.log_dir, "eval_config.json"), "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items() if not k.startswith("_")},
            f,
            indent=2,
        )

    sampling_params = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_response_len,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    n_targets = len(targets)
    milestones = sorted(int(m) for m in args.milestones.split(","))

    # Multi-replica shared work-queue (producer-consumer pool).
    # All (target, exp) items go in one queue; each replica runs `concurrency`
    # workers that pull the next pending item when free. This load-balances
    # hard targets across ALL replicas instead of pinning them to one shard.
    if args.planner_backend == "vllm":
        replica_urls = [u.strip().rstrip("/") for u in args.planner_url.split(",") if u.strip()]
    elif args.planner_backend == "sglang":
        replica_urls = [f"{u.strip().rstrip('/')}/generate" for u in args.sglang_url.split(",") if u.strip()]
    else:
        raise ValueError(f"Unsupported planner_backend: {args.planner_backend}")
    if not replica_urls:
        raise ValueError(f"No {args.planner_backend} planner URLs were configured")
    logger.info(
        f"Planner pool: {len(replica_urls)} {args.planner_backend} URL(s) "
        f"x {args.concurrency} workers each"
    )

    work_queue = asyncio.Queue()
    for exp_idx in range(args.num_exp):
        for target_idx in range(n_targets):
            t = targets[target_idx]
            target_smiles = t.get("target") if isinstance(t, dict) else t
            work_queue.put_nowait((target_idx, exp_idx, target_smiles))
    total_items = work_queue.qsize()

    # dummy semaphore (each task runs under a worker slot already; no extra limit)
    noop_sem = asyncio.Semaphore(10**9)
    collected = []

    async def worker(replica_url):
        while True:
            try:
                target_idx, exp_idx, target_smiles = work_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                r = await run_iterative_target(
                    target_smiles=target_smiles,
                    target_index=target_idx,
                    exp_index=exp_idx,
                    tokenizer=tokenizer,
                    tool_client=tool_client,
                    planner_url=replica_url,
                    sampling_params=sampling_params,
                    args=args,
                    tool_schemas=tool_schemas,
                    max_budget=args.max_budget,
                    semaphore=noop_sem,
                )
                collected.append(r)
            except Exception as e:
                logger.error(f"[{target_idx:03d}] exp={exp_idx} worker error: {e}")
            finally:
                work_queue.task_done()

    workers = []
    for url in replica_urls:
        for _ in range(args.concurrency):
            workers.append(asyncio.create_task(worker(url)))

    logger.info(
        f"Running {total_items} iterative planning tasks "
        f"({n_targets} targets x {args.num_exp} experiments) over "
        f"{len(replica_urls)} planner replica(s) x {args.concurrency} workers, "
        f"budget={args.max_budget}"
    )
    await asyncio.gather(*workers)
    results = collected

    # Organize by experiment
    exp_results = [[] for _ in range(args.num_exp)]
    for r in results:
        exp_results[r["exp_index"]].append(r)

    # Process transcripts into per-rollout route.json + transcript.json
    for exp_idx in range(args.num_exp):
        for r in sorted(exp_results[exp_idx], key=lambda x: x["target_index"]):
            rollouts = []
            for sid in r["session_ids"]:
                path = os.path.join(args.log_dir, "pending", f"{sid}.json")
                tx = json.load(open(path)) if os.path.exists(path) else None
                if tx:
                    rollouts.append(tx)
            if rollouts:
                process_target_rollouts(
                    args.log_dir, "eval", exp_idx, r["target_index"], rollouts
                )

    # Compute pass@1 (solved on the first rollout) per experiment. This is the
    # paper's headline metric: a target counts iff it is solved in a single
    # trajectory (num_rollouts == 1).
    exp_pass1_rates = []
    for exp_idx in range(args.num_exp):
        pass1 = sum(
            1 for r in exp_results[exp_idx] if r["solved"] and r["num_rollouts"] == 1
        )
        exp_pass1_rates.append(pass1 / n_targets)

    # Compute success rate at each budget milestone per experiment.
    exp_milestone_rates = {m: [] for m in milestones}
    for exp_idx in range(args.num_exp):
        for m in milestones:
            solved = sum(
                1
                for r in exp_results[exp_idx]
                if r["solved"] and r["solved_at_budget"] <= m
            )
            exp_milestone_rates[m].append(solved / n_targets)

    def mean_std(rates):
        return statistics.mean(rates), (
            statistics.stdev(rates) if len(rates) > 1 else 0.0
        )

    # Report
    logger.info(f"{'=' * 70}")
    logger.info("Iterative Planning Evaluation")
    logger.info(f"{'=' * 70}")
    logger.info(
        f"Targets: {n_targets}, Experiments: {args.num_exp}, Budget: {args.max_budget}"
    )
    logger.info("")
    logger.info(f"  {'Metric':>8}  {'Mean':>8}  {'Std':>8}  {'Solved':>12}")
    logger.info(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 12}")
    pass1_mean, pass1_std = mean_std(exp_pass1_rates)
    logger.info(
        f"  {'Pass@1':>8} {pass1_mean:>7.1%}  {pass1_std:>7.1%}  "
        f"{pass1_mean * n_targets:.1f}/{n_targets}"
    )
    for m in milestones:
        mean_rate, std_rate = mean_std(exp_milestone_rates[m])
        logger.info(
            f"  N<={m:<5d} {mean_rate:>7.1%}  {std_rate:>7.1%}  "
            f"{mean_rate * n_targets:.1f}/{n_targets}"
        )

    # Save summary
    summary = {
        "n_targets": n_targets,
        "num_exp": args.num_exp,
        "max_budget": args.max_budget,
        "pass@1": {"mean": pass1_mean, "std": pass1_std},
        "milestones": {
            str(m): {
                "mean": statistics.mean(exp_milestone_rates[m]),
                "std": (
                    statistics.stdev(exp_milestone_rates[m])
                    if args.num_exp > 1
                    else 0.0
                ),
            }
            for m in milestones
        },
        "per_experiment": [
            [
                {
                    "target_index": r["target_index"],
                    "solved": r["solved"],
                    "solved_at_budget": r["solved_at_budget"],
                    "total_budget": r["total_budget"],
                    "num_rollouts": r["num_rollouts"],
                }
                for r in exp_results[exp_idx]
            ]
            for exp_idx in range(args.num_exp)
        ],
    }
    with open(os.path.join(args.log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to {os.path.join(args.log_dir, 'summary.json')}")

    await tool_client.close()
    await client.aclose()


@chz.chz
class IterativeEvalConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    """Model name for tokenizer."""

    sglang_url: str = "http://localhost:30000"
    """SGLang server URL(s). Comma-separated list for a multi-replica shared pool,
    e.g. http://localhost:30000,http://localhost:30001 -- work is load-balanced
    across all listed replicas via one shared queue."""

    planner_backend: str = "sglang"
    """Planner protocol: sglang (legacy) or vllm (OpenAI completions API)."""

    planner_url: str = ""
    """Comma-separated vLLM OpenAI base URL(s), each ending in /v1."""

    planner_model: str = "RetroAgent"
    """vLLM served model name."""

    planner_api_key: str = ""
    """Optional Bearer token for vLLM."""

    targets: str = "uspto-190"
    """Target dataset key or a one-SMILES-per-line text file."""

    num_targets: int | None = None
    """Number of targets to evaluate. None for all."""

    target_offset: int = 0
    """Skip this many targets from the beginning (for splitting across servers)."""

    num_exp: int = 10
    """Number of experiments per target. Each experiment is a full iterative
    planning run (multiple rollouts until budget exhausted or solved)."""

    max_budget: int = 500
    """Max unique single-step model calls per target per experiment."""

    max_rollouts: int = 0
    """Max rollouts per experiment. 0=unlimited (budget-limited).
    Set to 1 for pass@1 evaluation (single rollout per experiment)."""

    milestones: str = "50,100,200,300,400,500"
    """Budget milestones to report (comma-separated)."""

    concurrency: int = 64
    """Max concurrent iterative planning tasks (target x experiment pairs)."""

    temperature: float = 0.0
    """Sampling temperature. The reported trained-model numbers use greedy
    decoding (temperature 0); stochasticity across the independent rollouts comes
    from the shuffled prediction/frontier orderings in the tool server, not from
    LLM sampling. Use temperature 0.6 for the untrained zero-shot baseline, where
    greedy decoding produces degenerate outputs."""

    top_k: int = 20
    """Top-k sampling."""

    top_p: float = 1.0
    """Top-p sampling."""

    max_turns: int = 120
    """Max turns per rollout (higher than training=80 for extra eval exploration)."""

    max_consecutive_failures: int = 5
    """Max turns without a successful expand before terminating a rollout."""

    max_total_tokens: int = 120000
    """Max total tokens per rollout."""

    max_response_len: int = 8192
    """Max new tokens per generation."""

    predict_templates_top_k: int = 20
    """Number of predictions from predict_templates."""

    log_dir: str = "logs/slime_retro_eval"
    """Output log directory."""

    tool_server_url: str = "http://127.0.0.1"
    """Tool server base URL."""

    tool_server_port: int = 8100
    """Tool server base port."""

    tool_server_instances: int = 16
    """Number of tool server instances."""


def main():
    args = chz.entrypoint(IterativeEvalConfig)
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
