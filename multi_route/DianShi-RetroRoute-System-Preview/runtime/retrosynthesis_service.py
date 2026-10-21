import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from runtime.planner_models import (
    default_planner_api_model,
    default_planner_model_id,
    resolve_llm_config,
)
from runtime.shared.llm_client import LlmCancelledError, LlmClient, LlmError
from runtime.step_api_client import (
    StepAPIClient,
    build_embedded_step_api_client,
    extract_reasoning_from_message,
    normalize_single_step_backend,
)
from runtime.aot import run_colony_loop
from runtime.errors import ServiceExhaustedError

logger = logging.getLogger(__name__)

_PLANNER_PRUNE_PROMPT_PLACEHOLDER = "{{PLANNER_PRUNE_PROMPT}}"


def _render_planner_system_prompt(prompt_text: str) -> str:
    """Strip legacy optional prompt sections and normalize whitespace."""
    return prompt_text.replace(_PLANNER_PRUNE_PROMPT_PLACEHOLDER, "").strip()


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def validate_smiles(smiles: str) -> tuple[bool, str]:
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES"
        return True, "Valid"
    except Exception:
        return False, "Invalid SMILES"


def build_external_llm_overrides(
    custom_config: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build per-request Planner LLM settings."""
    config = config or {}
    planner_cfg = config.get("planner", {}) or {}
    return {
        "base_url": str(custom_config["base_url"]).rstrip("/"),
        "api_key": str(custom_config["api_key"]),
        "model": str(custom_config["model_name"]),
        "temperature": float(custom_config.get("temperature", 0.0)),
        "disable_temperature": bool(custom_config.get("disable_temperature", False)),
        "max_tokens": int(custom_config.get("max_tokens", 32768)),
        "timeout": float(custom_config.get("timeout", 300.0)),
        "extra_body": dict(custom_config.get("extra_body") or {}),
        "planner_retry_count": int(planner_cfg.get("retry_count", 4)),
        "planner_retry_delay": float(planner_cfg.get("retry_delay", 1.0)),
        "planner_retry_backoff": float(planner_cfg.get("retry_backoff", 2.0)),
    }


def run_retrosynthesis(
    config: Dict[str, Any],
    smiles: str,
    max_steps: int,
    max_tree_depth: int,
    planner_api_model: Optional[str] = None,
    planner_model_id: Optional[str] = None,
    planner_llm_overrides: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    target_complete_paths: int = 5,
    session_id: str = "",
    no_limits: bool = False,
    step_api_client: Optional[StepAPIClient] = None,
    reach_max_depth_prune: Optional[bool] = None,
) -> Dict[str, Any]:
    session_id = session_id or ""
    started_at = datetime.now(timezone.utc)
    started_at_iso = started_at.isoformat()

    if no_limits:
        max_steps = float("inf")
        max_tree_depth = float("inf")
        target_complete_paths = float("inf")

    def emit(event: Dict[str, Any]) -> None:
        if progress_callback:
            progress_callback(event)

    emit(
        {
            "turn": 0,
            "status": "initializing",
            "message": "Initializing AND-OR Tree retrosynthesis...",
        }
    )

    single_step_config = config.get("single_step", {})
    search_cfg = config.get("search", {}) or {}
    reach_max_depth_prune = (
        bool(search_cfg.get("reach_max_depth_prune", False))
        if reach_max_depth_prune is None
        else bool(reach_max_depth_prune)
    )

    retro_config = single_step_config.get("retro", {})

    step_client = step_api_client or build_embedded_step_api_client(config)

    logger.info(
        "StepAPIClient ready | backend=%s endpoints=%s",
        normalize_single_step_backend(retro_config.get("backend", "dianshi")),
        retro_config.get("api_url") or retro_config.get("api_urls"),
    )

    starting_mols = step_client.starting_mols

    if planner_llm_overrides:
        provider_cfg = planner_llm_overrides
        planner_model_name = str(planner_llm_overrides["model"])
    else:
        resolved_model_id = planner_model_id or default_planner_model_id(config)
        provider_cfg = resolve_llm_config(config, resolved_model_id)
        planner_model_name = planner_api_model or default_planner_api_model(config)

    planner_cfg = config.get("planner", {}) or {}

    total_prompt_tokens = 0
    total_completion_tokens = 0
    llm_call_count = 0

    client = LlmClient(
        base_url=provider_cfg.get("base_url", ""),
        api_key=provider_cfg.get("api_key", ""),
        timeout=provider_cfg.get("timeout"),
    )
    planner_retry_count = int(
        provider_cfg.get("planner_retry_count", planner_cfg.get("retry_count", 4))
    )
    planner_retry_delay = float(
        provider_cfg.get("planner_retry_delay", planner_cfg.get("retry_delay", 1.0))
    )
    planner_retry_backoff = float(
        provider_cfg.get(
            "planner_retry_backoff", planner_cfg.get("retry_backoff", 2.0)
        )
    )

    def llm_stats_snapshot() -> dict[str, int]:
        return {
            "llm_calls": llm_call_count,
            "llm_prompt_tokens": total_prompt_tokens,
            "llm_completion_tokens": total_completion_tokens,
        }

    def call_llm(messages: List[Dict[str, str]]) -> Optional[tuple[str, str]]:
        nonlocal total_prompt_tokens, total_completion_tokens, llm_call_count
        try:
            response = client.chat(
                model=planner_model_name,
                messages=messages,
                temperature=provider_cfg.get("temperature", 0.0),
                disable_temperature=provider_cfg.get("disable_temperature", False),
                max_tokens=provider_cfg.get("max_tokens", 4096),
                extra_body=provider_cfg.get("extra_body"),
                max_attempts=planner_retry_count + 1,
                retry_delay=planner_retry_delay,
                retry_backoff=planner_retry_backoff,
                should_stop=should_stop,
            )
        except LlmCancelledError:
            return None
        except LlmError as exc:
            logger.error(
                "Planner LLM failed | model=%s attempts=%d error=%s",
                planner_model_name,
                exc.attempts,
                exc,
            )
            return None

        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens
        llm_call_count += 1
        reasoning_text = extract_reasoning_from_message(response.message, response.content)
        logger.info(
            "Planner LLM | prompt_tok=%d completion_tok=%d attempts=%d total=%.2fs",
            response.prompt_tokens,
            response.completion_tokens,
            response.attempts,
            response.elapsed_sec,
        )
        return response.content, reasoning_text

    async def single_step(mol_smiles: str) -> List:
        t0 = time.time()
        single_step._last_result = None
        try:
            result = await step_client.execute_async({"molecule": mol_smiles})
            if not isinstance(result, dict) or "results" not in result:
                logger.warning(
                    "Retro predict | smiles=%s api_time=%.2fs result_type=%s result_keys=%s",
                    mol_smiles[:30],
                    time.time() - t0,
                    type(result).__name__,
                    list(result.keys()) if isinstance(result, dict) else "N/A",
                )
                return []

            candidates = result["results"]
            raw_count = len(candidates)
            logger.info(
                "Retro predict | smiles=%s api_time=%.2fs retro_raw=%d",
                mol_smiles[:30],
                time.time() - t0,
                raw_count,
            )

            single_step._last_result = {
                "raw_count": raw_count,
                "filtered_count": len(candidates),
            }

            return candidates
        except asyncio.CancelledError:
            raise
        except ServiceExhaustedError:
            # Infrastructure failure (timeout / unreachable / HTTP error after
            # retries): must fail the whole case, not degrade to an empty
            # expansion. See runtime/errors.py.
            raise
        except Exception as e:
            logger.error(
                "Step API single-step failed | smiles=%.60s error=%s",
                mol_smiles,
                e,
                exc_info=True,
            )
            single_step._last_result = None
            return []

    emit(
        {
            "turn": 0,
            "status": "calling_llm",
            "message": "Starting AND-OR Tree exploration...",
        }
    )

    logger.info(
        "colony | starting retrosynthesis | max_steps=%d target_complete_paths=%d planner_model=%s single_step_backend=%s single_step_endpoints=%s",
        max_steps,
        target_complete_paths,
        planner_model_name or "Score Model",
        normalize_single_step_backend(retro_config.get("backend", "dianshi")),
        retro_config.get("api_url") or retro_config.get("api_urls"),
    )
    system_prompt_path = planner_cfg.get("system_prompt", "")
    system_prompt_text = ""
    if system_prompt_path:
        from pathlib import Path as _Path
        repo_root = _Path(__file__).resolve().parent.parent
        prompt_file = repo_root / system_prompt_path
        if prompt_file.is_file():
            system_prompt_text = _render_planner_system_prompt(
                prompt_file.read_text(),
            )
            logger.info(
                "Loaded planner system prompt from %s",
                prompt_file,
            )
        else:
            logger.warning("Planner system prompt not found at %s, using default", prompt_file)

    logger.info(
        "colony | entering run_colony_loop | target=%.60s",
        smiles,
    )
    t0_colony = time.time()
    worker_timeout = float(single_step_config.get("timeout", 300))
    colony_result = asyncio.run(
        run_colony_loop(
            target=smiles,
            max_iter=max_steps,
            target_complete_paths=target_complete_paths,
            max_depth=max_tree_depth,
            call_llm_fn=call_llm,
            single_step_fn=single_step,
            starting_mols=starting_mols,
            on_step=None,
            emit=emit,
            should_stop=should_stop,
            system_prompt_text=system_prompt_text,
            stats_snapshot_fn=llm_stats_snapshot,
            reach_max_depth_prune=reach_max_depth_prune,
            session_id=session_id,
            worker_timeout=worker_timeout,
        )
    )
    logger.info(
        "colony | run_colony_loop finished | elapsed=%.2fs success=%s termination=%s",
        time.time() - t0_colony,
        colony_result.get("success"),
        colony_result.get("termination_reason"),
    )

    finished_at = datetime.now(timezone.utc)
    finished_at_iso = finished_at.isoformat()
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)
    duration_s = duration_ms / 1000.0

    if colony_result["termination_reason"] == "cancelled":
        final_status = "cancelled"
    elif colony_result["success"]:
        solved_paths = int(colony_result.get("full_solved_paths") or 0)
        final_status = (
            "completed"
            if solved_paths >= target_complete_paths
            else "partial_success"
        )
    else:
        final_status = "failed"

    colony_run_stats = colony_result.get("run_stats", {})
    first_path_stats = (
        colony_result.get("run_stats_first")
        or colony_run_stats.get("first_path_stats")
    )

    run_stats = {
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "duration_ms": duration_ms,
        "single_step_calls": colony_result["turns"],
        "llm_calls": llm_call_count,
        "llm_prompt_tokens": total_prompt_tokens,
        "llm_completion_tokens": total_completion_tokens,
        "first_path_time_sec": colony_run_stats.get("first_path_elapsed_sec"),
        "reach_max_depth_prune": reach_max_depth_prune,
    }

    run_stats_first = None
    if isinstance(first_path_stats, dict):
        first_elapsed = first_path_stats.get("elapsed_sec")
        try:
            first_duration_ms = int(float(first_elapsed) * 1000)
        except (TypeError, ValueError):
            first_duration_ms = None
        first_finished_at = (
            (started_at + timedelta(milliseconds=first_duration_ms)).isoformat()
            if first_duration_ms is not None
            else None
        )
        run_stats_first = {
            "started_at": started_at_iso,
            "finished_at": first_finished_at,
            "duration_ms": first_duration_ms,
            "single_step_calls": first_path_stats.get("single_step_calls"),
            "llm_calls": first_path_stats.get("llm_calls", 0),
            "llm_prompt_tokens": first_path_stats.get("llm_prompt_tokens", 0),
            "llm_completion_tokens": first_path_stats.get("llm_completion_tokens", 0),
            "first_path_time_sec": first_elapsed,
            "reach_max_depth_prune": reach_max_depth_prune,
        }

    result = {
        "success": colony_result["success"],
        "termination_reason": colony_result["termination_reason"],
        "smiles": smiles,
        "turns": colony_result["turns"],
        "tree": colony_result["tree"],
        "solved_paths": colony_result["solved_paths"],
        "full_solved_paths": colony_result.get("full_solved_paths"),
        "solved_paths_detail": colony_result["solved_paths_detail"],
        "run_stats": run_stats,
        "run_stats_first": run_stats_first,
    }

    if not colony_result["success"]:
        solved_paths = int(colony_result.get("full_solved_paths") or 0)
        result["error"] = (
            f"Loop ended with no solution: {colony_result['termination_reason']}; "
            f"{colony_result['turns']} single-step calls, "
            f"{solved_paths} solved paths"
        )

    emit(
        {
            "turn": colony_result["turns"],
            "status": "run_complete",
            "message": f"Run complete ({final_status}) — {colony_result['turns']} calls, {duration_s:.1f}s",
        }
    )
    emit(
        {
            "turn": colony_result["turns"],
            "status": final_status,
            "message": "Done",
            "result": result,
        }
    )
    return result
