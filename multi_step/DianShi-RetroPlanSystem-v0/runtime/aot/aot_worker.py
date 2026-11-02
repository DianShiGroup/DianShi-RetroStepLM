"""Worker Agent for parallel retrosynthesis.

Each worker calls the embedded single-step model, cleans its candidates, and
passes every surviving candidate into the AND-OR tree. No second model judges
or reorders those candidates.
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .aot_worker_rules import clean_candidates
from ..errors import ServiceExhaustedError

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """A cleaned single-step candidate passed directly into the search tree.

    ``combined_score`` remains zero for compatibility with existing tree data;
    this branch never generates a quality score.
    """

    index: int
    reactants: list
    combined_score: float = 0.0
    feasibility: str = "UNJUDGED"
    rationale: str = ""
    score_source: str = "single_step"


@dataclass
class TaskRequest:
    task_id: str
    node_id: str
    molecule_smiles: str
    depth: int


@dataclass
class TaskResult:
    task_id: str
    node_id: str
    molecule_smiles: str
    status: str
    candidates: list[ScoredCandidate] = field(default_factory=list)
    score_gap: float = 0.0
    dead_end_signal: bool = False
    pipeline_stats: dict = field(default_factory=dict)


async def run_worker(
    task: TaskRequest,
    single_step_fn: Callable[[str], List],
    timeout: float = 300.0,
    should_stop: Optional[Callable[[], bool]] = None,
) -> TaskResult:
    result = TaskResult(
        task_id=task.task_id,
        node_id=task.node_id,
        molecule_smiles=task.molecule_smiles,
        status="success",
    )
    try:
        if should_stop and should_stop():
            result.status = "cancelled"
            return result
        t0_step = time.time()
        if inspect.iscoroutinefunction(single_step_fn):
            step_awaitable = single_step_fn(task.molecule_smiles)
        else:
            step_awaitable = asyncio.to_thread(
                single_step_fn, task.molecule_smiles
            )
        raw_candidates = await asyncio.wait_for(step_awaitable, timeout=timeout)
        logger.info(
            "Worker %s: single-step returned in %.2fs with %d candidates for %.60s",
            task.task_id,
            time.time() - t0_step,
            len(raw_candidates),
            task.molecule_smiles,
        )
    except asyncio.TimeoutError:
        # A single-step call exceeding the worker timeout means the service is
        # wedged/overloaded; fail the case instead of scoring this node as a
        # dead end (which would silently distort benchmark results).
        logger.error(
            "Worker %s: single-step timed out after %.0fs; failing case",
            task.task_id,
            timeout,
        )
        raise ServiceExhaustedError(
            "retro",
            "timeout",
            detail=f"worker timeout {timeout}s",
        )
    except ServiceExhaustedError:
        raise
    except Exception as exc:
        result.status = "all_failed"
        result.dead_end_signal = True
        logger.warning("Worker %s: single-step failed: %s", task.task_id, exc)
        return result

    if not raw_candidates:
        result.status = "all_failed"
        result.dead_end_signal = True
        return result

    raw_count = len(raw_candidates)
    cleaned = clean_candidates(raw_candidates)
    after_rules = len(cleaned)
    if not cleaned:
        result.status = "all_failed"
        result.dead_end_signal = True
        return result

    candidates = [
        ScoredCandidate(index=index, reactants=candidate)
        for index, candidate in enumerate(cleaned)
    ]
    result.candidates = candidates
    result.pipeline_stats = {
        "raw": raw_count,
        "after_rules": after_rules,
        "after_candidates": len(candidates),
    }
    return result
