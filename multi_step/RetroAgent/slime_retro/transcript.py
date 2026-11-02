"""
Per-episode transcript logger for retrosynthesis rollouts.

During a rollout, each generate()/eval call creates a TranscriptLogger that
saves per-target JSON to {log_dir}/pending/. After the rollout, the pending
JSONs are processed into per-rollout route.json + transcript.json artifacts.

Output structure:
    {log_dir}/
        {label}_{step}/
            target_{idx}/
                rollout_{idx}/
                    route.json        (solved route + diagnostics)
                    transcript.json   (full turn-by-turn episode)
        pending/                      (temporary per-target JSONs)

Visualization is off-pipeline: run analyzers/visualize.py on a route.json.
"""
import json
import logging
import os

from analyzers.route import (
    save_route_json,
    build_bb_stats_from_graph,
    build_error_summary_from_tool_calls,
)

logger = logging.getLogger(__name__)


class TranscriptLogger:
    """Collects per-turn data during an episode and saves it as JSON."""

    def __init__(self, target, session_id, log_dir):
        self.target = target
        self.session_id = session_id
        self.log_dir = log_dir
        self.turns = []
        self.header_info = {}
        self.summary_info = {}
        # Full accumulated response text (all model outputs + observations)
        self.response = None
        # Set by generate/eval before flush for route analysis
        self.graph_state = None
        self.tool_calls_log = []
        self.route_steps = []
        self.route_tree = None
        self.completed = False

    def flush(self):
        """Save transcript data as JSON to {log_dir}/pending/{session_id}.json."""
        pending_dir = os.path.join(self.log_dir, "pending")
        os.makedirs(pending_dir, exist_ok=True)
        path = os.path.join(pending_dir, f"{self.session_id}.json")
        data = {
            "target": self.target,
            "session_id": self.session_id,
            "header": self.header_info,
            "turns": self.turns,
            "summary": self.summary_info,
            "response": self.response,
            "graph_state": self.graph_state,
            "tool_calls_log": self.tool_calls_log,
            "route_steps": self.route_steps,
            "route_tree": self.route_tree,
            "completed": self.completed,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, default=str)


def process_target_rollouts(log_dir, label, step, target_idx, rollouts):
    """Write route.json + transcript.json for one target's rollouts.

    Args:
        log_dir: Base log directory.
        label: Label for the subdirectory (e.g. "step", "eval").
        step: Step/rollout index.
        target_idx: Numeric index of the target.
        rollouts: List of transcript dicts for this target (one per sample).
    """
    step_dir_name = f"{label}_{step:06d}"
    target_dir = f"target_{target_idx:03d}"
    pending_dir = os.path.join(log_dir, "pending")
    for rollout_idx, data in enumerate(rollouts):
        error_summary = build_error_summary_from_tool_calls(data["tool_calls_log"])
        bb_stats = build_bb_stats_from_graph(data["graph_state"], error_summary)
        rollout_dir = os.path.join(
            log_dir,
            step_dir_name,
            target_dir,
            f"rollout_{rollout_idx:03d}",
        )
        save_route_json(
            {
                "target_index": target_idx,
                "target_smiles": data["target"],
                "completed": data["completed"],
                "route_tree": data.get("route_tree"),
                "route_steps": data.get("route_steps"),
                "error_summary": error_summary,
                "bb_stats": bb_stats,
                "graph_state": data["graph_state"],
                "tool_calls": data["tool_calls_log"],
            },
            rollout_dir,
        )
        # Move the consumed pending JSON into the rollout directory.
        src = os.path.join(pending_dir, f"{data['session_id']}.json")
        if os.path.exists(src):
            os.rename(src, os.path.join(rollout_dir, "transcript.json"))


def flush_target_sessions(log_dir, label, step, target_sessions):
    """Process route.json + transcript.json per target from tracked session_ids.

    Args:
        log_dir: Base log directory containing pending/.
        label: Label for the subdirectory (e.g. "step", "eval").
        step: Step/rollout index for naming.
        target_sessions: Dict mapping target SMILES -> list of session_ids.

    Returns:
        List of all transcript dicts processed.
    """
    pending_dir = os.path.join(log_dir, "pending")
    all_transcripts = []
    for target_idx, (target, session_ids) in enumerate(target_sessions.items()):
        rollouts = []
        for sid in session_ids:
            path = os.path.join(pending_dir, f"{sid}.json")
            if os.path.exists(path):
                with open(path) as f:
                    rollouts.append(json.load(f))
        if rollouts:
            process_target_rollouts(log_dir, label, step, target_idx, rollouts)
            all_transcripts.extend(rollouts)
    logger.info(
        f"Flushed {len(target_sessions)} targets, {len(all_transcripts)} rollouts"
    )
    return all_transcripts
