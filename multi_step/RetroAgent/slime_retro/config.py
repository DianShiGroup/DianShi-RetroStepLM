"""
Centralized configuration for retrosynthesis agent training and evaluation.

All runtime-configurable constants are defined here and read from
environment variables where applicable. This module has no heavy
dependencies (stdlib only) so it can be imported by both
slime_retro/utils.py and slime_retro/generate.py.

Usage:
    from slime_retro.config import MAX_TURNS, REWARD_TURN_THRESHOLD, ...
"""

import os

# ============================================================================
# Episode configuration
#
# SGLang context = 65536 (set in the model script: qwen3_4b_instr_2507.sh)
# MAX_TOTAL_TOKENS must be < SGLang context - rollout-max-response-len (8192)
# If you change the SGLang context length, update MAX_TOTAL_TOKENS accordingly.
# ============================================================================
MAX_TURNS = 90
MAX_TOTAL_TOKENS = int(os.environ.get("RETRO_MAX_TOTAL_TOKENS", "56000"))
PREDICT_TEMPLATES_TOP_K = int(os.environ.get("RETRO_PREDICT_TEMPLATES_TOP_K", "20"))
# Expand interface: "single" or "multi" (ablated in the paper, RQ3).
# single: one prediction_id per call, returns a flat result.
# multi: comma-separated prediction_ids, returns {expansions: [...]}.
EXPAND_MODE = os.environ.get("RETRO_EXPAND_MODE", "multi")
RETURN_LOGPROB = True
# Early termination if no successful expand() in N consecutive turns.
# Set low (5) to quickly kill stuck rollouts (e.g., predict loops without expand).
MAX_CONSECUTIVE_FAILURES = 5

# ============================================================================
# Reward function
# Penalizes predict_templates calls (N), the budget metric we optimize for.
# Turn penalty disabled; MAX_CONSECUTIVE_FAILURES handles wasted turns.
# Retro-R1 comparison: they penalize all steps >40 at 0.02/step.
# With ~3 turns/expansion, N=20 ≈ 60 turns, comparable to their threshold.
# ============================================================================
REWARD_N_THRESHOLD = 20  # penalty-free predict_templates calls
REWARD_N_RATE = 0.02  # penalty per call above threshold
REWARD_N_CAP = 0.3  # max penalty from N (symmetric with depth cap)
REWARD_TURN_THRESHOLD = 999  # turn penalty disabled
REWARD_TURN_RATE = 0.0
REWARD_TURN_CAP = 0.0
REWARD_DEPTH_THRESHOLD = int(os.environ.get("RETRO_REWARD_DEPTH_THRESHOLD", "20"))
REWARD_DEPTH_RATE = float(os.environ.get("RETRO_REWARD_DEPTH_RATE", "0.02"))
REWARD_DEPTH_CAP = float(os.environ.get("RETRO_REWARD_DEPTH_CAP", "0.3"))
# Token length penalty: disabled by default (threshold=999999).
# Penalizing total trainable tokens is too coarse -- it punishes hard problems
# that need many turns, not verbose thinking. Per-turn thinking naturally
# decreases via RL + task reward (as shown by R1).
REWARD_TOKEN_THRESHOLD = int(os.environ.get("RETRO_REWARD_TOKEN_THRESHOLD", "999999"))
REWARD_TOKEN_RATE = float(os.environ.get("RETRO_REWARD_TOKEN_RATE", "0.0"))
REWARD_TOKEN_CAP = float(os.environ.get("RETRO_REWARD_TOKEN_CAP", "0.0"))
TOOL_FAILURE_PENALTY = -0.05  # per failed tool call (mild, allows exploration)
NO_TOOL_CALLS_PENALTY = -0.05  # per turn with no tool call (same as tool failure)
FORMAT_FAILURE_PENALTY = -1.5  # trajectory with zero valid tool calls (overthinking)

# ============================================================================
# Route depth (also used by tools/memory.py and slime_retro/prompts.py
# via their own defaults, keep in sync)
# ============================================================================
MAX_ROUTE_DEPTH = 30

# ============================================================================
# Chat template delimiters for multi-turn tool calling.
# Qwen3 uses ChatML (<|im_start|>/<|im_end|>).
# Tool responses must be wrapped in proper user turns so the model
# doesn't degenerate into generating <|im_start|> tokens mid-turn.
# ============================================================================
TURN_CLOSE = "\n<|im_start|>user\n"
TURN_REOPEN = "<|im_end|>\n<|im_start|>assistant\n"

# ============================================================================
# Tool server
# Runs on host (not in Docker). Port/instances set via env var or defaults.
# train.sh passes RETRO_TOOL_SERVER_INSTANCES=16 into Docker.
# ============================================================================
TOOL_SERVER_BASE_URL = "http://127.0.0.1"
TOOL_SERVER_BASE_PORT = int(os.environ.get("RETRO_TOOL_SERVER_PORT", 8100))
TOOL_SERVER_NUM_INSTANCES = int(os.environ.get("RETRO_TOOL_SERVER_INSTANCES", 16))
