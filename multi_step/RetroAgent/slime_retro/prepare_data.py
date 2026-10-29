"""
Prepare training data for slime retrosynthesis training.

This script loads target molecules, formats prompts using the Qwen3 chat
template with tool schemas, and saves the result as a parquet file.

Usage:
    # Training data (routes with depth > 4, as in the paper)
    python -m slime_retro.prepare_data depth_min=5 depth_max=30

    # Eval data
    python -m slime_retro.prepare_data output=datasets/slime_retro_eval_uspto190.parquet targets=uspto-190
"""

import json
import logging
import os
import random
from datetime import datetime, timezone

import chz
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from transformers import AutoTokenizer

from slime_retro.prompts import build_system_prompt, build_user_prompt
from data.loaders import load_target_data
from slime_retro.config import EXPAND_MODE

logger = logging.getLogger(__name__)


def get_tool_schemas(port=None):
    """Fetch tool schemas from a running tool server."""
    port = port or int(os.environ.get("RETRO_TOOL_SERVER_PORT", 8100))
    url = f"http://127.0.0.1:{port}/tool_schemas"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json()["schemas"]


def format_prompt(tokenizer, target_smiles, tool_schemas, max_route_depth=30):
    """Format a single prompt using the chat template with tools."""
    system_prompt = build_system_prompt(
        max_route_depth=max_route_depth,
        multi_expand=(EXPAND_MODE == "multi"),
    )
    user_prompt = build_user_prompt(target_smiles)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tool_schemas,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def build_and_save(targets, output, tokenizer, tool_schemas, args, extra_meta=None):
    """Format prompts, build parquet, and save."""
    random.seed(args.seed)
    random.shuffle(targets)

    # Log depth distribution
    depth_counts = {}
    for t in targets:
        d = t.get("depth")
        depth_counts[d] = depth_counts.get(d, 0) + 1
    logger.info(
        f"Depth distribution: {dict(sorted((k, v) for k, v in depth_counts.items() if k is not None))}"
    )

    # Format prompts
    logger.info(f"Formatting {len(targets)} prompts...")
    rows = []
    for i, datum in enumerate(targets):
        prompt_text = format_prompt(tokenizer, datum["target"], tool_schemas)
        label = {
            "target": datum["target"],
            "depth": datum.get("depth"),
            "dataset": args.targets,
        }
        rows.append({"prompt": prompt_text, "reward_model": json.dumps(label)})
        if (i + 1) % 500 == 0:
            logger.info(f"  Formatted {i + 1}/{len(targets)}")

    config = {
        "targets": args.targets,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "num_rows": len(rows),
        "depth_distribution": dict(sorted(depth_counts.items())),
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **(extra_meta or {}),
    }
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    metadata = table.schema.metadata or {}
    metadata[b"prepare_data_config"] = json.dumps(config).encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, output)
    logger.info(f"Saved {len(rows)} samples to {output}")


@chz.chz
class PrepareDataConfig:
    output: str = ""
    """Output parquet file path. Defaults to datasets/slime_retro_train.parquet."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    """Model name for tokenizer."""

    targets: str = "uspto-full"
    """Target source: uspto-full (training routes), or an eval set (uspto-190)."""

    depth_min: int = 5
    """Minimum route depth for training targets, inclusive (paper: depth > 4)."""

    depth_max: int = 30
    """Maximum route depth for training targets (inclusive)."""

    num_targets: int | None = None
    """Number of targets to process. None for no limit."""

    seed: int = 42
    """Random seed for shuffling targets."""


def main():
    args = chz.entrypoint(PrepareDataConfig)
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tool_schemas = get_tool_schemas()

    output = args.output or "datasets/slime_retro_train.parquet"
    targets = load_target_data(
        target_smiles=None,
        targets=args.targets,
        num_targets=args.num_targets,
        train_depth_min=args.depth_min,
        train_depth_max=args.depth_max,
    )
    logger.info(f"Loaded {len(targets)} targets")
    build_and_save(targets, output, tokenizer, tool_schemas, args)


if __name__ == "__main__":
    main()
