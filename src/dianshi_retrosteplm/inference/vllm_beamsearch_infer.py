#!/usr/bin/env python3
"""Run ordered multi-GPU vLLM beam-search inference on a cached dataset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import ray


def split_batch(batch: list[dict], shard_count: int) -> list[list[dict]]:
    """Split a batch into contiguous shards while preserving input order."""
    quotient, remainder = divmod(len(batch), shard_count)
    shards = []
    start = 0
    for index in range(shard_count):
        size = quotient + (1 if index < remainder else 0)
        shards.append(batch[start : start + size])
        start += size
    return shards


@ray.remote(num_gpus=1)
class BeamSearchWorker:
    """One vLLM instance bound to one GPU by Ray."""

    def __init__(
        self,
        model_path: str,
        beam_width: int,
        max_tokens: int,
        worker_index: int,
        gpu_memory_utilization: float,
        no_thinking: bool,
    ) -> None:
        cache_root = Path(tempfile.gettempdir()) / f"vllm_beam_worker_{worker_index}"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_root)
        os.environ["TORCH_COMPILE_CACHE_DIR"] = str(
            cache_root / "vllm" / "torch_compile_cache"
        )
        Path(os.environ["TORCH_COMPILE_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        print(
            f"Worker {worker_index}: CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
        )

        from vllm import LLM
        from vllm.sampling_params import BeamSearchParams

        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self.params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=max_tokens,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.chat_template_kwargs = {"enable_thinking": False} if no_thinking else {}

    def generate(self, batch: list[dict]) -> list[list[tuple[str, float]]]:
        if not batch:
            return []
        prompts = [
            {
                "prompt": self.tokenizer.apply_chat_template(
                    item["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            }
            for item in batch
        ]
        outputs = self.llm.beam_search(prompts, params=self.params)
        return [
            [
                (
                    sequence.text,
                    float(sequence.cum_logprob)
                    if sequence.cum_logprob is not None
                    else 0.0,
                )
                for sequence in output.sequences
            ]
            for output in outputs
        ]


def dispatch(
    batch: list[dict], workers: list[Any]
) -> list[list[tuple[str, float]]]:
    """Dispatch ordered shards and merge their results in the original order."""
    shards = split_batch(batch, len(workers))
    pending = [
        (index, workers[index].generate.remote(shard))
        for index, shard in enumerate(shards)
        if shard
    ]
    completed = [(index, ray.get(reference)) for index, reference in pending]
    completed.sort(key=lambda item: item[0])
    return [result for _, shard_results in completed for result in shard_results]


def inference_prompt(messages: Any, row_index: int) -> list[dict[str, str]]:
    """Return only the system/user prefix, never a ground-truth assistant turn."""
    if not isinstance(messages, list):
        raise ValueError(f"Dataset row {row_index} has no messages list.")
    prompt = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user"} and isinstance(content, str):
            prompt.append({"role": role, "content": content})
        if role == "user":
            break
    if not prompt or prompt[-1].get("role") != "user":
        raise ValueError(f"Dataset row {row_index} has no usable user message.")
    return prompt


def flatten_metadata(row: dict) -> dict[str, str]:
    metadata = row.get("meta_info")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "reactant": str(metadata.get("reactant", "") or ""),
        "product": str(metadata.get("product", "") or ""),
        "hash_id": str(metadata.get("hash_id", "") or ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cached-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Pass enable_thinking=False to compatible chat templates.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing result file."
    )
    args = parser.parse_args()
    for name in ("num_gpus", "batch_size", "max_tokens", "beam_width"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not args.cached_data_path.is_dir():
        parser.error(f"cached dataset does not exist: {args.cached_data_path}")
    if args.output_path.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output_path}; use --overwrite")
    model_candidate = Path(args.model_path).expanduser()
    looks_local = (
        model_candidate.is_absolute()
        or args.model_path.startswith(("./", "../", "models/"))
    )
    if looks_local:
        if not model_candidate.is_dir():
            parser.error(f"local model directory does not exist: {args.model_path}")
        args.model_path = str(model_candidate.resolve())
    return args


def main() -> int:
    args = parse_args()
    try:
        import datasets
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("Install the dependencies in requirements.txt first.") from exc
    if not hasattr(datasets, "load_from_disk"):
        raise RuntimeError(
            "The Hugging Face 'datasets' package is unavailable. "
            "Install the dependencies in requirements.txt first."
        )

    dataset = datasets.load_from_disk(str(args.cached_data_path))
    if not len(dataset):
        raise ValueError(f"Cached dataset is empty: {args.cached_data_path}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True)
    workers = [
        BeamSearchWorker.remote(
            args.model_path,
            args.beam_width,
            args.max_tokens,
            index,
            args.gpu_memory_utilization,
            args.no_thinking,
        )
        for index in range(args.num_gpus)
    ]

    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for start in tqdm(
                range(0, len(dataset), args.batch_size),
                desc="Beam search",
                unit="batch",
            ):
                end = min(start + args.batch_size, len(dataset))
                rows = [dataset[index] for index in range(start, end)]
                input_batch = [
                    {"prompt": inference_prompt(row.get("messages"), start + offset)}
                    for offset, row in enumerate(rows)
                ]
                results = dispatch(input_batch, workers)
                if len(results) != len(rows):
                    raise RuntimeError(
                        f"Inference returned {len(results)} results for {len(rows)} rows."
                    )
                for row, prompt, beams in zip(rows, input_batch, results):
                    item: dict[str, Any] = {
                        "messages": prompt["prompt"]
                        + [{"role": "assistant", "content": beams}],
                        "path": str(row.get("path", "") or ""),
                        **flatten_metadata(row),
                    }
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                output.flush()
        temporary.replace(args.output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        ray.shutdown()

    print(f"Results: {args.output_path}")
    print(f"Samples: {len(dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
