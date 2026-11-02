# slime Training Scripts

These scripts drive RL training with the slime framework (Megatron + SGLang) inside the `slimerl/slime:v0.3.0` Docker
image. The user-facing entry points are `scripts/train.sh` and `scripts/eval.sh` at the repo root; the scripts here are
the underlying Docker launcher and its per-model configs.

## Prerequisites

1. Download the `datasets/` files (see the top-level `README.md` for the per-file sources):
   ```bash
   python -m scripts.prepare_datasets
   ```

2. Start the tool server on the host (outside Docker):
   ```bash
   python -m tools.tool_server base_port=8100 num_instances=16
   ```

3. Build the prompt parquets (requires the running tool server for tool schemas):
   ```bash
   # Training data: routes with depth > 4 (as in the paper)
   python -m slime_retro.prepare_data depth_min=5 depth_max=30

   # Eval prompts
   python -m slime_retro.prepare_data output=datasets/slime_retro_eval_uspto190.parquet targets=uspto-190
   ```

4. Convert the HF checkpoint to Megatron torch_dist format (run once):
   ```bash
   bash slime_retro/scripts/convert_chkpt.sh
   ```

## Training

```bash
# Default (Qwen3-4B-Instruct-2507, GSPO), via the repo-root entry point
bash scripts/train.sh

# Or call this Docker launcher directly
bash slime_retro/scripts/train.sh

# Switch algorithm to GRPO (GSPO is the default)
ADV_ESTIMATOR=grpo bash scripts/train.sh

# Sampling-based eval during training
EVAL_TEMPERATURE=0.6 EVAL_TOP_K=20 bash scripts/train.sh
```

## Evaluation

Serve a checkpoint on SGLang (`export_hf.sh` exports a Megatron checkpoint to HF first, if needed), start the host tool
server, then run `scripts/eval.sh` (or `python -m slime_retro.eval`). See the top-level `README.md` for the full eval
flow. The evaluation uses the iterative-planning protocol with budget milestones, pass@1 (`max_rollouts=1`), optional
multi-replica pooling, and per-target resume.

```bash
bash scripts/eval.sh                       # USPTO-190, budget 500, T=0.6
TARGETS=chembl-1000 bash scripts/eval.sh   # out-of-distribution set
bash scripts/eval.sh max_rollouts=1        # pass@1
```

## Scripts

| Script | Description |
|--------|-------------|
| `setup.sh` | Fresh-host bootstrap (deps, weights from HuggingFace, `datasets/` via `scripts.prepare_datasets`, Docker image) |
| `train.sh` | Launch slime training in Docker |
| `convert_chkpt.sh` | Convert HF checkpoint to Megatron torch_dist format |
| `export_hf.sh` | Export Megatron checkpoint back to HF format |
| `verify_chkpt.sh` | Sanity-check a converted checkpoint |
| `models/common.sh` | Shared training config (slime/Megatron args, Ray, SGLang) |
| `models/qwen3_4b_instr_2507.sh` | Qwen3-4B-Instruct-2507 model config |

| Python module | Description |
|---------------|-------------|
| `slime_retro.eval` | Iterative-planning eval driver (multi-replica pool, resume, pass@1) |
| `slime_retro.eval_worker` | Per-episode eval engine (`eval_single_target`) used by `slime_retro.eval` |
