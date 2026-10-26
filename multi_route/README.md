# Multi-route planning frameworks

This folder hosts three independent multi-route retrosynthesis planning
frameworks. All three drive the same search recipe: an LLM planner explores a
tree or graph of molecules and reactions, proposes disconnections with the
**DianShi direct single-step model** (the model released in this repository —
see the root README), and every candidate must pass the **AOT\*
reaction-template gate** before it enters the search. The shipped
configurations are the ones the released numbers were produced with.

## The shared recipe

1. **Direct single-step model.** `DianShi-RetroStep-Direct` behind an
   OpenAI-compatible `/v1/chat/completions` endpoint. There is **no
   retrieval** — the model answers from its parameters alone.
2. **AOT\* template gate.** rdchiral template *application* (with a blurry
   second pass), then SCScore reranking; reactants outside the purchasable
   stock carry an `unsolved_penalty`.
3. **Purchasable building-block stock.** A standardized eMolecules sqlite
   (23M entries) used for exact canonical-SMILES availability checks.
4. **Runtime assets.** The rdchiral template library, the stock sqlite and the
   target sets are published as the Hugging Face dataset
   [DianShiGroup/DianShi-RetroRoute-Dataset](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroRoute-Dataset);
   each component README's assets table says where to place them.

5. **Planner.** [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
   behind an OpenAI-compatible endpoint. vLLM can serve the Hugging Face repo
   ID directly (it downloads on first use), or you can pre-download it with
   `huggingface-cli download`.

## SCScore checkpoints

Two checkpoints share the filename `model.ckpt-10654.as_numpy.json.gz` but are
different models. Do not swap them:

| Consumer | Fingerprint width | Checkpoint |
|---|---:|---|
| AOT\* template gate in RetroAgent | 2048 | `RetroAgent/inference/gate/weights/` |
| AOT\* template gate in Preview | 2048 | `DianShi-RetroRoute-System-Preview/runtime/single_step/aot_scscore/models/full_reaxys_model_2048bool/` |
| RetroAgent tool server (`calculate_scscore`) | 1024 | `RetroAgent/datasets/scscore/` |

The gate passes `FP_len=2048`; the tool server hardcodes `FP_len=1024`. Both
checkpoints are vendored in the code repository, and the runtime-assets dataset
does not contain either SCScore checkpoint.

## The three frameworks

| Folder | Lineage | Search harness | Planner |
|---|---|---|---|
| [`RetroAgent/`](RetroAgent/README.md) | fork of [SXKDZ/RetroAgent](https://huggingface.co/SXKDZ/RetroAgent) (COLM 2026) | HTTP tool server + AND-OR `GraphMemory` harness; agent issues tool calls | [Qwen3.5-397B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) |
| [`Retro-R1-modified/`](Retro-R1-modified/README.md) | fork of [GenSI-THUAIR/Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1) (NeurIPS 2025) | in-process `CALL` / `SELECT` agent loop | [Qwen3.5-397B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) |
| [`DianShi-RetroRoute-System-Preview/`](DianShi-RetroRoute-System-Preview/README.md) | original in-process bench harness | AND-OR colony dispatcher + workers, JSONL output | [Qwen3.5-397B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) |

Each folder is fully self-contained: its own README (quick start, assets
table, configuration reference), `requirements.txt`, serve scripts,
configuration and tests. The template-gate code is intentionally duplicated
in the two forks — they descend from different upstreams and stay
independently runnable.

## Target sets

All three frameworks use the same direct step model and template gate. The
target file is a runtime input, not a step-model setting:

| Dataset | Targets | Target file |
|---|---|---:|
| Pistachio-100 | 100 | `DianShi-RetroRoute-Dataset/test_dataset/pistachio_hard_tree.txt` |
| USPTO-190 | 190 | `DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt` |
| PatMol-150 | 150 | `DianShi-RetroRoute-Dataset/test_dataset/patmol_150_targets.txt` |

RetroAgent and Retro-R1-modified ship one `configs/dianshi_direct.yaml` and
select the target set with `--targets` / `--data_file`; they do not need
three dataset-specific configs. DianShi-RetroRoute-System-Preview ships one
lane config per benchmark because its batch runner also fixes the input and
output paths.

## Quick start (common shape)

```bash
# 0. per-component environment      see the component README
# 1. download the runtime assets    HF dataset; placement per component README
# 2. serve the two models           vLLM; serve scripts included in each component
# 3. run the evaluation             single command per component
```

Follow the component README for the exact commands — each one is a
three-step flow: environment, assets + `.env`, serve + evaluate.

## Result statistics

`tools/solved_by_n.py` computes solve rates at N = 50, 100, 200, 300, 400 and
500 from any of the three result formats. It auto-detects bench_v1 / Preview
JSONL, Retro-R1 `result.jsonl`, and RetroAgent `summary.json`:

```bash
python tools/solved_by_n.py path/to/result.jsonl
python tools/solved_by_n.py path/to/summary.json
```

It warns about results that look like infrastructure failures (service
exhaustion, timeouts, zero-call cases) and counts them as unsolved in the
reported rates, so a rerun is required before those numbers are used.

## Provenance and citation

- **RetroAgent** — upstream: the RetroAgent paper (COLM 2026) and the
  `SXKDZ/RetroAgent` weights; the direct single-step backend and the template
  gate are the addition in this folder. Bibtex in its README.
- **Retro-R1-modified** — upstream: Retro-R1 (NeurIPS 2025, MIT); the direct
  single-step backend and the template gate are the addition. Bibtex in its
  README.
- **DianShi-RetroRoute-System-Preview** — original evaluation harness.

Please cite the upstream works you use; attribution for the vendored AOT\*
scScore (connorcoley/scscore, MIT) and rdchiral is in each component README.
