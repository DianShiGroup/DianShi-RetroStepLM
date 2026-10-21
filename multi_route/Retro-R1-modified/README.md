# Retro-R1

Retro-R1 is an LLM-based agent for multi-step retrosynthetic planning. The
policy interacts with a retrosynthesis environment over a structured text
interface with two actions — `CALL` asks the single-step retrosynthesis model
for candidate reactions on a molecule, `SELECT` commits to one of them — and is
trained with reinforcement learning to decide when to call, what to select, and
when to stop.

This repository is the minimal inference-side fork of Retro-R1: the original
`agent_r1` training code, tools, reward, model files and package sources are
kept where they were, and `inference/` holds the API-driven single-step
inference stack we added on top.

On top of that stack we added a **single-step model served over HTTP, gated by
reaction templates**. That work is what the first half of this document is about.

Upstream: [GenSI-THUAIR/Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1)
(NeurIPS 2025, MIT). This fork is distributed under the same terms; see
[License](#license).

---

## Table of Contents

**Single-step inference with the template gate (our addition)**

1. [What we added](#what-we-added)
2. [Assets you must provide](#assets-you-must-provide)
3. [Quick start](#quick-start)
4. [Configurations](#configurations)
5. [Environment variables](#environment-variables)
6. [Resuming an interrupted run](#resuming-an-interrupted-run)

**Retro-R1 proper**

7. [Running the agent](#running-the-agent) · 8. [Command-line reference](#command-line-reference)
9. [Repository Structure](#repository-structure) · 10. [Citation and License](#citation-and-license)

---

# Single-step inference with the template gate

## What we added

`CALL` needs a single-step retrosynthesis model. Retro-R1 ships one as a local
MLP checkpoint; we added an HTTP alternative — a retrosynthesis LLM served
behind an OpenAI-compatible `/v1/chat/completions` API, whose predictions are
then filtered and reranked by a reaction-template gate before `SELECT` ever
sees them.

There is **no retrieval**. The step model answers from its parameters alone;
the only thing standing between its raw output and the search is the gate.

Two pieces:

| Piece | Where | What it does |
|---|---|---|
| Step backend | `inference/step/dianshi_direct.py` | Calls the step model over HTTP, parses the answer, applies the gate. `inference/step/factory.py` picks it from the config's `backend` field. |
| Template gate | `inference/gate/rdchiral_template_filter.py`, `aot_reactant_ranker.py`, `scscore_numpy.py` | Applies rdchiral templates to the predicted reactions and reranks them with AOT* scScore, so a proposed reaction has to actually apply before `SELECT` sees it. |

The gate does three things, in order:

1. **Template application.** Each predicted disconnection is matched against the
   rdchiral template library and the template is applied to check the reaction
   really works. Candidates whose template does not apply are dropped.
2. **Blurry matching.** When the strict pass comes up short, a second pass allows
   approximate reactant matches and returns the next-best templates instead of
   giving up.
3. **AOT* reranking.** Survivors are scored with SCScore — a reaction-template
   applicability model — and ordered by it. Reactants that are not in the
   building-block stock carry an `unsolved_penalty`.

`match_mode: apply` is the shipped setting; `ranking: aot` selects SCScore
reranking (`ranking: none` keeps the original order).

The original local MLP backend is still available and needs no config file:

```bash
python cli.py --data_file retro_assets/targets/uspto_190_targets_smiles.txt --llm configs/planner.yaml \
              --single_step one_step_model/retro_star_zero_ours.ckpt
```

The checkpoint is distributed by the upstream project, not by this repository;
see the assets table below.


## Assets you must provide

None of these are in the repository. Fill them in `.env`
(copy `.env.example`) and every config picks them up.

TODO(release): the "get it from" paths below are temporary absolute paths on
the internal shared storage; replace them with public download links before
publishing.

| Asset | Variable | Get it from | Notes |
|---|---|---|---|
| Runtime assets tarball | — | `/mnt/shared-storage-user/mineru4s/lixibao/retro_assets_20261015.tar.gz` | Unpacks to `retro_assets/`; brings the template library, the building-block stock and the target sets. |
| rdchiral template library | `WANJ_RDCHIRAL_TEMPLATE_JSONL` | in the tarball: `template/rdchiral_res_rxn_temp_count.jsonl` | One template per line. **Required** — the shipped config has the gate on. |
| Building blocks | `STARTING_MOLECULES` | in the tarball: `stock/emolecules_standardized.sqlite` | Required — the gate asks whether each reactant is purchasable. |
| Planner weights | `PLANNER_MODEL_PATH` | [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) | The policy LLM. Set `PLANNER_MODEL_PATH` to this repo ID; vLLM downloads it on first serve, or pre-download with `huggingface-cli download`. |
| Step-model weights | `SINGLE_STEP_MODEL_PATH` | the dianshi direct checkpoint: `/mnt/shared-storage-user/mineru4s/anxiao/ms-swift/output/0314_cot_v1_4b_10e_atomwise/v0-20260811-101041/checkpoint-47682` | Served separately from the planner. |
| MLP step-model checkpoint | `--single_step` | Only for the original local MLP backend. `retro_star_zero_ours.ckpt` (V3) is distributed by [upstream Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1) via [Google Drive](https://drive.google.com/drive/u/0/folders/13DdftEV0x55OZ8ZxHNAkmcvi_4x90hPI); place it under `one_step_model/`. |

The rdchiral template library resolves as explicit config value >
`WANJ_RDCHIRAL_TEMPLATE_JSONL`, and a fresh clone that has neither set gets an
error naming the variable. The AOT* SCScore checkpoint is vendored under
`inference/gate/weights/`, so nothing here needs a second repository; see
[Third-party code](#third-party-code).

## Quick start

Four steps. Unpack the runtime assets, create the environment, serve each model
on the machine that has the GPUs, then run one evaluation.

```bash
# 0. environment
conda env create --file environment.yaml && conda activate Retro_R1
pip install -r requirements.txt
# serving additionally needs vLLM (>= 0.18; validated with 0.26) on the GPU
# machine -- install it there; a separate environment is fine, the serve
# scripts only need `vllm` on PATH

# 1. unpack the runtime assets (from the repo root) and fill .env
tar xf /mnt/shared-storage-user/mineru4s/lixibao/retro_assets_20261015.tar.gz
cp .env.example .env && $EDITOR .env
#    WANJ_RDCHIRAL_TEMPLATE_JSONL=$PWD/retro_assets/template/rdchiral_res_rxn_temp_count.jsonl
#    STARTING_MOLECULES=$PWD/retro_assets/stock/emolecules_standardized.sqlite
#    TARGETS_FILE=$PWD/retro_assets/targets/uspto_190_targets_smiles.txt

# 2. serve the planner (the policy)          -> $PLANNER_BASE_URL
bash scripts/serve_planner.sh             # serves $PLANNER_MODEL_PATH
#    PLANNER_MODEL_PATH defaults to the Hugging Face repo ID
#    Qwen/Qwen3.5-397B-A17B-FP8; vLLM downloads it on first serve.
#    Pre-download: huggingface-cli download Qwen/Qwen3.5-397B-A17B-FP8

# 3. serve the single-step model             -> http://127.0.0.1:8001/v1
bash scripts/serve_single_step.sh         # serves $SINGLE_STEP_MODEL_PATH

# 4. run one evaluation (TARGETS_FILE in .env supplies the targets)
bash scripts/run_step_eval.sh --config configs/dianshi_direct.yaml
```

The same direct config serves all three target sets; only `TARGETS_FILE` or
`--data_file` changes:

| Dataset | Targets | Target file |
|---|---|---:|
| Pistachio-100 | 100 | `retro_assets/targets/pistachio_hard_tree.txt` |
| USPTO-190 | 190 | `retro_assets/targets/uspto_190_targets_smiles.txt` |
| PatMol-150 (internal) | 150 | `retro_assets/targets/final_product_routes_merged_root_smiles_dedup_longest_495_exclude_WanJRxn_Train_0314_exact_products_depth_max_ge_7_clean.txt` |

A dataset-specific config is not needed: the step model, gate and search
protocol are identical, and the target set is selected at runtime.

`run_step_eval.sh` loads `.env`, checks both endpoints and runs `cli.py` with
`--resume`; the output directory includes both the config and target-file
names (`results/<config>/<targets>`), so the three datasets do not collide.
For a fresh run, move the existing output directory aside first; otherwise the
existing target rows are reused.

To drive `cli.py` yourself instead:

```bash
python cli.py \
  --data_file retro_assets/targets/uspto_190_targets_smiles.txt \
  --llm configs/planner.yaml \
  --single_step_config configs/dianshi_direct.yaml \
  --output_dir results/direct
```

## Configurations

Two configs ship. The step backend is read from the config's `backend` field,
so `run_step_eval.sh` needs no backend flag.

| Config | Backend | Role |
|---|---|---|
| `configs/dianshi_direct.yaml` | `dianshi_direct` | The shipped step configuration: a direct step model (no retrieval) plus the rdchiral/AOT* gate on |
| `configs/planner.yaml` | — | The policy LLM: endpoint, served name, key |

Turning the gate off — `template_filter.enabled: false` — removes filtering and
reranking entirely, which is a different system and will not reproduce the
released numbers.

To reproduce a run with a different step model or decoder, copy the closest
config and change what you need — the fields that matter are:

| Field | Meaning |
|---|---|
| `backend` | Which adapter loads: `dianshi_direct` |
| `api_urls` | Endpoints of the step model (a list load-balances across replicas) |
| `model_name` | Name the step model is served under — must match `scripts/serve_single_step.sh` |
| `num_res` | Candidates requested per molecule |
| `max_tokens`, `temperature`, `timeout` | Decoding |
| `prompt_path_no_rag` | The prompt the step model is called with |
| `template_filter.*` | The gate: `enabled`, `match_mode`, `ranking`, `apply_max_templates`, `blurry_top_n`, `unsolved_penalty` |

Configs are plain YAML and may reference environment variables as `${VAR}` or
`${VAR:-fallback}`; the values live in `.env`, so a config never has to be
edited to point at a different machine. An unset `${VAR}` with no fallback is
deliberately left as-is, so a typo surfaces as an explicit error instead of a
silently empty path.

The building-block stock is passed to the step tool separately, with
`--starting_molecules`, rather than through the config.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `WANJ_RDCHIRAL_TEMPLATE_JSONL` | configs | rdchiral template library — required by the gate |
| `STARTING_MOLECULES` | `--starting_molecules` | Building-block stock the gate treats as purchasable |
| `SCSCORE_WEIGHTS` | template gate | Alternative SCScore checkpoint; defaults to the vendored one |
| `PLANNER_*` | `serve_planner.sh` | Weights, served name, port, TP, context length |
| `SINGLE_STEP_MODEL`, `SINGLE_STEP_MODEL_PATH`, `SINGLE_STEP_PORT`, `SINGLE_STEP_TP` | `serve_single_step.sh` | Step-model serving |

Resolution order for every asset is **config value → environment variable**.
A relative config path is taken against the repository root; a fresh clone that
has neither set gets an error naming the variable to set.

## Resuming an interrupted run

Every (target, experiment) pair writes its result to
`results/expN_tgtNNN.json` under the run's log directory, and `--resume` is
always on. A pair whose file already exists is read back instead of recomputed,
so a run that dies half way through can simply be restarted.

This is a plain file check, not a cache: nothing is shared between runs or
machines, and deleting the results directory just forces a recompute.


---

# Retro-R1 proper

## Running the agent

`cli.py` is the entry point for inference. It takes a target file (one SMILES
per line, or a validation parquet), a planner configuration, and a step
configuration, and writes per-target results plus a trajectory:

```bash
python cli.py \
  --data_file retro_assets/targets/uspto_190_targets_smiles.txt \
  --llm configs/planner.yaml \
  --single_step_config configs/dianshi_direct.yaml \
  --max_turns 100 --maxstep 30 --topk 10 \
  --output_dir results/run1 --resume
```

`--resume` skips targets already present in the output directory, so an
interrupted sweep can be restarted as-is.

The `CALL` / `SELECT` loop, the environment and the RL training code are the
upstream project's and are unchanged here; see the [upstream
repository](https://github.com/GenSI-THUAIR/Retro-R1) for the method, the
reward and the training entry points.

## Command-line reference

| Flag | Meaning |
|---|---|
| `--data_file` | Targets: a `.txt` with one SMILES per line, or a validation parquet |
| `--max_samples` | Only run the first N targets |
| `--llm` | Planner YAML (`configs/planner.yaml`) |
| `--model`, `--api_base`, `--api_key`, `--temperature`, `--max_tokens`, `--timeout` | Planner overrides; CLI values win over the YAML |
| `--single_step_config` | Step-model YAML; picks the backend from its `backend` field |
| `--single_step` | A local MLP `.ckpt` instead of an HTTP model (mutually exclusive with the above) |
| `--max_turns` | Max tool-calling turns per target (default 100) |
| `--maxstep` | Max synthesis steps allowed (default 30) |
| `--topk` | Candidates returned per `CALL` (default 10) |
| `--workers` | Parallel targets |
| `--starting_molecules` | Override the building-block file |
| `--force_noloop` | Prevent cycles in the search |
| `--output_dir`, `--resume` | Where results go, and skip what is already there |
| `--save_trajectory`, `--verbose` | Per-target transcript dump / verbose logging |

## Repository Structure

### Single-step inference with the template gate (ours)

| Path | Description |
|------|-------------|
| `cli.py` | Inference entry point |
| `inference/runner.py` | Argument parsing, planner/step resolution, the `CALL`/`SELECT` loop |
| `inference/step/dianshi_direct.py` | DianShi-Direct step backend: HTTP step model plus the template gate |
| `inference/step/mlp.py` | The original local MLP step model |
| `inference/step/factory.py` | Builds the step tool named by the config's `backend` |
| `inference/gate/rdchiral_template_filter.py` | rdchiral template gate (exact / blurry / apply-match modes) |
| `inference/gate/aot_reactant_ranker.py` | AOT* scScore reranking of applied templates |
| `inference/gate/scscore_numpy.py` | AOT*'s standalone NumPy scorer, vendored (see [Third-party code](#third-party-code)) |
| `inference/gate/weights/` | The published 2048-bit SCScore checkpoint |
| `inference/support/config_loader.py` | YAML loading with `${VAR}` / `${VAR:-fallback}` expansion |
| `inference/support/prompt_sections.py` | Step-prompt parsing and message assembly |
| `inference/support/smiles_canonical.py` | Canonical SMILES helpers shared by the backend and the gate |
| `inference/support/smiles_atomwise.py` | Atom-wise SMILES formatting for the step prompt |
| `inference/test_rdchiral_template_filter.py` | Template-gate unit tests |
| `inference/planner.py`, `inference/prompts/` | Planner client and prompt files |
| `inference/stock.py` | Building-block lookup |
| `configs/dianshi_direct.yaml`, `configs/planner.yaml` | The shipped step configuration, plus the planner |
| `.env.example` | Template for every asset path and endpoint (`.env` holds the real values, untracked) |
| `scripts/serve_planner.sh`, `scripts/serve_single_step.sh` | Serve either model with vLLM |
| `scripts/run_step_eval.sh` | One entry point for the step backend |

### Agent, training and evaluation (upstream)

| Path | Description |
|------|-------------|
| `agent_r1/` | PPO/GRPO training on top of verl: rollout generator, reward, tool env |
| `agent_r1/tool/tools/` | The `CALL` / `SELECT` tools and their prompts |
| `examples/`, `dataset/`, `one_step_model/` | Upstream targets, route data and MLP step-model assets |
| `packages/` | Vendored dependencies (`rdchiral`, `nvidia-resiliency-ext`) |

## Citation and License

If you use Retro-R1, please cite the upstream work:

```bibtex
@inproceedings{liu2025retror1,
  title     = {{RETRO-R1}: {LLM}-based Agentic Retrosynthesis},
  author    = {Liu, Wei and Feng, Jiangtao and Hong, Liyang and Song, Yuxuan and Li, Yuqiang and Zhang, Shufei and Bai, Lei and Zhou, Haohan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

This fork adds the direct single-step model and its reaction-template gate
described above; please cite this repository as well when you use them.

### Third-party code

The template gate and its reranker are adapted from public sources rather than
written from scratch. Both are covered by this repository's MIT license, with
attribution below.

**AOT\* — `inference/gate/scscore_numpy.py`, `inference/gate/aot_reactant_ranker.py`**

The scScore scorer and the single-step reward shape come from
[AOT\*](https://github.com/ShawnKS/AOTstar) (Song et al., 2025). We vendored
its standalone NumPy scorer so this repository runs with no extra checkout, and
adapted it:

- removed the `six` shim (`dict.iteritems` → `dict.items`), so the package no
  longer carries an undeclared dependency;
- removed the Python 2 `cPickle` weight loader, so only the published
  JSON+gzip checkpoints are supported;
- turned the progress `print` into a log record and dropped the module's
  `__main__` demo;
- kept the explicit scalar reduction in the sigmoid. NumPy 2 refuses to convert
  a length-one array to a Python float, and the last layer leaves exactly that,
  so without it the score is not even readable.

The scoring arithmetic is unchanged: we checked the vendored scorer against the
original on ten molecules including edge cases, and the values are
bit-identical. The weights at `inference/gate/weights/` are the published
`full_reaxys_model_2048bool` checkpoint from
[connorcoley/scscore](https://github.com/connorcoley/scscore) (MIT). That
repository's `full_reaxys_model_1024bool` checkpoint is a *different* model and
cannot be substituted: the gate scores with `FP_len=2048`.

```bibtex
@article{song2025aotstar,
  title={AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search},
  author={Song, Xiaozhuang and Pan, Xuanhao and Zhao, Xinjian and
          Ye, Hangting and Zhang, Shufei and Tang, Jian and Yu, Tianshu},
  journal={arXiv preprint arXiv:2509.20988},
  year={2025}
}
```

**rdchiral — `packages/`, and the gate's `match_mode: apply` path**

Templates are applied with rdchiral, vendored under `packages/rdchiral/`; see
its own license there.

**Removed: the rule-based Retrosim search**

An earlier version fell back to AOT*'s similarity-based RAG when no candidate
passed the template gate. Across our full campaign it loaded 99 times, failed
111 times and never returned a result, so it contributed to no published
number. It is gone rather than left in as dead configuration.
### License

MIT, inherited from [GenSI-THUAIR/Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1).
Vendored dependencies keep their own licenses (`packages/rdchiral/LICENSE`).
The rdchiral template library is **not** part of this repository — obtain it
from its own source and point `WANJ_RDCHIRAL_TEMPLATE_JSONL` at it. The AOT*
scScore checkpoint *is* included, under the terms described in
[Third-party code](#third-party-code).