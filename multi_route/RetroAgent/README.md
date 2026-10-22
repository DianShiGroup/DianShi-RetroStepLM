# RetroAgent

RetroAgent is an LLM agent for multi-step retrosynthesis planning. It decomposes
a target molecule into commercially available building blocks by searching over
an AND-OR graph of molecules and reactions, driven entirely by tool calls. The
graph, the single-step reaction model, building-block lookup, and molecular
scoring all live in an HTTP tool server that forms the agent's harness; the
policy is a language model that reasons over the returned observations and
issues the next tool call.

This repository is the fork we release our addition on: the original
RetroAgent release (agent, training pipeline, tool server) is kept where it
was — trained weights: [SXKDZ/RetroAgent](https://huggingface.co/SXKDZ/RetroAgent).

On top of that we added a **single-step model served over HTTP, gated by
reaction templates**. That work is what the first half of this document is
about, and all you need to reproduce the direct+gate results; the second half
documents the original release.

---

## Table of Contents

**Single-step inference with the template gate (our addition)**

1. [What we added](#what-we-added)
2. [Assets you must provide](#assets-you-must-provide)
3. [Quick start](#quick-start)
4. [Configurations](#configurations)
5. [Environment variables](#environment-variables)
6. [Resuming an interrupted run](#resuming-an-interrupted-run)

**RetroAgent proper**

7. [Running the agent](#running-the-agent) · 8. [Tool Server](#tool-server)
9. [Repository Structure](#repository-structure) · 10. [Citation and License](#citation-and-license)

---

# Single-step inference with the template gate

## What we added

The stock tool server proposes single-step disconnections with a template model
(`external/desp/`). We replaced that proposal step with an HTTP **single-step
model**: a retrosynthesis LLM served behind an OpenAI-compatible
`/v1/chat/completions` API, whose predictions are then filtered and reranked by
a reaction-template gate before they are allowed into the search.

There is **no retrieval**. The step model answers from its parameters alone;
the only thing standing between its raw output and the search is the gate.

Two pieces:

| Piece | Where | What it does |
|---|---|---|
| Step backend | `inference/step/dianshi_direct.py` | Calls the step model over HTTP, parses the answer, applies the gate. The tool server loads it from the config's `backend` field. |
| Template gate | `inference/gate/rdchiral_template_filter.py`, `aot_reactant_ranker.py`, `scscore_numpy.py` | Applies rdchiral templates to the predicted reactions and reranks them with AOT* scScore, so a proposed reaction has to actually apply before it enters the search. |

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

> **About the `slime_retro` name.** Our scripts drive the evaluation through
> `slime_retro.eval`, the upstream eval driver — the name is inherited. The
> module does **not** depend on the slime RL framework: no Docker image, no
> Megatron checkpoint, no training stack is needed for anything in this half
> of the README.

## Assets you must provide

None of these are in the repository. Fill them in `.env`
(copy `.env.example`) and every config picks them up.

| Asset | Variable | Get it from | Notes |
|---|---|---|---|
| Runtime assets | — | [DianShiGroup/DianShi-RetroRoute-Dataset](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroRoute-Dataset) | Download to `DianShi-RetroRoute-Dataset/`; brings the template library, the building-block stock, the tool-server SCScore and the target sets. Place the files as described in its `README.md`. |
| rdchiral template library | `DIANSHI_RDCHIRAL_TEMPLATE_JSONL` | in the dataset: `rxn_template/rdchiral_res_rxn_temp_count.jsonl` | One template per line. **Required** — the shipped config has the gate on. |
| Building blocks | `STARTING_MOLECULES` | in the dataset: `stock/emolecules_standardized.sqlite` | Required — the gate asks whether each reactant is purchasable. |
| Planner weights | `PLANNER_MODEL_PATH` | [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) | The policy LLM. Set `PLANNER_MODEL_PATH` to this repo ID; vLLM downloads it on first serve, or pre-download with `huggingface-cli download`. |
| Step-model weights | `SINGLE_STEP_MODEL_PATH` | [DianShiGroup/DianShi-RetroStep-LM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStep-LM-Direct) | Served separately from the planner. |

The rdchiral template library resolves as explicit config value >
`DIANSHI_RDCHIRAL_TEMPLATE_JSONL`, and a fresh clone that has neither set gets an
error naming the variable. The AOT* SCScore checkpoint is vendored under
`inference/gate/weights/`, so nothing here needs a second repository; see
[Third-party code](#third-party-code).

## Quick start

Four steps, no cluster scheduler. Download the runtime assets, serve each model
on the machine that has the GPUs, then run one evaluation.

```bash
# 0. environment
pip install -r requirements.txt           # tool server + eval driver (host python)
# serving additionally needs vLLM (>= 0.18; validated with 0.26) on the GPU
# machine -- install it there; the serve scripts only need `vllm` on PATH

# 1. download the runtime assets (from the repo root; see "Assets you must provide")
hf download DianShiGroup/DianShi-RetroRoute-Dataset --repo-type dataset --local-dir DianShi-RetroRoute-Dataset
mkdir -p dataset
cp DianShi-RetroRoute-Dataset/stock/emolecules_standardized.sqlite dataset/   # tool-server default path
mkdir -p datasets/scscore
cp DianShi-RetroRoute-Dataset/scscore/model.ckpt-10654.as_numpy.json.gz datasets/scscore/
cp .env.example .env && $EDITOR .env      # asset paths, endpoints, credentials

# 2. serve the planner (the policy)          -> $PLANNER_BASE_URL
bash scripts/serve_planner.sh             # serves $PLANNER_MODEL_PATH
#    PLANNER_MODEL_PATH defaults to the Hugging Face repo ID
#    Qwen/Qwen3.5-397B-A17B-FP8; vLLM downloads it on first serve.
#    Pre-download: huggingface-cli download Qwen/Qwen3.5-397B-A17B-FP8

# 3. serve the single-step model             -> http://127.0.0.1:8001/v1
bash scripts/serve_single_step.sh         # serves $SINGLE_STEP_MODEL_PATH

# 4. run one evaluation
bash scripts/run_step_eval.sh \
    --config configs/dianshi_direct.yaml \
    --targets DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt
```

The same direct config serves all three target sets; only `--targets` changes:

| Dataset | Targets | Target file |
|---|---|---:|
| Pistachio-100 | 100 | `DianShi-RetroRoute-Dataset/test_dataset/pistachio_hard_tree.txt` |
| USPTO-190 | 190 | `DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt` |
| PatMol-150 | 150 | `DianShi-RetroRoute-Dataset/test_dataset/patmol_150_targets.txt` |

A dataset-specific config is not needed: the step model, gate and search
protocol are identical, and the target set is selected at runtime.

`run_step_eval.sh` loads `.env`, checks both endpoints, starts a tool server
bound to that config, runs the evaluation and shuts the tool server down. The
output directory is named after the config and target file
(`results/<config>/<targets>`), so the three datasets do not collide.
`--resume` is always on, so an interrupted run can simply be restarted. For a
fresh run, move the existing `results/<config>/<targets>` directory aside
first; otherwise the existing target rows are reused.

## Configurations

One config ships: `configs/dianshi_direct.yaml`. The backend is read from the
config's `backend` field, so `run_step_eval.sh` needs no backend flag.

That file is the configuration the released numbers were produced with: a
direct step model (no retrieval) plus the rdchiral/AOT* gate on. Turning the
gate off — `template_filter.enabled: false` — removes filtering and reranking
entirely, which is a different system and will not reproduce them.

To reproduce a run with a different step model or decoder, copy the config and
change what you need — the fields that matter are:

| Field | Meaning |
|---|---|
| `backend` | Picks the step backend; `dianshi_direct` is the shipped one |
| `retro.api_url`, `retro.model` | Step-model endpoint and served name — must match `scripts/serve_single_step.sh` |
| `retro.num_res` | Candidates requested per molecule |
| `max_tokens`, `temperature`, `timeout` | Decoding |
| `prompt_path_no_rag` | The prompt the step model is called with |
| `template_filter.*` | The gate: `enabled`, `match_mode`, `ranking`, `apply_max_templates`, `blurry_top_n`, `unsolved_penalty` |

Configs are plain YAML and may reference environment variables as `${VAR}` or
`${VAR:-fallback}`; the values live in `.env`, so a config never has to be
edited to point at a different machine. An unset `${VAR}` with no fallback is
deliberately left as-is, so a typo surfaces as an explicit error instead of a
silently empty path.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DIANSHI_RDCHIRAL_TEMPLATE_JSONL` | configs | rdchiral template library — required by the gate |
| `STARTING_MOLECULES` | configs | Building-block stock the gate treats as purchasable |
| `SCSCORE_WEIGHTS` | template gate | Alternative SCScore checkpoint; defaults to the vendored one |
| `TOKENIZER_MODEL_NAME` | eval scripts | Tokenizer that renders the planner prompt; defaults to the public Qwen3-4B, whose chat template is the one the released numbers were produced with |
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

# RetroAgent proper

The rest of this document describes the original RetroAgent release that this
fork builds on. None of it is needed for the direct+gate results above.

## Running the agent

The agent's harness is the tool server; the policy is any LLM served behind a
generation endpoint. Evaluation runs through `slime_retro.eval`
([Tool Server](#tool-server) below has the launch commands):

```bash
# drive the pieces yourself: tool server + eval driver
RETRO_SINGLE_STEP_CONFIG=configs/dianshi_direct.yaml \
  python -m tools.tool_server base_port=8100 num_instances=1
bash scripts/eval.sh        # or scripts/eval_step.sh for an OpenAI-compatible planner

# the paper's original configuration: the bundled DESP template model as the
# step model, with no gate — start the tool server WITHOUT RETRO_SINGLE_STEP_CONFIG
python -m tools.tool_server base_port=8100 num_instances=1
bash scripts/eval.sh
```

Main `slime_retro.eval` fields (chz-style `key=value` arguments):

| Field | Meaning |
|---|---|
| `planner_backend` | `vllm` (OpenAI completions API) or `sglang` (legacy) |
| `planner_url`, `planner_model`, `planner_api_key` | Planner endpoint for the `vllm` backend; comma-separated URLs form a load-balanced pool |
| `sglang_url` | Planner endpoint(s) for the `sglang` backend |
| `model_name` | Tokenizer that renders the planner prompt; defaults to `Qwen/Qwen3-4B-Instruct-2507` |
| `targets` | A file with one SMILES per line, or a dataset key (`uspto-190`, `chembl-1000`) |
| `max_budget`, `milestones`, `num_exp`, `max_rollouts`, `max_turns` | Search-budget protocol (the released numbers use budget 500, `num_exp=10`) |
| `temperature`, `top_k`, `top_p`, `max_response_len` | Decoding |
| `predict_templates_top_k` | Candidates surfaced per `predict_templates` call |
| `concurrency`, `tool_server_port`, `tool_server_instances` | Parallel targets and tool-server pool |
| `log_dir` | Where results and transcripts go |

The RL training pipeline (slime + Megatron + SGLang inside the
`slimerl/slime:v0.3.0` Docker image; entry points `scripts/train.sh` and
`slime_retro/scripts/`) is included for completeness and is not needed for the
direct+gate results; see the paper for the method and the reward.

## Tool Server

The tool server (`tools/tool_server.py`) launches `num_instances`
FastAPI/uvicorn processes on consecutive ports from `base_port`. Each process
opens the shared on-disk stock database, loads its own single-step and SCScore
resources, and serves per-session `GraphMemory`. Clients assign new sessions
round-robin across instances, then pin each session to its instance (session
affinity).

```bash
# Single instance for local development
python -m tools.tool_server base_port=8100

# 16 instances on ports 8100-8115
python -m tools.tool_server base_port=8100 num_instances=16

# Health / inspection
curl http://127.0.0.1:8100/metrics
curl http://127.0.0.1:8100/tool_schemas
```

`RETRO_SINGLE_STEP_CONFIG` selects the direct+gate backend
(see [Quick start](#quick-start)); without it the server uses the bundled
`RetroPredictor`. Training and evaluation reach the server via
`RETRO_TOOL_SERVER_PORT` (default `8100`) and `RETRO_TOOL_SERVER_INSTANCES`
(default `16`); `num_instances` must be at least `RETRO_TOOL_SERVER_INSTANCES`.

### Tools exposed to the agent

The agent interacts with the environment through 8 tools: 4 memory tools that
query and modify the AND-OR graph, and 4 chemistry tools that provide
heuristic guidance.

| Tool | Description |
|------|-------------|
| `predict_templates(molecule_id)` | ML single-step candidates `P1..Pk`; cycle-causing and duplicate predictions are pre-filtered, results cached per molecule |
| `expand(molecule_id, prediction_id)` | Apply prediction(s) as a reaction node; comma-separated IDs allowed (`"P1,P3"`) |
| `get_frontier()` | List `OPEN` molecules still needing decomposition; also signals when the root is solved |
| `inspect_node(node_id)` | Node status, parents, and children |
| `get_molecule_info(smiles)` | Molecular weight, logP, atom / ring counts |
| `search_building_blocks(smiles)` | Commercial-availability check (exact canonical match) |
| `calculate_synthetic_accessibility(smiles)` | SA score (1-10) |
| `calculate_scscore(smiles)` | SCScore synthetic complexity (1-5) |

Route completion is detected automatically: the server sets `route_solved` on
the tool result when the root molecule becomes `SOLVED`. There is no explicit
"finish" tool.

## Repository Structure

### Single-step inference with the template gate (ours)

| Path | Description |
|------|-------------|
| `inference/step/dianshi_direct.py` | The shipped step backend: OpenAI-compatible chat completions, plus the template gate |
| `inference/gate/rdchiral_template_filter.py` | rdchiral template gate (exact / blurry / apply-match modes) |
| `inference/gate/aot_reactant_ranker.py` | AOT* scScore reranking of applied templates |
| `inference/gate/scscore_numpy.py` | AOT*'s standalone NumPy scorer, vendored (see [Third-party code](#third-party-code)) |
| `inference/gate/weights/` | The published 2048-bit SCScore checkpoint |
| `inference/support/config_loader.py` | YAML loading with `${VAR}` / `${VAR:-fallback}` expansion |
| `inference/support/prompt_sections.py` | Step-prompt parsing and message assembly |
| `inference/support/smiles_canonical.py` | Canonical SMILES helpers shared by the backend and the gate |
| `inference/support/smiles_atomwise.py` | Atom-wise SMILES formatting for the step prompt |
| `inference/prompts/` | The step prompt |
| `inference/test_rdchiral_template_filter.py` | Template-gate unit tests |
| `configs/dianshi_direct.yaml` | The shipped step configuration |
| `.env.example` | Template for every asset path and endpoint (`.env` holds the real values, untracked) |
| `scripts/serve_planner.sh`, `scripts/serve_single_step.sh` | Serve either model with vLLM |
| `scripts/run_step_eval.sh` | One entry point for the step backend |

### Original release (upstream)

| Path | Description |
|------|-------------|
| `scripts/prepare_datasets.py` | Download the `datasets/` files from their upstream sources (DESP via figshare, LLM-Syn-Planner, PDVN, connorcoley/scscore — exact URLs and sizes are in the script) |
| `scripts/train.sh`, `slime_retro/scripts/` | RL training entry points (Docker `slimerl/slime:v0.3.0`, Megatron + slime + SGLang) |
| `slime_retro/generate.py` | Multi-turn rollout generator + `reward_func` (slime custom callbacks) |
| `slime_retro/eval.py`, `eval_worker.py`, `transcript.py` | The iterative-planning eval driver, per-episode engine, and transcript writer |
| `slime_retro/prepare_data.py` | Build the training / eval prompt parquets from USPTO-Full routes |
| `tools/` | FastAPI tool server, async `ToolClient`, AND-OR `GraphMemory`, chemistry tools |
| `external/desp/` | Vendored `RetroPredictor` single-step retrosynthesis model |
| `data/` | Target/building-block loaders; `BuildingBlockChecker` (exact SQLite lookup over 23.1M building blocks) |
| `analyzers/` | Route analysis and standalone route visualization |

## Citation and License

If you use RetroAgent, please cite:

```bibtex
@inproceedings{zhu2026retroagent,
  title     = {RetroAgent: Harnessing {LLMs} to Search Over Structured Memory for Agentic Retrosynthesis Planning},
  author    = {Zhu, Yanqiao and Gan, Jingru and Sun, Xiaoqi and Sun, Fang and Shi, Yidan and Islam, Md Mofijul and Shang, Chao and Gao, Wenhao and Coley, Connor W. and Sun, Yizhou and Wang, Wei},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
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

**rdchiral — the gate's `match_mode: apply` path**

Templates are applied with rdchiral, installed from PyPI (see
`requirements.txt`); see its own license for terms.

**Removed: the rule-based Retrosim search**

An earlier version fell back to AOT*'s similarity-based RAG when no candidate
passed the template gate. Across our full campaign it loaded 99 times, failed
111 times and never returned a result, so it contributed to no published
number. It is gone rather than left in as dead configuration.

### License

See [`LICENSE`](LICENSE) for the RetroAgent code license (MIT). The
single-step retrosynthesis model vendored under `external/desp/` is derived
from its upstream project and remains subject to that project's original
license and attribution (see [`NOTICE`](NOTICE)). Building blocks originate
from eMolecules and the route data from
[Retro\*](https://github.com/binghong-ml/retro_star) — cite the upstream work
you use. The rdchiral template library is **not** part of this repository —
obtain it from its own source and point `DIANSHI_RDCHIRAL_TEMPLATE_JSONL` at it.
The AOT* scorer and its checkpoint are vendored under `inference/gate/`; see
[Third-party code](#third-party-code) for attribution and terms.
