# Dianshi-Retro Bench

A standalone evaluation framework for multi-step retrosynthetic planning. A
planner LLM searches over an AND–OR tree of molecules and reactions, calling a
single-step retrosynthesis model through an OpenAI-compatible API; every
candidate reaction must pass an **AOT\* reaction-template gate** — rdchiral
template application plus SCScore reranking — before it enters the search. The
search ends when the target molecule is decomposed into purchasable building
blocks or the single-step budget is exhausted.

The configuration shipped in this repository is the one the released numbers
were produced with: **Qwen3.5-397B as the planner, DianShi direct direct as the
single-step model, AOT\* gate on**. This `opensource` branch contains only what
that reproduction needs; other single-step backends (RAG, RSGPT,
RXNGraphormer), other gate variants and historical experiment configs were
removed here and remain on the `v2.1.1-bench-v1` branch.

## How the search works

### The AND–OR tree

The search state is a tree held in memory (per target):

- **MOL / ROOT nodes are OR nodes.** A molecule may be decomposed in several
  ways; one successful decomposition is enough.
- **REACTION nodes are AND nodes.** Each reaction node is one concrete
  candidate returned by the single-step model, and *all* of its reactants must
  be solved for the reaction to count.
- `SOLVED` status propagates upward automatically. The target is solved when
  the root becomes `SOLVED`.

### The planner loop

The planner (an LLM behind an OpenAI-compatible chat endpoint) never touches
the tree directly. Between its calls, the **dispatcher** renders the current
tree into a text context, computes the legal expand options, and asks the
planner to pick one (`option_id`). The dispatcher also handles everything else:

- **Lock advancement.** The search keeps a *locked primary* — the route the
  planner is currently driving. When that node becomes SOLVED, FAILED, or loses
  its live frontier, the dispatcher walks the lock up automatically (OR-retry,
  then AND-sibling, then the nearest ancestor with a live frontier) and the
  next planner call simply shows a different locked primary.
- **Action filtering.** Expand actions on solved/failed/pruned nodes or outside
  the locked subtree are rejected with feedback in the next call.
- **Termination.** When no expandable frontier remains anywhere, the loop ends.

The planner's system prompt (`runtime/prompts/planner_system_v2.md`) carries
the search policy: prefer routes that reduce unresolved synthetic burden —
core-scaffold simplification outranks stock-bearing small fragments, and
repeated functional-group reshuffling without simplification is penalized by
policy.

### The single-step model

One-step disconnections are proposed by a retrosynthesis model served behind an
OpenAI-compatible `/v1/chat/completions` endpoint. There is **no retrieval**:
the model answers from its parameters alone. Candidates are post-processed
(atom-wise SMILES formatting, deduplication, `num_res` cap) before gating.

### The AOT\* template gate

After base SMILES cleaning, every candidate reaction must pass:

1. **Template application.** The prediction is matched against the rdchiral
   template library and the template is *applied* to check the reaction really
   works. Candidates whose template does not apply are dropped.
2. **Blurry matching.** When the strict pass comes up short, a second pass
   allows approximate reactant matches and returns the next-best templates
   instead of giving up.
3. **AOT\* reranking.** Survivors are scored with SCScore (a reaction-template
   applicability model, 2048-bit checkpoint vendored in
   `runtime/single_step/aot_scscore/`) and ordered by it. Reactants that are
   not in the purchasable building-block stock carry an `unsolved_penalty`.

Only `PASS` candidates become REACTION nodes. Gate definitions and parameter
conventions: [docs/VALIDATE.md](docs/VALIDATE.md).

### Budget and success

`search.max_steps` caps the number of single-step model calls per target
(500 in the shipped lanes). A target counts as solved when the root OR node
becomes SOLVED with `search.target_complete_paths` complete routes.

## Assets

| Asset | Source |
|---|---|
| Runtime assets (rdchiral template library, building-block stock, target sets) | [DianShiGroup/DianShi-RetroRoute-Dataset](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroRoute-Dataset) |
| Planner weights (Qwen3.5-397B-A17B-FP8) | [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) — vLLM can serve the repo ID directly (it downloads on first use), or pre-download with `huggingface-cli download`. |
| Step-model weights (DianShi-RetroStep-LM-Direct) | [DianShiGroup/DianShi-RetroStep-LM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStep-LM-Direct) |

The SCScore NumPy implementation and the 2048-bit checkpoint used by the AOT\*
gate are vendored in `runtime/single_step/aot_scscore/` — no extra download.

## Quick start

```bash
# 0. Environment (Python >= 3.11)
pip install -r requirements.txt
# Serving additionally needs vLLM (>= 0.18; validated with 0.26) on the GPU
# machine.

# 1. Download the runtime assets, then set up .env
hf download DianShiGroup/DianShi-RetroRoute-Dataset --repo-type dataset --local-dir DianShi-RetroRoute-Dataset
cp .env.example .env && $EDITOR .env
#    Defaults point at ./DianShi-RetroRoute-Dataset/... for the template library, the stock
#    and the target files. The lane configs read targets from
#    ./DianShi-RetroRoute-Dataset/test_dataset/; edit the lane yaml if you downloaded elsewhere.
#    At startup the gate should log: stock=23081629 | templates=412027.
```

```bash
# 2. Serve the models on the GPU machine
bash scripts/serve_planner.sh          # planner, plain vLLM, 8 GPUs, TP=8
#    MODEL_PATH defaults to the Hugging Face repo ID
#    Qwen/Qwen3.5-397B-A17B-FP8; vLLM downloads it on first use.
#    Pre-download: huggingface-cli download Qwen/Qwen3.5-397B-A17B-FP8
bash scripts/serve_single_step.sh      # step model, plain vLLM
#    Set SINGLE_STEP_MODEL_PATH in .env; any OpenAI-compatible server works.

# 3. Run an evaluation (CPU is enough; runs in-process)
python cli.py --config configs/bench/qwen_direct_uspto190.yaml
python cli.py --config configs/bench/qwen_direct_pistachio100.yaml
python cli.py --config configs/bench/qwen_direct_patmol_150.yaml
```

## Evaluation lanes

Three lanes ship, one per benchmark set:

| Config | Benchmark | Target file | Targets |
|---|---|---|---:|
| `qwen_direct_uspto190.yaml` | USPTO-190 (public) | `DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt` | 190 |
| `qwen_direct_pistachio100.yaml` | Pistachio hard-tree 100 (public) | `DianShi-RetroRoute-Dataset/test_dataset/pistachio_hard_tree.txt` | 100 |
| `qwen_direct_patmol_150.yaml` | PatMol-150 | `DianShi-RetroRoute-Dataset/test_dataset/patmol_150_targets.txt` | 150 |

The lane configs read target files directly from
`DianShi-RetroRoute-Dataset/test_dataset/`. `run.output` and `log_file` differ per
lane. Endpoints and asset paths are set in `.env`; global defaults live in
`config.yaml`.

### Output

Each run writes one JSONL under `results/bench/<dataset>/` — one line per
target: `idx`, `target_mol`, `success`, `solved_paths`,
`termination_reason`, `metrics` (wall time, single-step calls, LLM tokens)
and the task `config` summary; solved routes include the full `trees`.
The shipped lanes set `resume: false`: a fresh run refuses to overwrite an
existing output, so move or delete the old JSONL before rerunning. Set
`resume: true` only to continue an interrupted run; it skips existing target
rows, including failed ones.

The shipped lanes also set `concurrency: 2`, which is safe for a single
planner/step replica. Raise it only if both endpoints have spare capacity.

## Configuration reference

Global defaults live in `config.yaml`; a lane yaml overrides them wholesale.
Endpoints, API keys and asset paths are **not** stored in the configs: they
are `$VAR` / `${VAR:-default}` references resolved from the environment, with
the repo-root `.env` loaded automatically (`cp .env.example .env`). The fields
that matter:

| Section | Meaning |
|---|---|
| `starting_molecules` | Purchasable stock (standardized eMolecules sqlite) |
| `search` | Single-step call budget (`max_steps`), tree depth, target complete paths |
| `planner` | Planner model and system prompt |
| `llm_config.providers` | Planner endpoint, key, decoding (`$PLANNER_*`) |
| `single_step.retro` | Step backend (`dianshi`), endpoints (`$SINGLE_STEP_*`), `num_res`, optional cache |
| `single_step.template_filter` | The gate: `match_mode: apply`, `ranking: aot`, `blurry`, `blurry_top_n`, `unsolved_penalty` |
| `run` | Input/output paths, concurrency, resume |

Endpoints, API keys and asset paths are **not** stored in the configs: they
are `$VAR` / `${VAR:-default}` references resolved from the environment, with
the repo-root `.env` (see `.env.example`) loaded automatically. The JSONL
result files alone reproduce the evaluation — no database is involved.

## Repository layout

| Path | Description |
|---|---|
| `cli.py` | Config-driven batch evaluation entry point |
| `runtime/aot/` | AND–OR tree, dispatcher, planner prompt/response handling, workers |
| `runtime/single_step/` | Step client assembly, candidate filters, AOT\* gate; `aot_scscore/` is the vendored SCScore |
| `runtime/step_api_client.py` | In-process step client wiring (backend / gate) |
| `configs/bench/` | The three evaluation lanes |
| `scripts/serve_planner.sh`, `scripts/serve_single_step.sh` | Plain vLLM launchers for the planner and step model |
| `docs/VALIDATE.md` | Gate definitions and parameter conventions |
| `tests/` | Unit tests |

## Tests

```bash
PYTHONPATH=. python -m pytest -q
```

## Third-party code

The AOT\* SCScore NumPy implementation and the 2048-bit checkpoint are vendored
from [connorcoley/scscore](https://github.com/connorcoley/scscore) (MIT); the
gate's reranking is adapted from
[AOT\*](https://github.com/ShawnKS/AOTstar) (Song et al., 2025). Template
application uses [rdchiral](https://github.com/coleygroup/rdchiral) (installed
from PyPI). Building-block data derives from eMolecules and
[Retro\*](https://github.com/binghong-ml/retro_star) — cite the upstream works
you use.
