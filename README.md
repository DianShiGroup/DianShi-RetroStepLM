# DianShi-RetroStep

English | [中文](README.zh.md)

Single-step retrosynthesis evaluation and multi-step synthesis-route planning with **DianShi-RetroStep-LM-Direct**.

- **Single-step evaluation:** build `sl_cot` (step-by-step chain-of-thought) test datasets from DianShi-RxnStep and USPTO-50K, run vLLM beam search, and report exact-match Top-k accuracy.
- **Multi-step planning:** three frameworks under `multi_route/` use an LLM planner to search for routes to purchasable building blocks, combining the direct model with rdchiral template application and AOT* SCScore reranking.

## Hugging Face resources

| Resource | Link | Contents |
|---|---|---|
| Model | [DianShiGroup/DianShi-RetroStep-LM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStep-LM-Direct) | Single-step retrosynthesis model (Qwen3-4B fine-tuned, no retrieval) |
| Dataset | [DianShiGroup/DianShi-RetroRoute-Dataset](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroRoute-Dataset) | Multi-step runtime assets: rdchiral template library, eMolecules building-block stock, SCScore checkpoint, and the USPTO-190 / Pistachio-100 / PatMol-150 target sets |

~~~bash
# Single-step: the default --model-path of scripts/infer_beam_search.sh
hf download DianShiGroup/DianShi-RetroStep-LM-Direct --local-dir models/DianShi-RetroStep-Direct

# Multi-step: run inside the selected framework directory
cd multi_route/DianShi-RetroRoute-System-Preview
hf download DianShiGroup/DianShi-RetroRoute-Dataset --repo-type dataset --local-dir DianShi-RetroRoute-Dataset
~~~

The single-step scripts and the multi-step frameworks also accept the model ID `DianShiGroup/DianShi-RetroStep-LM-Direct` directly; vLLM downloads it on first use.

## Getting started

| Task | Entry point | Guide |
|---|---|---|
| Build single-step evaluation datasets | `scripts/build_model_tests.sh` | [Building the datasets](#building-the-datasets) |
| Run single-step inference and Top-k evaluation | `scripts/infer_beam_search.sh` | [Beam-search inference and evaluation](#beam-search-inference-and-evaluation) |
| Run multi-step route planning | A framework under `multi_route/` | [Multi-step route planning](#multi-step-route-planning) |
| Compare solve rates across search budgets | `multi_route/tools/solved_by_n.py` | [Multi-step results and statistics](#multi-step-results-and-statistics) |

## Repository layout

~~~text
DianShi-RetroStep/
  datasets/
    DianShi-RxnStep-dataset/      Test-IN, Test-OUT, Val-IN, Val-OUT splits
    USPTO-50K/                    Raw train, validation, and test splits
  multi_route/
    RetroAgent/                   Tool-calling planner with AND-OR graph memory
    Retro-R1-modified/            CALL / SELECT planning environment
    DianShi-RetroRoute-System-Preview/
                                  AND-OR tree dispatcher and benchmark CLI
    tools/solved_by_n.py          Solve rates across the three result formats
    README.md                     Multi-step framework overview
  scripts/
    build_model_tests.sh          Build sl_cot evaluation datasets
    infer_beam_search.sh          Beam-search inference + evaluation
  src/dianshi_retrostep/
    data/                         Dataset builder, atom-wise SMILES conversion
    inference/                    Multi-GPU vLLM beam search
    evaluation/                   Exact-match Top-k evaluation
    prompts/prompt_sft_sl_cot.txt sl_cot prompt template
  pyproject.toml
  requirements.txt
~~~

## Single-step evaluation

Commands in this section run from the repository root. The `scripts/` entry points switch to the repository root and add `src` to `PYTHONPATH` automatically.

### Installation

Python 3.10 or later.

~~~bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .   # optional: run the modules without the shell scripts
~~~

This environment covers the single-step pipeline only. Multi-step frameworks have their own requirements (see below).

### Building the datasets

Source data and the prompt template are included in the repository:

- `datasets/DianShi-RxnStep-dataset`: Test-IN, Test-OUT, Val-IN, Val-OUT.
- `datasets/USPTO-50K`: raw train, validation, and test splits; the builder uses validation and test.

Building runs on CPU:

~~~bash
scripts/build_model_tests.sh                                   # all datasets and splits
scripts/build_model_tests.sh --dry-run                         # print the build plan only
scripts/build_model_tests.sh --datasets test_out --splits test
scripts/build_model_tests.sh --datasets test_in,test_out,uspto_50k --splits test,val
scripts/build_model_tests.sh --force                           # rebuild existing outputs
~~~

Set `PYTHON_BIN` to use a different interpreter. Each product SMILES is converted to atom-wise form (space-separated tokens), matching the model's training format. Outputs are written under `model_tests/`:

~~~text
model_tests/
  test_in/sl_cot/
    testset_sl_cot/                   JSONL
    testset_sl_cot_new_cached/        Hugging Face cached dataset
    validationset_sl_cot/
    validationset_sl_cot_new_cached/
  test_out/sl_cot/...
  uspto_50k/sl_cot/...
~~~

### Beam-search inference and evaluation

`scripts/infer_beam_search.sh` runs vLLM beam search on an `sl_cot` cached dataset, then prints exact-match Top-k accuracy. Pass the Hugging Face model ID directly; vLLM downloads the weights on first use:

~~~bash
scripts/infer_beam_search.sh \
  --model-path DianShiGroup/DianShi-RetroStep-LM-Direct \
  --data-path model_tests/test_out/sl_cot/testset_sl_cot_new_cached \
  --num-gpus 8 \
  --batch-size 512
~~~

`--model-path` also accepts a local directory; the default is `models/DianShi-RetroStep-Direct`. To evaluate another split, pass its cached directory to `--data-path`, for example `model_tests/test_in/sl_cot/testset_sl_cot_new_cached` or `model_tests/uspto_50k/sl_cot/testset_sl_cot_new_cached`.

| Option | Default |
|---|---|
| `--data-path` | `model_tests/test_out/sl_cot/testset_sl_cot_new_cached` |
| `--output-dir` | `results` |
| `--num-gpus` / `--batch-size` | `1` / `128` |
| `--beam-width` / `--max-tokens` | `10` / `1024` |
| Top-k metrics | 1, 3, 5, 10 |
| Thinking | enabled, as in the reported experiments (`--no-thinking` to turn it off) |

Each option can also be set by its uppercase environment variable, for example `MODEL_PATH=... NUM_GPUS=4 scripts/infer_beam_search.sh`. Existing result files are protected; pass `--overwrite` to replace one.

The runner chains two modules, which can also be run separately (after `pip install -e .` or with `PYTHONPATH=src`):

- `dianshi_retrostep.inference.vllm_beamsearch_infer` keeps only the system and user messages (so the stored answer never leaks into the prompt), shards each batch across one-GPU Ray workers, runs ordered beam search, and writes one JSON line per sample with the ranked beams and cumulative log probabilities.
- `dianshi_retrostep.evaluation.eval_retro_beam_topk` extracts the answer from each beam, canonicalizes reference and predicted reactants with RDKit, ignores reactant order, and reports exact-match Top-k accuracy. It accepts `<answer>` tags, plain or atom-wise SMILES, markdown-wrapped answers, and text after `</think>`.

~~~bash
python3 -m dianshi_retrostep.inference.vllm_beamsearch_infer \
  --model-path DianShiGroup/DianShi-RetroStep-LM-Direct \
  --cached-data-path model_tests/test_out/sl_cot/testset_sl_cot_new_cached \
  --output-path results/sl_cot_test_out_beam10.jsonl \
  --num-gpus 1 \
  --beam-width 10

python3 -m dianshi_retrostep.evaluation.eval_retro_beam_topk \
  results/sl_cot_test_out_beam10.jsonl \
  --ks 1 3 5 10 \
  --output-json results/sl_cot_test_out_beam10_metrics.json
~~~

## Multi-step route planning

Each framework has its own dependencies, configuration, and `.env`. The [multi-step overview](multi_route/README.md) links to each framework's full guide.

### Frameworks

The three frameworks are independently runnable. Their shipped direct configurations use [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) as the planner and [DianShi-RetroStep-LM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStep-LM-Direct) as the single-step model.

| Framework | Search interface | Setup and configuration |
|---|---|---|
| RetroAgent | An agent calls tools backed by an HTTP server and AND-OR `GraphMemory` | [RetroAgent guide](multi_route/RetroAgent/README.md) |
| Retro-R1-modified | An in-process agent uses `CALL` to request candidates and `SELECT` to apply one | [Retro-R1 guide](multi_route/Retro-R1-modified/README.md) |
| DianShi-RetroRoute-System-Preview | A dispatcher presents legal expansion options to the planner and manages an AND-OR tree | [Preview guide](multi_route/DianShi-RetroRoute-System-Preview/README.md) |

### Shared planning pipeline

1. The planner chooses a molecule or route frontier to explore.
2. The DianShi direct model proposes single-step disconnections through an OpenAI-compatible `/v1/chat/completions` endpoint. The shipped direct configurations use the model without retrieval.
3. The AOT* gate applies rdchiral reaction templates, with a blurry matching pass when needed, then reranks candidates using SCScore synthetic complexity and an `unsolved_penalty` for reactants outside the stock. Only accepted candidates enter the search.
4. The framework checks reactant availability against a standardized eMolecules SQLite stock using canonical SMILES. A reaction is solved when all its reactants are solved; a molecule is solved when a complete decomposition is found.
5. Search stops when the target is solved or a framework-specific search limit is reached. See each framework's guide for its call budget, turn limit, depth limit, and stopping rules.

For the template-gate contract and parameters, see [VALIDATE.md](multi_route/DianShi-RetroRoute-System-Preview/docs/VALIDATE.md).

### Required models and runtime assets

| Resource | Preparation |
|---|---|
| DianShi direct single-step weights | Set `SINGLE_STEP_MODEL_PATH` in the selected framework's `.env` to `DianShiGroup/DianShi-RetroStep-LM-Direct` or a local copy. |
| Planner weights | Set `PLANNER_MODEL_PATH` to `Qwen/Qwen3.5-397B-A17B-FP8` or a local copy. vLLM can download the Hugging Face model on first use. |
| rdchiral template library | Set `DIANSHI_RDCHIRAL_TEMPLATE_JSONL` to `DianShi-RetroRoute-Dataset/rxn_template/rdchiral_res_rxn_temp_count.jsonl`. |
| Purchasable building-block stock | Set `STARTING_MOLECULES` to the standardized eMolecules SQLite file; pass `--starting_molecules` explicitly when running Retro-R1. |
| Target molecules | Provide one target SMILES per line. The benchmark target paths are listed below. |
| AOT* SCScore weights | The 2048-bit checkpoint is included under both forks' `inference/gate/weights/`. Preview expects a copy under `runtime/single_step/aot_scscore/models/full_reaxys_model_2048bool/`; the example below prepares it. RetroAgent also needs the tool-server SCScore asset described in its guide. |

The template library, stock database, SCScore checkpoint, and target files are published as the Hugging Face dataset [DianShiGroup/DianShi-RetroRoute-Dataset](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroRoute-Dataset). Download it into the selected framework directory as `DianShi-RetroRoute-Dataset/`, place assets according to the selected framework's guide, then copy its `.env.example` to `.env` and configure model paths, endpoint URLs, and API keys.

### Target sets

Paths in this table are relative to the selected framework directory.

| Benchmark | Targets | Target file |
|---|---:|---|
| USPTO-190 | 190 | `DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt` |
| Pistachio-100 | 100 | `DianShi-RetroRoute-Dataset/test_dataset/pistachio_hard_tree.txt` |
| PatMol-150 | 150 | `DianShi-RetroRoute-Dataset/test_dataset/patmol_150_targets.txt` |

RetroAgent selects targets with `--targets`; Retro-R1 supports `--targets` in its runner or `--data_file` in `cli.py`. Preview supplies a separate benchmark YAML for each target set.

### Quick start: Preview

Run the following from the repository root in a Python 3.11 or later environment:

~~~bash
cd multi_route/DianShi-RetroRoute-System-Preview
python3 -m pip install -r requirements.txt

# Download the runtime assets into this framework directory.
hf download DianShiGroup/DianShi-RetroRoute-Dataset --repo-type dataset --local-dir DianShi-RetroRoute-Dataset

# Place the repository's existing 2048-bit SCScore checkpoint where Preview expects it.
mkdir -p runtime/single_step/aot_scscore/models/full_reaxys_model_2048bool
cp ../RetroAgent/inference/gate/weights/model.ckpt-10654.as_numpy.json.gz \
  runtime/single_step/aot_scscore/models/full_reaxys_model_2048bool/

cp .env.example .env
# Edit .env: set SINGLE_STEP_MODEL_PATH, planner/step URLs, keys, and asset paths.
~~~

Serve the two models in separate terminals on the GPU machine, from the Preview directory. The serve scripts use vLLM; the component guide specifies `vllm>=0.18`, validated with `0.26`. This serving environment has different version requirements from the root single-step beam-search environment. Assign GPU devices for each service according to your deployment; the supplied defaults use planner TP=8 and single-step TP=1.

~~~bash
# Terminal 1: planner service
bash scripts/serve_planner.sh
~~~

~~~bash
# Terminal 2: single-step service
bash scripts/serve_single_step.sh
~~~

Once the configured endpoints are available, run an evaluation from the Preview directory. The evaluation process can run on CPU while the models run on the GPU machine:

~~~bash
python3 cli.py --config configs/bench/qwen_direct_uspto190.yaml
# Or evaluate the Pistachio targets:
python3 cli.py --config configs/bench/qwen_direct_pistachio100.yaml
~~~

The supplied benchmark YAMLs enable the template gate and set `max_steps: 500`, `target_complete_paths: 1`, and `concurrency: 2`. Use these benchmark YAMLs for evaluation; Preview's default `config.yaml` does not define a batch `run` section and has the gate disabled.

### Running the two forks

First follow the corresponding framework guide to install its dependencies, prepare its assets and `.env`, and start the planner and single-step services. Then run one of these commands from the repository root:

RetroAgent's tool server loads two assets from fixed relative paths, so place them before the first run:

~~~bash
(
  cd multi_route/RetroAgent
  mkdir -p dataset datasets/scscore
  cp DianShi-RetroRoute-Dataset/stock/emolecules_standardized.sqlite dataset/
  cp DianShi-RetroRoute-Dataset/scscore/model.ckpt-10654.as_numpy.json.gz datasets/scscore/
)
~~~

~~~bash
(
  cd multi_route/RetroAgent
  PYTHON_BIN="$(command -v python3)" bash scripts/run_step_eval.sh \
    --config configs/dianshi_direct.yaml \
    --targets DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt
)
~~~

~~~bash
(
  cd multi_route/Retro-R1-modified
  bash scripts/run_step_eval.sh \
    --config configs/dianshi_direct.yaml \
    --targets DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt \
    --starting_molecules DianShi-RetroRoute-Dataset/stock/emolecules_standardized.sqlite
)
~~~

RetroAgent's runner starts the configured tool server, runs the evaluation, and shuts down that server. Retro-R1's runner invokes the in-process planning environment. Both reuse existing target results on reruns; choose a new output/log directory or move existing results aside for a fresh evaluation.

### Multi-step results and statistics

Preview writes one row per target under `results/bench/<dataset>/`, including success, termination reason, model-call counts, timing, token usage, and solved route trees. The supplied benchmark YAMLs set `resume: false` and refuse to overwrite an existing output. Set `resume: true` to continue an interrupted run; existing rows, including failed targets, are skipped.

Retro-R1 writes `result.jsonl` under its configured output directory. RetroAgent writes per-target results and `summary.json` under its log directory. Consult the framework guide for complete output details.

From the repository root, compute solve rates at N = 50, 100, 200, 300, 400, and 500 using the shared statistics script:

~~~bash
python3 multi_route/tools/solved_by_n.py \
  multi_route/DianShi-RetroRoute-System-Preview/results/bench/uspto_190/qwen_direct_rdchiral_aot.jsonl

# The same script accepts Retro-R1 JSONL or RetroAgent summary.json.
python3 multi_route/tools/solved_by_n.py /path/to/result.jsonl
python3 multi_route/tools/solved_by_n.py /path/to/summary.json
~~~

The script detects the result format automatically and reports possible infrastructure failures. It computes rates over the supplied records; confirm target-set completeness and resolve reported failures before comparing runs. For RetroAgent `summary.json`, it uses the first experiment.

## License

This repository is licensed under [Apache-2.0](LICENSE), except for the two
upstream forks under `multi_route/`, which keep their own terms:

| Path | License |
|---|---|
| Everything else, including `src/` and `multi_route/DianShi-RetroRoute-System-Preview/` | Apache-2.0 ([`LICENSE`](LICENSE)) |
| `multi_route/RetroAgent/` | MIT ([`LICENSE`](multi_route/RetroAgent/LICENSE), [`NOTICE`](multi_route/RetroAgent/NOTICE)) |
| `multi_route/Retro-R1-modified/` | MIT ([`LICENSE`](multi_route/Retro-R1-modified/LICENSE)) |

Vendored dependencies keep their own licenses — see
`multi_route/Retro-R1-modified/packages/*/LICENSE` and the component READMEs.
The datasets and model weights are distributed separately under the terms stated
on their Hugging Face pages; the eMolecules building-block data in particular
has its own terms, so check them before redistributing.

## Attribution

The multi-step forks build on [RetroAgent](https://huggingface.co/SXKDZ/RetroAgent) and [Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1). Template application uses [rdchiral](https://github.com/coleygroup/rdchiral); reranking adapts [AOT*](https://github.com/ShawnKS/AOTstar) with the [SCScore](https://github.com/connorcoley/scscore) scorer and weights. Building-block data derives from eMolecules and the sources documented in each framework guide. See the component READMEs, licenses, and notices for citations and third-party terms.
