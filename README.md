# DianShi-RetroStep

English | [中文](README.zh.md)

Single-step retrosynthesis evaluation and multi-step synthesis-route planning with **DianShi-RetroStepLM-Direct**.

- **Single-step evaluation:** build intermediate-label test datasets from DianShi-PatentRxnStep-Dataset and USPTO-50K, run vLLM beam search, and report exact-match Top-k accuracy. The prompt asks for synthons, then leaving groups, then the final reactants, which is the intermediate-label format the model was trained on.
- **Multi-step planning:** three frameworks under `multi_route/` use an LLM planner to search for routes to purchasable building blocks, combining RetroStepLM-Direct with rdchiral template application and AOT* SCScore reranking.

## Hugging Face resources

| Resource | Link | Contents |
|---|---|---|
| Model | [DianShiGroup/DianShi-RetroStepLM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStepLM-Direct) | Single-step retrosynthesis model (Qwen3-4B-Base fine-tuned, no retrieval) |
| Dataset | [DianShiGroup/DianShi-RetroPlanSystem-Assets](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroPlanSystem-Assets) | Multi-step runtime assets: rdchiral template library, eMolecules building-block stock, and the USPTO-190 / Pistachio Hard / PatMol-150 target sets |

~~~bash
# Single-step: the default --model-path of scripts/infer_beam_search.sh
hf download DianShiGroup/DianShi-RetroStepLM-Direct --local-dir models/DianShi-RetroStepLM-Direct

# Multi-step: run inside the selected framework directory
cd multi_route/DianShi-RetroPlanSystem-v0
hf download DianShiGroup/DianShi-RetroPlanSystem-Assets --repo-type dataset --local-dir DianShi-RetroPlanSystem-Assets
~~~

The single-step scripts and the multi-step frameworks also accept the model ID `DianShiGroup/DianShi-RetroStepLM-Direct` directly; vLLM downloads it on first use.

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
    DianShi-PatentRxnStep-Dataset/
                                  Test-2001-2024, Test-2025, Val-2001-2024, Val-2025
    USPTO-50K/                    Raw train, validation, and test splits
  multi_route/
    RetroAgent/                   Tool-calling planner with AND-OR graph memory
    Retro-R1-modified/            CALL / SELECT planning environment
    DianShi-RetroPlanSystem-v0/   AND-OR tree dispatcher and benchmark CLI
    tools/solved_by_n.py          Solve rates across the three result formats
    README.md                     Multi-step framework overview
  scripts/
    build_model_tests.sh          Build intermediate-label evaluation datasets
    infer_beam_search.sh          Beam-search inference + evaluation
  src/dianshi_retrostep/
    data/                         Dataset builder, atom-wise SMILES conversion
    inference/                    Multi-GPU vLLM beam search
    evaluation/                   Exact-match Top-k evaluation
    prompts/prompt_sft_intermediate_label.txt
                                  Intermediate-label prompt template
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

- `datasets/DianShi-PatentRxnStep-Dataset`: four evaluation partitions. Test-2001-2024 and Val-2001-2024 (3,798 reactions each) cover reactions recorded in 2001–2024; Test-2025 and Val-2025 (1,886 reactions each) are temporally held out, with every reaction recorded in 2025.
- `datasets/USPTO-50K`: raw train, validation, and test splits; the builder uses validation and test.

Building runs on CPU. Pass `test_2001_2024`, `test_2025` or `uspto_50k` to `--datasets`; each builds its test and validation partitions:

~~~bash
scripts/build_model_tests.sh                                   # all datasets and splits
scripts/build_model_tests.sh --dry-run                         # print the build plan only
scripts/build_model_tests.sh --datasets test_2025 --splits test
scripts/build_model_tests.sh --datasets test_2001_2024,test_2025,uspto_50k --splits test,val
scripts/build_model_tests.sh --force                           # rebuild existing outputs
~~~

Set `PYTHON_BIN` to use a different interpreter. Each product SMILES is converted to atom-wise form (space-separated tokens), matching the model's training format. Outputs are written under `model_tests/`:

~~~text
model_tests/
  test_2001_2024/intermediate_label/
    testset_intermediate_label/                   JSONL
    testset_intermediate_label_new_cached/        Hugging Face cached dataset
    validationset_intermediate_label/
    validationset_intermediate_label_new_cached/
  test_2025/intermediate_label/...
  uspto_50k/intermediate_label/...
~~~

### Beam-search inference and evaluation

`scripts/infer_beam_search.sh` runs vLLM beam search on an intermediate-label cached dataset, then prints exact-match Top-k accuracy. Pass the Hugging Face model ID directly; vLLM downloads the weights on first use:

~~~bash
scripts/infer_beam_search.sh \
  --model-path DianShiGroup/DianShi-RetroStepLM-Direct \
  --data-path model_tests/test_2025/intermediate_label/testset_intermediate_label_new_cached \
  --num-gpus 8 \
  --batch-size 512
~~~

`--model-path` also accepts a local directory; the default is `models/DianShi-RetroStepLM-Direct`. To evaluate another split, pass its cached directory to `--data-path`, for example `model_tests/test_2001_2024/intermediate_label/testset_intermediate_label_new_cached` or `model_tests/uspto_50k/intermediate_label/testset_intermediate_label_new_cached`. Without `--output-path`, results go to `results/<model>_<dataset>_<split>_beam<width>.jsonl`, so each test set gets its own file.

| Option | Default |
|---|---|
| `--data-path` | `model_tests/test_2025/intermediate_label/testset_intermediate_label_new_cached` |
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
  --model-path DianShiGroup/DianShi-RetroStepLM-Direct \
  --cached-data-path model_tests/test_2025/intermediate_label/testset_intermediate_label_new_cached \
  --output-path results/test_2025_intermediate_label_beam10.jsonl \
  --num-gpus 1 \
  --beam-width 10

python3 -m dianshi_retrostep.evaluation.eval_retro_beam_topk \
  results/test_2025_intermediate_label_beam10.jsonl \
  --ks 1 3 5 10 \
  --output-json results/test_2025_intermediate_label_beam10_metrics.json
~~~

## Multi-step route planning

Each framework has its own dependencies, configuration, and `.env`. The [multi-step overview](multi_route/README.md) links to each framework's full guide.

### Frameworks

The three frameworks are independently runnable. Their shipped direct configurations use [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) as the planner and [DianShi-RetroStepLM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStepLM-Direct) as the single-step model.

| Framework | Search interface | Setup and configuration |
|---|---|---|
| RetroAgent | An agent calls tools backed by an HTTP server and AND-OR `GraphMemory` | [RetroAgent guide](multi_route/RetroAgent/README.md) |
| Retro-R1-modified | An in-process agent uses `CALL` to request candidates and `SELECT` to apply one | [Retro-R1 guide](multi_route/Retro-R1-modified/README.md) |
| DianShi-RetroPlanSystem-v0 | A dispatcher presents legal expansion options to the planner and manages an AND-OR tree | [DianShi-RetroPlanSystem-v0 guide](multi_route/DianShi-RetroPlanSystem-v0/README.md) |

### Shared planning pipeline

1. The planner chooses a molecule or route frontier to explore.
2. RetroStepLM-Direct proposes single-step disconnections through an OpenAI-compatible `/v1/chat/completions` endpoint. The shipped direct configurations use the model without retrieval.
3. The AOT* gate applies rdchiral reaction templates, with a blurry matching pass when needed, then reranks candidates using SCScore synthetic complexity and an `unsolved_penalty` for reactants outside the stock. Only accepted candidates enter the search. The paper calls this step AOT*-style template-based reaction validation.
4. The framework checks reactant availability against a standardized eMolecules SQLite stock using canonical SMILES. A reaction is solved when all its reactants are solved; a molecule is solved when a complete decomposition is found.
5. Search stops when the target is solved or a framework-specific search limit is reached. See each framework's guide for its call budget, turn limit, depth limit, and stopping rules.

For the template-gate contract and parameters, see [VALIDATE.md](multi_route/DianShi-RetroPlanSystem-v0/docs/VALIDATE.md).

### Required models and runtime assets

| Resource | Preparation |
|---|---|
| RetroStepLM-Direct single-step weights | Set `SINGLE_STEP_MODEL_PATH` in the selected framework's `.env` to `DianShiGroup/DianShi-RetroStepLM-Direct` or a local copy. |
| Planner weights | Set `PLANNER_MODEL_PATH` to `Qwen/Qwen3.5-397B-A17B-FP8` or a local copy. vLLM can download the Hugging Face model on first use. |
| rdchiral template library | Set `DIANSHI_RDCHIRAL_TEMPLATE_JSONL` to `DianShi-RetroPlanSystem-Assets/rxn_template/rdchiral_res_rxn_temp_count.jsonl`. |
| Purchasable building-block stock | Set `STARTING_MOLECULES` to the standardized eMolecules SQLite file (23,081,629 building blocks); pass `--starting_molecules` explicitly when running Retro-R1. The stock comes from the Retro* eMolecules building-block list, standardized with RDKit 2025.9.5, and then converted to SQLite. |
| Target molecules | Provide one target SMILES per line. The benchmark target paths are listed below. |
| AOT* SCScore weights | The 2048-bit gate checkpoint is included with each framework's gate assets. The 1024-bit tool-server checkpoint is vendored under `RetroAgent/datasets/scscore/`. Neither needs an extra download or copy. |

The template library, stock database, and target files are published as the Hugging Face dataset [DianShiGroup/DianShi-RetroPlanSystem-Assets](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroPlanSystem-Assets). Download it into the selected framework directory as `DianShi-RetroPlanSystem-Assets/`, place assets according to the selected framework's guide, then copy its `.env.example` to `.env` and configure model paths, endpoint URLs, and API keys.

### Target sets

Paths in this table are relative to the selected framework directory.

| Benchmark | Targets | Target file |
|---|---:|---|
| USPTO-190 | 190 | `DianShi-RetroPlanSystem-Assets/test_dataset/uspto_190_targets_smiles.txt` |
| Pistachio Hard | 100 | `DianShi-RetroPlanSystem-Assets/test_dataset/pistachio_hard_tree.txt` |
| PatMol-150 | 150 | `DianShi-RetroPlanSystem-Assets/test_dataset/patmol_150_targets.txt` |

RetroAgent selects targets with `--targets`; Retro-R1 supports `--targets` in its runner or `--data_file` in `cli.py`. DianShi-RetroPlanSystem-v0 supplies a separate benchmark YAML for each target set.

### Quick start: DianShi-RetroPlanSystem-v0

Run the following from the repository root in a Python 3.11 or later environment:

~~~bash
cd multi_route/DianShi-RetroPlanSystem-v0
python3 -m pip install -r requirements.txt

# Download the runtime assets into this framework directory.
hf download DianShiGroup/DianShi-RetroPlanSystem-Assets --repo-type dataset --local-dir DianShi-RetroPlanSystem-Assets

cp .env.example .env
# Edit .env: set SINGLE_STEP_MODEL_PATH, planner/step URLs, keys, and asset paths.
~~~

Serve the two models in separate terminals on the GPU machine, from the DianShi-RetroPlanSystem-v0 directory. The serve scripts use vLLM; the component guide specifies `vllm>=0.18`, validated with `0.26`. This serving environment has different version requirements from the root single-step beam-search environment. Assign GPU devices for each service according to your deployment; the supplied defaults use planner TP=8 and single-step TP=1.

~~~bash
# Terminal 1: planner service
bash scripts/serve_planner.sh
~~~

~~~bash
# Terminal 2: single-step service
bash scripts/serve_single_step.sh
~~~

Once the configured endpoints are available, run an evaluation from the DianShi-RetroPlanSystem-v0 directory. The evaluation process can run on CPU while the models run on the GPU machine:

~~~bash
python3 cli.py --config configs/bench/qwen_direct_uspto190.yaml
# Or evaluate the Pistachio Hard targets:
python3 cli.py --config configs/bench/qwen_direct_pistachio_hard.yaml
~~~

The supplied benchmark YAMLs enable the template gate and set `max_steps: 500`, `target_complete_paths: 1`, and `concurrency: 2`. Use these benchmark YAMLs for evaluation; the framework's default `config.yaml` does not define a batch `run` section and has the gate disabled.

### Running the two forks

First follow the corresponding framework guide to install its dependencies, prepare its assets and `.env`, and start the planner and single-step services. Then run one of these commands from the repository root:

RetroAgent's tool server loads the stock from a fixed relative path, so place it before the first run:

~~~bash
(
  cd multi_route/RetroAgent
  mkdir -p dataset
  cp DianShi-RetroPlanSystem-Assets/stock/emolecules_standardized.sqlite dataset/
)
~~~

~~~bash
(
  cd multi_route/RetroAgent
  PYTHON_BIN="$(command -v python3)" bash scripts/run_step_eval.sh \
    --config configs/dianshi_direct.yaml \
    --targets DianShi-RetroPlanSystem-Assets/test_dataset/uspto_190_targets_smiles.txt
)
~~~

~~~bash
(
  cd multi_route/Retro-R1-modified
  bash scripts/run_step_eval.sh \
    --config configs/dianshi_direct.yaml \
    --targets DianShi-RetroPlanSystem-Assets/test_dataset/uspto_190_targets_smiles.txt \
    --starting_molecules DianShi-RetroPlanSystem-Assets/stock/emolecules_standardized.sqlite
)
~~~

RetroAgent's runner starts the configured tool server, runs the evaluation, and shuts down that server. Retro-R1's runner invokes the in-process planning environment. Both reuse existing target results on reruns; choose a new output/log directory or move existing results aside for a fresh evaluation.

### Multi-step results and statistics

DianShi-RetroPlanSystem-v0 writes one row per target under `results/bench/<dataset>/`, including success, termination reason, model-call counts, timing, token usage, and solved route trees. The supplied benchmark YAMLs set `resume: false` and refuse to overwrite an existing output. Set `resume: true` to continue an interrupted run; existing rows, including failed targets, are skipped.

Retro-R1 writes `result.jsonl` under its configured output directory. RetroAgent writes per-target results and `summary.json` under its log directory. Consult the framework guide for complete output details.

From the repository root, compute solve rates at N = 50, 100, 200, 300, 400, and 500 using the shared statistics script:

~~~bash
python3 multi_route/tools/solved_by_n.py \
  multi_route/DianShi-RetroPlanSystem-v0/results/bench/uspto_190/qwen_direct_rdchiral_aot.jsonl

# The same script accepts Retro-R1 JSONL or RetroAgent summary.json.
python3 multi_route/tools/solved_by_n.py /path/to/result.jsonl
python3 multi_route/tools/solved_by_n.py /path/to/summary.json
~~~

The script detects the result format automatically and reports possible infrastructure failures. It computes rates over the supplied records; confirm target-set completeness and resolve reported failures before comparing runs. For RetroAgent `summary.json`, it uses the first experiment.

## License

The whole repository, including both forks under `multi_route/`, is released
under [Apache-2.0](LICENSE).

`RetroAgent` and `Retro-R1-modified` build on upstream projects released under
the MIT license. As MIT requires, the upstream copyright notices are kept in
[`multi_route/RetroAgent/LICENSE`](multi_route/RetroAgent/LICENSE) (with
[`NOTICE`](multi_route/RetroAgent/NOTICE)) and
[`multi_route/Retro-R1-modified/LICENSE`](multi_route/Retro-R1-modified/LICENSE).
Vendored third-party packages under `multi_route/Retro-R1-modified/packages/`
keep their own license files.

The model weights are released under Apache-2.0, the license of the
Qwen3-4B-Base model they are fine-tuned from. The runtime-assets dataset
contains third-party data (the eMolecules raw-material stock and the USPTO-190
and Pistachio Hard benchmarks) that remain under their original terms; see its
Hugging Face page before redistributing.

## Attribution

The multi-step forks build on [RetroAgent](https://huggingface.co/SXKDZ/RetroAgent) and [Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1). Template application uses [rdchiral](https://github.com/coleygroup/rdchiral); reranking adapts [AOT*](https://github.com/ShawnKS/AOTstar) with the [SCScore](https://github.com/connorcoley/scscore) scorer and weights. Building-block data derives from eMolecules and the sources documented in each framework guide. See the component READMEs, licenses, and notices for citations and third-party terms.
