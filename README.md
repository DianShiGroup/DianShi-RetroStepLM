# DianShi-RetroStep

This repository contains the DianShi-RxnStep and USPTO-50K evaluation data, plus a local pipeline that builds model-test datasets, runs vLLM beam-search inference, and evaluates Top-k retrosynthesis accuracy.

## Repository layout

~~~text
DianShi-RetroStep/
  configs/
    model_test_config.json        ChemRAG database configuration
  datasets/
    DianShi-RxnStep-dataset/      Test-IN, Test-OUT, Val-IN, Val-OUT splits
    USPTO-50K/                    Raw train, validation, and test splits
  scripts/
    build_model_tests.sh          Entry point: build evaluation datasets
    infer_beam_search.sh          Entry point: beam-search inference + evaluation
  src/
    dianshi_retrostep/
      data/
        build_model_tests.py      Dataset builder
        insert_rag_into_test.py   ChemRAG retrieval
        smiles_atomwise.py        Atom-wise SMILES conversion
      inference/
        vllm_beamsearch_infer.py  Multi-GPU vLLM beam search
      evaluation/
        eval_retro_beam_topk.py   Exact-match Top-k evaluation
      prompts/                    System prompt templates
  pyproject.toml
  requirements.txt
~~~

All commands below are run from the repository root. The shell scripts change to the repository root automatically and add src to PYTHONPATH, so they can also be called from any directory.

## Installation

Python 3.10 or later is recommended.

~~~bash
python3 -m pip install -r requirements.txt
~~~

Optionally, install the package in editable mode to run the modules without the shell scripts:

~~~bash
python3 -m pip install -e .
~~~

The pipeline uses local CPU processes. A ChemRAG database is stored separately because its index files are large.

## Required database configuration

The db field in configs/model_test_config.json is required and must be a non-empty list before the pipeline can run. It is intentionally empty in the downloaded repository.

Each list item must contain:

- name: a stable ASCII label used in output directory names.
- path: the ChemRAG database directory, written relative to the repository root. Absolute paths are rejected so the configuration remains portable.

Example:

~~~json
{
  "db": [
    {
      "name": "Uretro",
      "path": "../chemrag-data/chemrag_db"
    },
    {
      "name": "usptodb",
      "path": "../chemrag-data/chemrag_uspto_db"
    }
  ]
}
~~~

Common labels from the established model_tests layout are Uretro, usptodb, le_2016, and template_remain_180w. A label such as Uretro creates directory names such as testset_s2s_Uretro_rag and testset_sl_cot_Uretro_rag.

Each database directory must contain config.json, metadata.jsonl, product_fps.npy, line_offsets.npy, and faiss_product.index. If config.json points to an unavailable source JSONL file, the pipeline falls back to metadata.jsonl.

## Local use

After filling the db list, run:

~~~bash
scripts/build_model_tests.sh
~~~

The shell script changes to the repository root automatically and only uses repository-relative entry paths. A different configuration file can be passed with --config. Set PYTHON_BIN if a different Python executable is needed:

~~~bash
PYTHON_BIN=python scripts/build_model_tests.sh
~~~

Useful examples:

~~~bash
scripts/build_model_tests.sh --dry-run
scripts/build_model_tests.sh --datasets uspto_50k --splits test
scripts/build_model_tests.sh --variants s2s,sl_cot
scripts/build_model_tests.sh --variants s2s_rag_Uretro,sl_cot_rag_Uretro
scripts/build_model_tests.sh --force
~~~

The available RAG variant keys are generated from the database names in configs/model_test_config.json.

## Output naming

Outputs are written under model_tests and follow the established naming convention:

~~~text
model_tests/
  test_in/
    s2s/
      testset_s2s/
      testset_s2s_atomwise/
      testset_s2s_new_cached/
      testset_s2s_atomwise_new_cached/
      testset_s2s_Uretro_rag/
      testset_s2s_Uretro_rag_atomwise/
      testset_s2s_Uretro_rag_new_cached/
      testset_s2s_Uretro_rag_atomwise_new_cached/
    sl_cot/
      testset_sl_cot/
      testset_sl_cot_atomwise/
      testset_sl_cot_new_cached/
      testset_sl_cot_atomwise_new_cached/
    sl_cot_rag/
      testset_sl_cot_Uretro_rag/
      testset_sl_cot_Uretro_rag_atomwise/
      testset_sl_cot_Uretro_rag_new_cached/
      testset_sl_cot_Uretro_rag_atomwise_new_cached/
~~~

The same pattern is used for test_out and uspto_50k. Validation outputs begin with validationset instead of testset.

Temporary retrieval results are stored under rag_cache. Generated model_tests and rag_cache directories are not required as inputs.

## Input data

- datasets/DianShi-RxnStep-dataset contains the Test-IN, Test-OUT, Val-IN, and Val-OUT splits.
- datasets/USPTO-50K contains the raw train, validation, and test splits. The builder uses validation and test.

## Beam-search inference and evaluation

The local inference entry point runs vLLM beam search and then automatically prints exact-match Top-k accuracy:

~~~bash
scripts/infer_beam_search.sh
~~~

The pipeline contains two Python modules:

- dianshi_retrostep.inference.vllm_beamsearch_infer loads a Hugging Face cached dataset, keeps only the system and user messages, distributes each batch across one-GPU Ray workers, runs ordered vLLM beam search, and writes one JSON object per sample. Each output row contains the original product/reactant metadata and the ranked generated beams with cumulative log probabilities.
- dianshi_retrostep.evaluation.eval_retro_beam_topk reads that JSONL output, canonicalizes the reference and predicted reactant SMILES with RDKit, ignores reactant component order, and reports exact-match Top-k accuracy. It accepts direct SMILES, atom-wise SMILES, answer-tag output, markdown-wrapped answers, and answers following a closing think tag.

Its portable defaults are:

- Model: models/DianShi-RetroStep-Direct
- Cached dataset: model_tests/test_out/s2s/testset_s2s_atomwise_new_cached
- Results directory: results
- Beam width: 10
- Top-k metrics: 1, 3, 5, and 10

The model may be specified with a repository-relative path, an absolute local path, or a Hugging Face model ID:

~~~bash
scripts/infer_beam_search.sh \
  --model-path /path/to/DianShi-RetroStep/models/DianShi-RetroStep-Direct \
  --data-path model_tests/test_out/s2s/testset_s2s_atomwise_new_cached \
  --num-gpus 8 \
  --batch-size 512
~~~

To evaluate another generated cached dataset, pass its directory through --data-path. The path should follow the model_tests naming layout documented above.

The JSONL inference output and its metrics JSON are written under results by default. Existing result files are protected; pass --overwrite to replace one. Direct SMILES, answer-tag output, atom-wise SMILES, and output following a closing think tag are all supported by the evaluator.

The modules can also be run separately (after pip install -e ., or with PYTHONPATH=src):

~~~bash
python3 -m dianshi_retrostep.inference.vllm_beamsearch_infer \
  --model-path models/DianShi-RetroStep-Direct \
  --cached-data-path model_tests/test_out/s2s/testset_s2s_atomwise_new_cached \
  --output-path results/direct_test_out_beam10.jsonl \
  --num-gpus 1 \
  --beam-width 10 \
  --no-thinking

python3 -m dianshi_retrostep.evaluation.eval_retro_beam_topk \
  results/direct_test_out_beam10.jsonl \
  --ks 1 3 5 10 \
  --output-json results/direct_test_out_beam10_metrics.json
~~~

Environment variables with the same uppercase names can also configure the runner, for example:

~~~bash
MODEL_PATH=models/DianShi-RetroStep-Direct \
NUM_GPUS=4 \
scripts/infer_beam_search.sh
~~~
