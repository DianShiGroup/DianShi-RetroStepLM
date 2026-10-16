# DianShi-RetroStep

This repository contains the DianShi-RxnStep and USPTO-50K evaluation data, plus a local pipeline that builds sl_cot (step-by-step chain-of-thought) model-test datasets, runs vLLM beam-search inference, and evaluates Top-k retrosynthesis accuracy.

## Repository layout

~~~text
DianShi-RetroStep/
  datasets/
    DianShi-RxnStep-dataset/      Test-IN, Test-OUT, Val-IN, Val-OUT splits
    USPTO-50K/                    Raw train, validation, and test splits
  scripts/
    build_model_tests.sh          Entry point: build sl_cot evaluation datasets
    infer_beam_search.sh          Entry point: beam-search inference + evaluation
  src/
    dianshi_retrostep/
      data/
        build_model_tests.py      Dataset builder
        smiles_atomwise.py        Atom-wise SMILES conversion
      inference/
        vllm_beamsearch_infer.py  Multi-GPU vLLM beam search
      evaluation/
        eval_retro_beam_topk.py   Exact-match Top-k evaluation
      prompts/
        prompt_sft_sl_cot.txt     sl_cot prompt template
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

## Building the evaluation datasets

Dataset building runs locally on CPU and needs no extra configuration:

~~~bash
scripts/build_model_tests.sh
~~~

Set PYTHON_BIN if a different Python executable is needed:

~~~bash
PYTHON_BIN=python scripts/build_model_tests.sh
~~~

Useful examples:

~~~bash
scripts/build_model_tests.sh --dry-run
scripts/build_model_tests.sh --datasets test_out --splits test
scripts/build_model_tests.sh --datasets test_in,test_out,uspto_50k --splits test,val
scripts/build_model_tests.sh --force
~~~

Existing outputs are skipped unless --force is given.

## Output naming

Outputs are written under model_tests:

~~~text
model_tests/
  test_in/
    sl_cot/
      testset_sl_cot/                       JSONL
      testset_sl_cot_atomwise/              JSONL with atom-wise SMILES
      testset_sl_cot_new_cached/            Hugging Face cached dataset
      testset_sl_cot_atomwise_new_cached/   Hugging Face cached dataset, atom-wise
      validationset_sl_cot/
      validationset_sl_cot_atomwise/
      validationset_sl_cot_new_cached/
      validationset_sl_cot_atomwise_new_cached/
  test_out/
    sl_cot/
      ...
  uspto_50k/
    sl_cot/
      ...
~~~

## Input data

- datasets/DianShi-RxnStep-dataset contains the Test-IN, Test-OUT, Val-IN, and Val-OUT splits.
- datasets/USPTO-50K contains the raw train, validation, and test splits. The builder uses validation and test.

## Beam-search inference and evaluation

The local inference entry point runs vLLM beam search on an sl_cot cached dataset and then automatically prints exact-match Top-k accuracy:

~~~bash
scripts/infer_beam_search.sh
~~~

The pipeline contains two Python modules:

- dianshi_retrostep.inference.vllm_beamsearch_infer loads a Hugging Face cached dataset, keeps only the system and user messages, distributes each batch across one-GPU Ray workers, runs ordered vLLM beam search, and writes one JSON object per sample. Each output row contains the original product/reactant metadata and the ranked generated beams with cumulative log probabilities.
- dianshi_retrostep.evaluation.eval_retro_beam_topk reads that JSONL output, canonicalizes the reference and predicted reactant SMILES with RDKit, ignores reactant component order, and reports exact-match Top-k accuracy. It extracts the final answer from the answer tag of the sl_cot output, and also accepts direct SMILES, atom-wise SMILES, markdown-wrapped answers, and answers following a closing think tag.

Its portable defaults are:

- Model: models/DianShi-RetroStep-Direct
- Cached dataset: model_tests/test_out/sl_cot/testset_sl_cot_atomwise_new_cached
- Results directory: results
- Beam width: 10
- Top-k metrics: 1, 3, 5, and 10

The model may be specified with a repository-relative path, an absolute local path, or a Hugging Face model ID:

~~~bash
scripts/infer_beam_search.sh \
  --model-path /path/to/DianShi-RetroStep/models/DianShi-RetroStep-Direct \
  --data-path model_tests/test_out/sl_cot/testset_sl_cot_atomwise_new_cached \
  --num-gpus 8 \
  --batch-size 512
~~~

To evaluate another sl_cot split, pass its cached directory through --data-path, for example model_tests/test_in/sl_cot/testset_sl_cot_atomwise_new_cached or model_tests/uspto_50k/sl_cot/testset_sl_cot_atomwise_new_cached.

The JSONL inference output and its metrics JSON are written under results by default. Existing result files are protected; pass --overwrite to replace one.

The modules can also be run separately (after pip install -e ., or with PYTHONPATH=src):

~~~bash
python3 -m dianshi_retrostep.inference.vllm_beamsearch_infer \
  --model-path models/DianShi-RetroStep-Direct \
  --cached-data-path model_tests/test_out/sl_cot/testset_sl_cot_atomwise_new_cached \
  --output-path results/sl_cot_test_out_beam10.jsonl \
  --num-gpus 1 \
  --beam-width 10 \
  --no-thinking

python3 -m dianshi_retrostep.evaluation.eval_retro_beam_topk \
  results/sl_cot_test_out_beam10.jsonl \
  --ks 1 3 5 10 \
  --output-json results/sl_cot_test_out_beam10_metrics.json
~~~

Environment variables with the same uppercase names can also configure the runner, for example:

~~~bash
MODEL_PATH=models/DianShi-RetroStep-Direct \
NUM_GPUS=4 \
scripts/infer_beam_search.sh
~~~
