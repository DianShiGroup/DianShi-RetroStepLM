# DianShi-RetroStep: Beam-search inference

The complete installation, dataset-building, and output-naming guide is available in README.md. This file describes the beam-search inference and evaluation programs.

Source code lives under src/dianshi_retrostep (data, inference, evaluation, prompts) and entry-point scripts under scripts. Only the sl_cot (step-by-step chain-of-thought) datasets are built and evaluated. All commands are run from the repository root.

## Programs

### dianshi_retrostep.inference.vllm_beamsearch_infer

This program:

- Loads an sl_cot Hugging Face cached dataset generated under model_tests.
- Keeps only the system and user messages, preventing a stored ground-truth assistant message from leaking into inference.
- Distributes each batch across one-GPU Ray workers.
- Runs ordered vLLM beam search.
- Writes one JSON object per sample, preserving product/reactant metadata and storing ranked beams with cumulative log probabilities.

Required arguments are --model-path, --cached-data-path, and --output-path. GPU count, batch size, maximum generated tokens, beam width, and GPU memory utilization are configurable.

### dianshi_retrostep.evaluation.eval_retro_beam_topk

This program reads the beam-search JSONL output and reports exact-match Top-k retrosynthesis accuracy. It canonicalizes reference and predicted reactant SMILES with RDKit and treats dot-separated reactant component order as chemically equivalent.

Recognized prediction formats include:

- Direct reactant SMILES.
- Atom-wise SMILES with spaces between tokens.
- Answers enclosed in answer tags.
- Markdown-wrapped answers.
- Answers following a closing think tag.

It prints accuracy and diagnostics to the terminal and can optionally save the complete metrics as JSON.

## Recommended local runner

scripts/infer_beam_search.sh runs both programs in sequence: inference first, followed by Top-k evaluation.

Its defaults are:

- Model: models/DianShi-RetroStep-Direct
- Dataset: model_tests/test_out/sl_cot/testset_sl_cot_new_cached
- Output directory: results
- Beam width: 10
- Metrics: Top-1, Top-3, Top-5, and Top-10

~~~bash
scripts/infer_beam_search.sh
~~~

An explicit model path can be supplied:

~~~bash
scripts/infer_beam_search.sh \
  --model-path /path/to/DianShi-RetroStep/models/DianShi-RetroStep-Direct \
  --data-path model_tests/test_out/sl_cot/testset_sl_cot_new_cached \
  --num-gpus 8 \
  --batch-size 512
~~~

## Running the Python modules separately

Install the package with python3 -m pip install -e . first, or prefix the commands with PYTHONPATH=src.

~~~bash
python3 -m dianshi_retrostep.inference.vllm_beamsearch_infer \
  --model-path models/DianShi-RetroStep-Direct \
  --cached-data-path model_tests/test_out/sl_cot/testset_sl_cot_new_cached \
  --output-path results/sl_cot_test_out_beam10.jsonl \
  --num-gpus 1 \
  --beam-width 10 \
  --no-thinking

python3 -m dianshi_retrostep.evaluation.eval_retro_beam_topk \
  results/sl_cot_test_out_beam10.jsonl \
  --ks 1 3 5 10 \
  --output-json results/sl_cot_test_out_beam10_metrics.json
~~~
