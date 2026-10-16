# Multi-step planning frameworks

This folder hosts three independent multi-step retrosynthesis planning
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
   public target sets ship as one tarball; each component README's assets
   table says where to place them.
   TODO(release): replace the internal asset paths with public download links.

## The three frameworks

| Folder | Lineage | Search harness | Planner |
|---|---|---|---|
| [`RetroAgent/`](RetroAgent/README.md) | fork of [SXKDZ/RetroAgent](https://huggingface.co/SXKDZ/RetroAgent) (COLM 2026) | HTTP tool server + AND-OR `GraphMemory` harness; agent issues tool calls | Qwen3.5-397B |
| [`Retro-R1-modified/`](Retro-R1-modified/README.md) | fork of [GenSI-THUAIR/Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1) (NeurIPS 2025) | in-process `CALL` / `SELECT` agent loop | Qwen3.5-397B |
| [`DianShi-RetroRoute-System-Preview/`](DianShi-RetroRoute-System-Preview/README.md) | original in-process bench harness | AND-OR colony dispatcher + workers, JSONL output | Qwen3.5-397B |

Each folder is fully self-contained: its own README (quick start, assets
table, configuration reference), `requirements.txt`, serve scripts,
configuration and tests. The template-gate code is intentionally duplicated
in the two forks — they descend from different upstreams and stay
independently runnable.

## Quick start (common shape)

```bash
# 0. per-component environment      see the component README
# 1. unpack the runtime assets      tarball; placement per component README
# 2. serve the two models           vLLM; serve scripts included in each component
# 3. run the evaluation             single command per component
```

Follow the component README for the exact commands — each one is a
three-step flow: environment, assets + `.env`, serve + evaluate.

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
