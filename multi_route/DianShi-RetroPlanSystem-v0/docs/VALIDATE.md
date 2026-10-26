# Single-step candidate gate

The shipped lanes use the AOT\*-style template gate before a single-step
candidate enters the search. The gate is configured in
`configs/bench/qwen_direct_*.yaml` (DianShi-RetroPlanSystem-v0) and `configs/dianshi_direct.yaml`
(RetroAgent / Retro-R1-modified).

## Pipeline

1. **Product preselection.** Compute a product PatternFP and keep templates
   whose Tversky containment score is 1 (`alpha=0`, `beta=1`).
2. **Structural matching.** Require one-to-one SMARTS matching on both the
   product and reactant sides.
3. **Template application.** Apply up to 256 matching templates with rdchiral
   and rewrite the candidate reactants.
4. **Blurry fallback.** If the strict pass yields no candidate, rank the
   product-preselected templates by reactant-side PatternFP similarity, take
   the top 1,000, and score the resulting reactant sets with the AOT\*-style
   reward.
5. **Output handling.** Discard rewritten outputs in which the target product,
   or any of its components, appears as a reactant.

## Parameters

| Parameter | Value |
|---|---|
| `template_filter.enabled` | `true` |
| `template_filter.backend` | `rdchiral` |
| `template_filter.match_mode` | `apply` |
| `apply_max_templates` | `256` |
| `blurry` | `true` |
| `blurry_top_n` | `1000` |
| `ranking` | `aot` |
| `unsolved_penalty` | `0.1` (applied only on the blurry fallback path) |

The reward is an AOT\*-style reactant score over non-stock reactants. The
256-template apply cap, the 1,000-candidate fallback pool and `gamma = 0.1`
are the settings of this implementation; AOT\* itself has different defaults.

## Library and stock

- Template library: `rxn_template/rdchiral_res_rxn_temp_count.jsonl` from the
  DianShi-RetroPlanSystem-Assets bundle (412,027 entries), exposed as
  `DIANSHI_RDCHIRAL_TEMPLATE_JSONL`.
- Purchasable stock: `stock/emolecules_standardized.sqlite`
  (23,081,629 entries), exposed as `STARTING_MOLECULES`.

## Not used in the shipped lanes

The forward round-trip filter (`forward_filter.enabled: false`), the
LocalTemplate backend and any rule-based fallback are not part of the shipped
configurations.
