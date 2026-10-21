
# 单步候选 Gate

每个反向单步候选在写入 AND-OR tree 前，只经过以下两个 Gate。只有 `PASS` 会留下；`FAIL` 和 `UNKNOWN`（超时、报错、无法抽取或配置缺失）都丢掉。

## 1. 前向 Round-trip Gate

- 输入：规范化、排序并用 `.` 拼接的反应物 SMILES。
- 模型：`/path/to/forward_model_checkpoint`。
- Prompt：`/path/to/forward_prompt.txt`（实际文件为 `.txt`）。
- 前向模型使用 beam search 生成 top-3 产物。将 `<answer>` 中的产物和当前产物都做 RDKit 规范化；当前产物出现在 top-3 中则 `PASS`，否则 `FAIL`。
- 如果前向服务超时、报错或没有有效 `<answer>`，结果为 `FAIL`，丢掉候选。

## 2. 反应模板 Gate

由 `single_step.template_filter.match_mode` 选择怎么用模板。

### exact（默认）

同一 Gate，两种抽取/库，由 `single_step.template_filter.backend` 选择：`localtemplate`（默认）或 `rdchiral`。不要把两套库混用。将候选反应物与当前产物组成前向反应 `reactants>>product`，atom mapping 后抽取模板；mapping 只是抽取前置步骤，不是额外 Gate。canonicalize 与建库脚本 `step2_2_get_rxn_temp_rdchil_localtemp_mp.py` 相同：`>>` 两侧 `.` 片段排序后精确匹配。命中 `PASS`，抽出 SMARTS 但不在库中、抽到空串、atom mapping / 抽取超时或异常都是 `FAIL`，丢掉候选。

### LocalTemplate

- `backend: localtemplate`
- 模板库：`./data/forward_and_template/WanJRxn_Train_0314_unique_rxn_temp.txt`（每行一条 `reaction_smarts`）
- 抽取：`LocalTemplate.template_extractor`，`use_stereo=True`、`use_symbol=True`、`retro=True`

### RDChiral

- `backend: rdchiral`
- 模板库：`/path/to/rdchiral_res_rxn_temp_count.jsonl`（每行 `{"key": "<SMARTS>", "value": <count>}`）
- 抽取：`rdchiral.template_extractor.extract_from_reaction`，取 `reaction_smarts`
- 评测结果用 `rdchiral.jsonl`；旧 LocalTemplate 结果已改名为 `localtemplate.jsonl` / `forward_localtemplate.jsonl`

### apply（AOT*）

把单步返回的一列候选一起送进 AOT* 入口，出来是过滤并改写后的列表。需要 `backend: rdchiral` 和 WanJ rdchiral jsonl 库。

```yaml
single_step:
  template_filter:
    enabled: true
    backend: rdchiral
    match_mode: apply
    template_path: /path/to/rdchiral_res_rxn_temp_count.jsonl
```

对每个候选：产物 Tversky ≥ 1 → 两边子结构匹配取第一条模板 → `rdchiralRun` 改写反应物 → 套不出再 blurry → 全部失败再做一次 Retrosim。活下来的去重，可按库存/SCScore 排序。前向 Gate 若也打开，仍先跑前向，再把剩下的送进这一列。
