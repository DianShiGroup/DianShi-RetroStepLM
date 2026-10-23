# DianShi-RetroStep

[English](README.md) | 中文

基于 **DianShi-RetroStep-LM-Direct** 的单步逆合成评测与多步合成路线规划。

- **单步评测：** 从 DianShi-RxnStep 和 USPTO-50K 构建 `sl_cot`（逐步思维链）测试集，运行 vLLM 束搜索，报告 Top-k 精确匹配准确率。
- **多步规划：** `multi_route/` 下的三个框架用 LLM planner 搜索到可购买砌块的路线，结合 direct 模型、rdchiral 模板应用和 AOT* SCScore 重排。

## Hugging Face 资源

| 资源 | 链接 | 内容 |
|---|---|---|
| 模型 | [DianShiGroup/DianShi-RetroStep-LM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStep-LM-Direct) | 单步逆合成模型（基于 Qwen3-4B 微调，不使用检索） |
| 数据集 | [DianShiGroup/DianShi-RetroRoute-Dataset](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroRoute-Dataset) | 多步规划运行资源：rdchiral 模板库、eMolecules 砌块库存、SCScore checkpoint，以及 USPTO-190 / Pistachio-100 / PatMol-150 目标集 |

~~~bash
# 单步：scripts/infer_beam_search.sh 的默认 --model-path
hf download DianShiGroup/DianShi-RetroStep-LM-Direct --local-dir models/DianShi-RetroStep-Direct

# 多步：在所选框架目录下运行
cd multi_route/DianShi-RetroRoute-System-Preview
hf download DianShiGroup/DianShi-RetroRoute-Dataset --repo-type dataset --local-dir DianShi-RetroRoute-Dataset
~~~

单步脚本和多步框架也可以直接使用模型 ID `DianShiGroup/DianShi-RetroStep-LM-Direct`，vLLM 会在首次使用时自动下载。

## 快速入口

| 任务 | 入口 | 说明 |
|---|---|---|
| 构建单步评测数据集 | `scripts/build_model_tests.sh` | [构建数据集](#构建数据集) |
| 单步推理与 Top-k 评测 | `scripts/infer_beam_search.sh` | [束搜索推理与评测](#束搜索推理与评测) |
| 多步路线规划 | `multi_route/` 下的某个框架 | [多步路线规划](#多步路线规划) |
| 比较不同搜索预算下的求解率 | `multi_route/tools/solved_by_n.py` | [多步结果与统计](#多步结果与统计) |

## 仓库结构

~~~text
DianShi-RetroStep/
  datasets/
    DianShi-RxnStep-dataset/      Test-IN、Test-OUT、Val-IN、Val-OUT 划分
    USPTO-50K/                    原始 train、validation、test 划分
  multi_route/
    RetroAgent/                   带 AND-OR 图记忆的工具调用式 planner
    Retro-R1-modified/            CALL / SELECT 规划环境
    DianShi-RetroRoute-System-Preview/
                                  AND-OR 树调度器与评测 CLI
    tools/solved_by_n.py          三种结果格式通用的求解率统计
    README.md                     多步框架总览
  scripts/
    build_model_tests.sh          构建 sl_cot 评测数据集
    infer_beam_search.sh          束搜索推理 + 评测
  src/dianshi_retrostep/
    data/                         数据集构建、atom-wise SMILES 转换
    inference/                    多 GPU vLLM 束搜索
    evaluation/                   Top-k 精确匹配评测
    prompts/prompt_sft_sl_cot.txt sl_cot 提示词模板
  pyproject.toml
  requirements.txt
~~~

## 单步评测

本节命令均在仓库根目录运行。`scripts/` 下的入口脚本会自动切换到仓库根目录，并把 `src` 加入 `PYTHONPATH`。

### 安装

需要 Python 3.10 及以上。

~~~bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .   # 可选：不经 shell 脚本直接运行模块
~~~

该环境只覆盖单步流程；多步框架有各自的依赖（见下文）。

### 构建数据集

源数据和提示词模板已包含在仓库中：

- `datasets/DianShi-RxnStep-dataset`：Test-IN、Test-OUT、Val-IN、Val-OUT。
- `datasets/USPTO-50K`：原始 train、validation、test 划分；构建时使用 validation 和 test。

构建在 CPU 上运行：

~~~bash
scripts/build_model_tests.sh                                   # 全部数据集和划分
scripts/build_model_tests.sh --dry-run                         # 只打印构建计划
scripts/build_model_tests.sh --datasets test_out --splits test
scripts/build_model_tests.sh --datasets test_in,test_out,uspto_50k --splits test,val
scripts/build_model_tests.sh --force                           # 重建已有输出
~~~

可通过 `PYTHON_BIN` 指定其他 Python 解释器。产物 SMILES 会被转换成 atom-wise 形式（token 之间以空格分隔），与模型训练格式一致。输出写在 `model_tests/` 下：

~~~text
model_tests/
  test_in/sl_cot/
    testset_sl_cot/                   JSONL
    testset_sl_cot_new_cached/        Hugging Face 缓存数据集
    validationset_sl_cot/
    validationset_sl_cot_new_cached/
  test_out/sl_cot/...
  uspto_50k/sl_cot/...
~~~

### 束搜索推理与评测

`scripts/infer_beam_search.sh` 在 `sl_cot` 缓存数据集上运行 vLLM 束搜索，然后打印 Top-k 精确匹配准确率。可以直接传 Hugging Face 模型 ID，vLLM 会在首次使用时下载权重：

~~~bash
scripts/infer_beam_search.sh \
  --model-path DianShiGroup/DianShi-RetroStep-LM-Direct \
  --data-path model_tests/test_out/sl_cot/testset_sl_cot_new_cached \
  --num-gpus 8 \
  --batch-size 512
~~~

`--model-path` 也可以是本地目录，默认值为 `models/DianShi-RetroStep-Direct`。评测其他划分时，把对应缓存目录传给 `--data-path`，例如 `model_tests/test_in/sl_cot/testset_sl_cot_new_cached` 或 `model_tests/uspto_50k/sl_cot/testset_sl_cot_new_cached`。

| 选项 | 默认值 |
|---|---|
| `--data-path` | `model_tests/test_out/sl_cot/testset_sl_cot_new_cached` |
| `--output-dir` | `results` |
| `--num-gpus` / `--batch-size` | `1` / `128` |
| `--beam-width` / `--max-tokens` | `10` / `1024` |
| Top-k 指标 | 1、3、5、10 |
| Thinking | 在 chat template 中关闭（用 `--enable-thinking` 开启） |

每个选项也可以用对应的大写环境变量设置，例如 `MODEL_PATH=... NUM_GPUS=4 scripts/infer_beam_search.sh`。已有结果文件不会被覆盖；需要替换时加 `--overwrite`。

该脚本串联两个模块，也可以分别运行（先 `pip install -e .`，或在命令前加 `PYTHONPATH=src`）：

- `dianshi_retrostep.inference.vllm_beamsearch_infer`：只保留 system 和 user 消息（避免数据集中的标准答案泄漏进输入），把每个批次分配给单 GPU 的 Ray worker，按序执行束搜索，每个样本写一行 JSON，包含排序后的候选及累积对数概率。
- `dianshi_retrostep.evaluation.eval_retro_beam_topk`：从每条候选中提取答案，用 RDKit 规范化标准答案和预测反应物，忽略反应物顺序，报告 Top-k 精确匹配准确率。支持 `<answer>` 标签、普通或 atom-wise SMILES、Markdown 包裹的答案，以及 `</think>` 之后的文本。

~~~bash
python3 -m dianshi_retrostep.inference.vllm_beamsearch_infer \
  --model-path DianShiGroup/DianShi-RetroStep-LM-Direct \
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

## 多步路线规划

每个框架都有自己的依赖、配置和 `.env`。[多步框架总览](multi_route/README.md)链接到各框架的完整说明。

### 框架

三个框架可以独立运行。随附的 direct 配置使用 [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) 作为 planner，[DianShi-RetroStep-LM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStep-LM-Direct) 作为单步模型。

| 框架 | 搜索接口 | 安装与配置 |
|---|---|---|
| RetroAgent | Agent 调用由 HTTP 服务和 AND-OR `GraphMemory` 支撑的工具 | [RetroAgent 说明](multi_route/RetroAgent/README.md) |
| Retro-R1-modified | 进程内 agent 用 `CALL` 请求候选、用 `SELECT` 应用其中一个 | [Retro-R1 说明](multi_route/Retro-R1-modified/README.md) |
| DianShi-RetroRoute-System-Preview | 调度器向 planner 提供合法的扩展选项并管理 AND-OR 树 | [Preview 说明](multi_route/DianShi-RetroRoute-System-Preview/README.md) |

### 共同的规划流程

1. Planner 选择要展开的分子或路线前沿。
2. DianShi direct 模型通过 OpenAI 兼容的 `/v1/chat/completions` 接口提出单步断键。随附的 direct 配置不使用检索。
3. AOT* 门控应用 rdchiral 反应模板（必要时做一轮 blurry 匹配），再按 SCScore 合成复杂度重排候选，并对库存之外的反应物施加 `unsolved_penalty`。只有通过的候选进入搜索。
4. 框架用规范化 SMILES 在标准化的 eMolecules SQLite 库存中检查反应物是否可购。一个反应的所有反应物都已解决时该反应即解决；找到完整分解时分子即解决。
5. 目标被解决或达到框架的搜索上限时停止。调用预算、轮数上限、深度上限和停止规则见各框架说明。

模板门控的约定和参数见 [VALIDATE.md](multi_route/DianShi-RetroRoute-System-Preview/docs/VALIDATE.md)。

### 所需模型与运行资源

| 资源 | 准备方式 |
|---|---|
| DianShi direct 单步权重 | 在所选框架的 `.env` 中把 `SINGLE_STEP_MODEL_PATH` 设为 `DianShiGroup/DianShi-RetroStep-LM-Direct` 或本地副本。 |
| Planner 权重 | 把 `PLANNER_MODEL_PATH` 设为 `Qwen/Qwen3.5-397B-A17B-FP8` 或本地副本。vLLM 可在首次使用时下载。 |
| rdchiral 模板库 | 把 `DIANSHI_RDCHIRAL_TEMPLATE_JSONL` 设为 `DianShi-RetroRoute-Dataset/rxn_template/rdchiral_res_rxn_temp_count.jsonl`。 |
| 可购砌块库存 | 把 `STARTING_MOLECULES` 设为标准化的 eMolecules SQLite 文件；运行 Retro-R1 时需显式传 `--starting_molecules`。 |
| 目标分子 | 每行一个目标 SMILES。评测目标路径见下文。 |
| AOT* SCScore 权重 | 2048-bit checkpoint 已包含在两个 fork 的 `inference/gate/weights/` 下。Preview 需要在 `runtime/single_step/aot_scscore/models/full_reaxys_model_2048bool/` 下放一份副本，下面的示例会完成这一步。RetroAgent 还需要其说明中提到的 tool-server SCScore 资源。 |

模板库、库存数据库、SCScore checkpoint 和目标文件已发布为 Hugging Face 数据集 [DianShiGroup/DianShi-RetroRoute-Dataset](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroRoute-Dataset)。把它下载到所选框架目录下的 `DianShi-RetroRoute-Dataset/`，按所选框架的说明放置资源，然后把其 `.env.example` 复制为 `.env`，配置模型路径、接口 URL 和 API key。

### 目标集

表中路径相对于所选框架目录。

| 评测集 | 目标数 | 目标文件 |
|---|---:|---|
| USPTO-190 | 190 | `DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt` |
| Pistachio-100 | 100 | `DianShi-RetroRoute-Dataset/test_dataset/pistachio_hard_tree.txt` |
| PatMol-150 | 150 | `DianShi-RetroRoute-Dataset/test_dataset/patmol_150_targets.txt` |

RetroAgent 用 `--targets` 选择目标；Retro-R1 的 runner 支持 `--targets`，`cli.py` 支持 `--data_file`。Preview 为每个目标集提供单独的评测 YAML。

### 快速开始：Preview

在 Python 3.11 及以上环境中，从仓库根目录运行：

~~~bash
cd multi_route/DianShi-RetroRoute-System-Preview
python3 -m pip install -r requirements.txt

# 把运行资源下载到该框架目录。
hf download DianShiGroup/DianShi-RetroRoute-Dataset --repo-type dataset --local-dir DianShi-RetroRoute-Dataset

# 把仓库中已有的 2048-bit SCScore checkpoint 放到 Preview 期望的位置。
mkdir -p runtime/single_step/aot_scscore/models/full_reaxys_model_2048bool
cp ../RetroAgent/inference/gate/weights/model.ckpt-10654.as_numpy.json.gz \
  runtime/single_step/aot_scscore/models/full_reaxys_model_2048bool/

cp .env.example .env
# 编辑 .env：设置 SINGLE_STEP_MODEL_PATH、planner/单步模型的 URL、key 和资源路径。
~~~

在 GPU 机器上、Preview 目录下，用两个终端分别部署两个模型。部署脚本使用 vLLM；组件说明要求 `vllm>=0.18`，已在 `0.26` 上验证。该部署环境与根目录的单步束搜索环境版本要求不同。按实际部署分配 GPU；默认配置为 planner TP=8、单步模型 TP=1。

~~~bash
# 终端 1：planner 服务
bash scripts/serve_planner.sh
~~~

~~~bash
# 终端 2：单步模型服务
bash scripts/serve_single_step.sh
~~~

两个接口可用后，在 Preview 目录下运行评测。评测进程可以在 CPU 上运行，模型在 GPU 机器上运行：

~~~bash
python3 cli.py --config configs/bench/qwen_direct_uspto190.yaml
# 或评测 Pistachio 目标：
python3 cli.py --config configs/bench/qwen_direct_pistachio100.yaml
~~~

随附的评测 YAML 开启了模板门控，并设置 `max_steps: 500`、`target_complete_paths: 1`、`concurrency: 2`。评测请使用这些 YAML；Preview 默认的 `config.yaml` 没有定义批量 `run` 段，且门控是关闭的。

### 运行两个 fork

先按对应框架说明安装依赖、准备资源和 `.env`，并启动 planner 与单步模型服务。然后在仓库根目录运行以下任一命令：

RetroAgent 的 tool server 会从固定的相对路径加载两个资源，首次运行前请先放好：

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

RetroAgent 的 runner 会启动配置的 tool server、运行评测，然后关闭该服务。Retro-R1 的 runner 调用进程内规划环境。两者在重跑时都会复用已有的目标结果；如需全新评测，请换新的输出/日志目录或移走已有结果。

### 多步结果与统计

Preview 在 `results/bench/<dataset>/` 下为每个目标写一行，包含是否成功、终止原因、模型调用次数、耗时、token 用量和已求解的路线树。随附的评测 YAML 设置了 `resume: false`，不会覆盖已有输出。设置 `resume: true` 可续跑中断的评测；已有的行（包括失败目标）会被跳过。

Retro-R1 在配置的输出目录下写 `result.jsonl`。RetroAgent 在日志目录下写每个目标的结果和 `summary.json`。完整的输出说明见各框架文档。

在仓库根目录，用共享统计脚本计算 N = 50、100、200、300、400、500 时的求解率：

~~~bash
python3 multi_route/tools/solved_by_n.py \
  multi_route/DianShi-RetroRoute-System-Preview/results/bench/uspto_190/qwen_direct_rdchiral_aot.jsonl

# 同一脚本也接受 Retro-R1 的 JSONL 或 RetroAgent 的 summary.json。
python3 multi_route/tools/solved_by_n.py /path/to/result.jsonl
python3 multi_route/tools/solved_by_n.py /path/to/summary.json
~~~

脚本会自动识别结果格式，并报告可能的基础设施故障。求解率基于所提供的记录计算；比较不同运行前，请确认目标集完整并处理已报告的故障。对于 RetroAgent 的 `summary.json`，脚本使用第一个实验。

## 许可证

本仓库以 [Apache-2.0](LICENSE) 发布，但 `multi_route/` 下的两个上游 fork 沿用各自的条款：

| 路径 | 许可证 |
|---|---|
| 其余全部内容，含 `src/` 与 `multi_route/DianShi-RetroRoute-System-Preview/` | Apache-2.0（[`LICENSE`](LICENSE)） |
| `multi_route/RetroAgent/` | MIT（[`LICENSE`](multi_route/RetroAgent/LICENSE)、[`NOTICE`](multi_route/RetroAgent/NOTICE)） |
| `multi_route/Retro-R1-modified/` | MIT（[`LICENSE`](multi_route/Retro-R1-modified/LICENSE)） |

内置的第三方依赖保留各自的许可证，见 `multi_route/Retro-R1-modified/packages/*/LICENSE`
及各组件 README。数据集与模型权重单独发布，适用其 Hugging Face 页面上声明的条款；
其中 eMolecules 砌块数据另有自己的使用条款，再分发前请先确认。

## 致谢

多步 fork 基于 [RetroAgent](https://huggingface.co/SXKDZ/RetroAgent) 和 [Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1)。模板应用使用 [rdchiral](https://github.com/coleygroup/rdchiral)；重排改编自 [AOT*](https://github.com/ShawnKS/AOTstar)，使用 [SCScore](https://github.com/connorcoley/scscore) 打分器及其权重。砌块数据来自 eMolecules 以及各框架说明中记录的来源。引用和第三方条款见各组件的 README、许可证和声明文件。
