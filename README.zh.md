# DianShi-RetroStep

[English](README.md) | 中文

基于 **DianShi-RetroStepLM-Direct** 的单步逆合成评测与多步合成路线规划。

- 🔬 **单步评测：** 从 DianShi-PatentRxnStep-Dataset 和 USPTO-50K 构建 intermediate-label 测试集，运行 vLLM 束搜索，报告 Top-k 精确匹配准确率。提示词要求模型依次给出合成子（synthon）、离去基团（leaving group）和最终反应物，即模型训练所用的 intermediate-label 格式。
- ⚗️ **多步规划：** `multi_route/` 下的三个框架用 LLM planner 搜索到可购买原料的路线，结合 RetroStepLM-Direct、rdchiral 模板应用和 AOT* SCScore 重排。

## 📢 最新动态

- **2026-10-28** — [DianShi-RetroStepLM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStepLM-Direct) 模型权重已开源，同时开源[单步推理与评测代码](#-单步评测)。
- **2026-10-28** — [三个多步规划框架的推理代码](#多步路线规划)已开源，运行资源发布为 [DianShi-RetroPlanSystem-Assets](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroPlanSystem-Assets)。

## 🤗 Hugging Face 资源

| 资源 | 链接 | 内容 |
|---|---|---|
| 模型 | [DianShiGroup/DianShi-RetroStepLM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStepLM-Direct) | 单步逆合成模型（基于 Qwen3-4B-Base 微调，不使用检索） |
| 数据集 | [DianShiGroup/DianShi-RetroPlanSystem-Assets](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroPlanSystem-Assets) | 多步规划运行资源：rdchiral 模板库、eMolecules 原料库，以及 USPTO-190 / Pistachio Hard / PatMol-150 目标集 |

~~~bash
# 单步：scripts/infer_beam_search.sh 的默认 --model-path
hf download DianShiGroup/DianShi-RetroStepLM-Direct --local-dir models/DianShi-RetroStepLM-Direct

# 多步：在所选框架目录下运行
cd multi_route/DianShi-RetroPlanSystem-v0
hf download DianShiGroup/DianShi-RetroPlanSystem-Assets --repo-type dataset --local-dir DianShi-RetroPlanSystem-Assets
~~~

单步脚本和多步框架也可以直接使用模型 ID `DianShiGroup/DianShi-RetroStepLM-Direct`，vLLM 会在首次使用时自动下载。

## 🚀 快速入口

| 任务 | 入口 | 说明 |
|---|---|---|
| 构建单步评测数据集 | `scripts/build_model_tests.sh` | [构建数据集](#构建数据集) |
| 单步推理与 Top-k 评测 | `scripts/infer_beam_search.sh` | [束搜索推理与评测](#束搜索推理与评测) |
| 多步路线规划 | `multi_route/` 下的某个框架 | [多步路线规划](#多步路线规划) |
| 比较不同搜索预算下的求解率 | `multi_route/tools/solved_by_n.py` | [多步结果与统计](#多步结果与统计) |

## 📂 仓库结构

~~~text
DianShi-RetroStep/
  datasets/
    DianShi-PatentRxnStep-Dataset/
                                  Test-2001-2024、Test-2025、Val-2001-2024、Val-2025
    USPTO-50K/                    原始 train、validation、test 划分
  multi_route/
    RetroAgent/                   带 AND-OR 图记忆的工具调用式 planner
    Retro-R1-modified/            CALL / SELECT 规划环境
    DianShi-RetroPlanSystem-v0/   AND-OR 树调度器与评测 CLI
    tools/solved_by_n.py          三种结果格式通用的求解率统计
    README.md                     多步框架总览
  scripts/
    build_model_tests.sh          构建 intermediate-label 评测数据集
    infer_beam_search.sh          束搜索推理 + 评测
  src/dianshi_retrostep/
    data/                         数据集构建、atom-wise SMILES 转换
    inference/                    多 GPU vLLM 束搜索
    evaluation/                   Top-k 精确匹配评测
    prompts/prompt_sft_intermediate_label.txt
                                  intermediate-label 提示词模板
  pyproject.toml
  requirements.txt
~~~

## 🔬 单步评测

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

- `datasets/DianShi-PatentRxnStep-Dataset`：四个评测分区。Test-2001-2024 与 Val-2001-2024（各 3,798 条反应）覆盖 2001–2024 年记录的反应；Test-2025 与 Val-2025（各 1,886 条反应）为时间外推集，全部为 2025 年记录的反应。
- `datasets/USPTO-50K`：原始 train、validation、test 划分；构建时使用 validation 和 test。

构建在 CPU 上运行。`--datasets` 可取 `test_2001_2024`、`test_2025` 或 `uspto_50k`，每个都会构建其 test 与 validation 分区：

~~~bash
scripts/build_model_tests.sh                                   # 全部数据集和划分
scripts/build_model_tests.sh --dry-run                         # 只打印构建计划
scripts/build_model_tests.sh --datasets test_2025 --splits test
scripts/build_model_tests.sh --datasets test_2001_2024,test_2025,uspto_50k --splits test,val
scripts/build_model_tests.sh --force                           # 重建已有输出
~~~

可通过 `PYTHON_BIN` 指定其他 Python 解释器。产物 SMILES 会被转换成 atom-wise 形式（token 之间以空格分隔），与模型训练格式一致。输出写在 `model_tests/` 下：

~~~text
model_tests/
  test_2001_2024/intermediate_label/
    testset_intermediate_label/                   JSONL
    testset_intermediate_label_new_cached/        Hugging Face 缓存数据集
    validationset_intermediate_label/
    validationset_intermediate_label_new_cached/
  test_2025/intermediate_label/...
  uspto_50k/intermediate_label/...
~~~

### 束搜索推理与评测

`scripts/infer_beam_search.sh` 在 intermediate-label 缓存数据集上运行 vLLM 束搜索，然后打印 Top-k 精确匹配准确率。可以直接传 Hugging Face 模型 ID，vLLM 会在首次使用时下载权重：

~~~bash
scripts/infer_beam_search.sh \
  --model-path DianShiGroup/DianShi-RetroStepLM-Direct \
  --data-path model_tests/test_2025/intermediate_label/testset_intermediate_label_new_cached \
  --num-gpus 8 \
  --batch-size 512
~~~

`--model-path` 也可以是本地目录，默认值为 `models/DianShi-RetroStepLM-Direct`。评测其他划分时，把对应缓存目录传给 `--data-path`，例如 `model_tests/test_2001_2024/intermediate_label/testset_intermediate_label_new_cached` 或 `model_tests/uspto_50k/intermediate_label/testset_intermediate_label_new_cached`。不传 `--output-path` 时，结果写到 `results/<模型>_<数据集>_<划分>_beam<宽度>.jsonl`，每个测试集各有一个文件。

| 选项 | 默认值 |
|---|---|
| `--data-path` | `model_tests/test_2025/intermediate_label/testset_intermediate_label_new_cached` |
| `--output-dir` | `results` |
| `--num-gpus` / `--batch-size` | `1` / `128` |
| `--beam-width` / `--max-tokens` | `10` / `1024` |
| Top-k 指标 | 1、3、5、10 |
| Thinking | 开启，与论文实验一致（用 `--no-thinking` 关闭） |

每个选项也可以用对应的大写环境变量设置，例如 `MODEL_PATH=... NUM_GPUS=4 scripts/infer_beam_search.sh`。已有结果文件不会被覆盖；需要替换时加 `--overwrite`。

该脚本串联两个模块，也可以分别运行（先 `pip install -e .`，或在命令前加 `PYTHONPATH=src`）：

- `dianshi_retrostep.inference.vllm_beamsearch_infer`：只保留 system 和 user 消息（避免数据集中的标准答案泄漏进输入），把每个批次分配给单 GPU 的 Ray worker，按序执行束搜索，每个样本写一行 JSON，包含排序后的候选及累积对数概率。
- `dianshi_retrostep.evaluation.eval_retro_beam_topk`：从每条候选中提取答案，用 RDKit 规范化标准答案和预测反应物，忽略反应物顺序，报告 Top-k 精确匹配准确率。支持 `<answer>` 标签、普通或 atom-wise SMILES、Markdown 包裹的答案，以及 `</think>` 之后的文本。

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

<a id="多步路线规划"></a>

## ⚗️ 多步路线规划

每个框架都有自己的依赖、配置和 `.env`。[多步框架总览](multi_route/README.md)链接到各框架的完整说明。

### 框架

三个框架可以独立运行。随附的 direct 配置使用 [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) 作为 planner，[DianShi-RetroStepLM-Direct](https://huggingface.co/DianShiGroup/DianShi-RetroStepLM-Direct) 作为单步模型。

| 框架 | 搜索接口 | 安装与配置 |
|---|---|---|
| RetroAgent | Agent 调用由 HTTP 服务和 AND-OR `GraphMemory` 支撑的工具 | [RetroAgent 说明](multi_route/RetroAgent/README.md) |
| Retro-R1-modified | 进程内 agent 用 `CALL` 请求候选、用 `SELECT` 应用其中一个 | [Retro-R1 说明](multi_route/Retro-R1-modified/README.md) |
| DianShi-RetroPlanSystem-v0 | 调度器向 planner 提供合法的扩展选项并管理 AND-OR 树 | [DianShi-RetroPlanSystem-v0 说明](multi_route/DianShi-RetroPlanSystem-v0/README.md) |

### 共同的规划流程

1. Planner 选择要展开的分子或路线前沿。
2. RetroStepLM-Direct 通过 OpenAI 兼容的 `/v1/chat/completions` 接口提出单步断键。随附的 direct 配置不使用检索。
3. AOT* 门控应用 rdchiral 反应模板（必要时做一轮 blurry 匹配），再按 SCScore 合成复杂度重排候选，并对原料库之外的反应物施加 `unsolved_penalty`。只有通过的候选进入搜索。论文中称这一步为 AOT*-style template-based reaction validation。
4. 框架用规范化 SMILES 在标准化的 eMolecules SQLite 原料库中检查反应物是否可购。一个反应的所有反应物都已解决时该反应即解决；找到完整分解时分子即解决。
5. 目标被解决或达到框架的搜索上限时停止。调用预算、轮数上限、深度上限和停止规则见各框架说明。

模板门控的约定和参数见 [VALIDATE.md](multi_route/DianShi-RetroPlanSystem-v0/docs/VALIDATE.md)。

### 所需模型与运行资源

| 资源 | 准备方式 |
|---|---|
| RetroStepLM-Direct 单步权重 | 在所选框架的 `.env` 中把 `SINGLE_STEP_MODEL_PATH` 设为 `DianShiGroup/DianShi-RetroStepLM-Direct` 或本地副本。 |
| Planner 权重 | 把 `PLANNER_MODEL_PATH` 设为 `Qwen/Qwen3.5-397B-A17B-FP8` 或本地副本。vLLM 可在首次使用时下载。 |
| rdchiral 模板库 | 把 `DIANSHI_RDCHIRAL_TEMPLATE_JSONL` 设为 `DianShi-RetroPlanSystem-Assets/rxn_template/rdchiral_res_rxn_temp_count.jsonl`。 |
| 可购买原料库 | 把 `STARTING_MOLECULES` 设为标准化的 eMolecules SQLite 文件（23,081,629 个可购买原料）；运行 Retro-R1 时需显式传 `--starting_molecules`。原料库来自 Retro* 的 eMolecules 可购买原料列表，用 RDKit 2025.9.5 标准化后转换为 SQLite。 |
| 目标分子 | 每行一个目标 SMILES。评测目标路径见下文。 |
| AOT* SCScore 权重 | 2048-bit gate checkpoint 已随每个框架的 gate 代码一起提供；1024-bit tool-server checkpoint 已 vendored 在 `RetroAgent/datasets/scscore/` 下，不需要额外下载或复制。 |

模板库、原料库和目标文件已发布为 Hugging Face 数据集 [DianShiGroup/DianShi-RetroPlanSystem-Assets](https://huggingface.co/datasets/DianShiGroup/DianShi-RetroPlanSystem-Assets)。把它下载到所选框架目录下的 `DianShi-RetroPlanSystem-Assets/`，按所选框架的说明放置资源，然后把其 `.env.example` 复制为 `.env`，配置模型路径、接口 URL 和 API key。

### 目标集

表中路径相对于所选框架目录。

| 评测集 | 目标数 | 目标文件 |
|---|---:|---|
| USPTO-190 | 190 | `DianShi-RetroPlanSystem-Assets/test_dataset/uspto_190_targets_smiles.txt` |
| Pistachio Hard | 100 | `DianShi-RetroPlanSystem-Assets/test_dataset/pistachio_hard_tree.txt` |
| PatMol-150 | 150 | `DianShi-RetroPlanSystem-Assets/test_dataset/patmol_150_targets.txt` |

RetroAgent 用 `--targets` 选择目标；Retro-R1 的 runner 支持 `--targets`，`cli.py` 支持 `--data_file`。DianShi-RetroPlanSystem-v0 为每个目标集提供单独的评测 YAML。

### 快速开始：DianShi-RetroPlanSystem-v0

在 Python 3.11 及以上环境中，从仓库根目录运行：

~~~bash
cd multi_route/DianShi-RetroPlanSystem-v0
python3 -m pip install -r requirements.txt

# 把运行资源下载到该框架目录。
hf download DianShiGroup/DianShi-RetroPlanSystem-Assets --repo-type dataset --local-dir DianShi-RetroPlanSystem-Assets

cp .env.example .env
# 编辑 .env：设置 SINGLE_STEP_MODEL_PATH、planner/单步模型的 URL、key 和资源路径。
~~~

在 GPU 机器上、DianShi-RetroPlanSystem-v0 目录下，用两个终端分别部署两个模型。部署脚本使用 vLLM；组件说明要求 `vllm>=0.18`，已在 `0.26` 上验证。该部署环境与根目录的单步束搜索环境版本要求不同。按实际部署分配 GPU；默认配置为 planner TP=8、单步模型 TP=1。

~~~bash
# 终端 1：planner 服务
bash scripts/serve_planner.sh
~~~

~~~bash
# 终端 2：单步模型服务
bash scripts/serve_single_step.sh
~~~

两个接口可用后，在 DianShi-RetroPlanSystem-v0 目录下运行评测。评测进程可以在 CPU 上运行，模型在 GPU 机器上运行：

~~~bash
python3 cli.py --config configs/bench/qwen_direct_uspto190.yaml
# 或评测 Pistachio Hard 目标：
python3 cli.py --config configs/bench/qwen_direct_pistachio_hard.yaml
~~~

随附的评测 YAML 开启了模板门控，并设置 `max_steps: 500`、`target_complete_paths: 1`、`concurrency: 2`。评测请使用这些 YAML；该框架默认的 `config.yaml` 没有定义批量 `run` 段，且门控是关闭的。

### 运行两个 fork

先按对应框架说明安装依赖、准备资源和 `.env`，并启动 planner 与单步模型服务。然后在仓库根目录运行以下任一命令：

RetroAgent 的 tool server 会从固定的相对路径加载原料库，首次运行前请先放好：

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

RetroAgent 的 runner 会启动配置的 tool server、运行评测，然后关闭该服务。Retro-R1 的 runner 调用进程内规划环境。两者在重跑时都会复用已有的目标结果；如需全新评测，请换新的输出/日志目录或移走已有结果。

### 多步结果与统计

DianShi-RetroPlanSystem-v0 在 `results/bench/<dataset>/` 下为每个目标写一行，包含是否成功、终止原因、模型调用次数、耗时、token 用量和已求解的路线树。随附的评测 YAML 设置了 `resume: false`，不会覆盖已有输出。设置 `resume: true` 可续跑中断的评测；已有的行（包括失败目标）会被跳过。

Retro-R1 在配置的输出目录下写 `result.jsonl`。RetroAgent 在日志目录下写每个目标的结果和 `summary.json`。完整的输出说明见各框架文档。

在仓库根目录，用共享统计脚本计算 N = 50、100、200、300、400、500 时的求解率：

~~~bash
python3 multi_route/tools/solved_by_n.py \
  multi_route/DianShi-RetroPlanSystem-v0/results/bench/uspto_190/qwen_direct_rdchiral_aot.jsonl

# 同一脚本也接受 Retro-R1 的 JSONL 或 RetroAgent 的 summary.json。
python3 multi_route/tools/solved_by_n.py /path/to/result.jsonl
python3 multi_route/tools/solved_by_n.py /path/to/summary.json
~~~

脚本会自动识别结果格式，并报告可能的基础设施故障。求解率基于所提供的记录计算；比较不同运行前，请确认目标集完整并处理已报告的故障。对于 RetroAgent 的 `summary.json`，脚本使用第一个实验。

## 📜 许可证

整个仓库（包括 `multi_route/` 下的两个 fork）以 [Apache-2.0](LICENSE) 发布。

`RetroAgent` 和 `Retro-R1-modified` 基于以 MIT 许可证发布的上游项目。按 MIT 的要求，
上游的版权声明保留在 [`multi_route/RetroAgent/LICENSE`](multi_route/RetroAgent/LICENSE)
（及 [`NOTICE`](multi_route/RetroAgent/NOTICE)）和
[`multi_route/Retro-R1-modified/LICENSE`](multi_route/Retro-R1-modified/LICENSE) 中。
`multi_route/Retro-R1-modified/packages/` 下内置的第三方包保留各自的许可证文件。

模型权重以 Apache-2.0 发布，与其微调所基于的 Qwen3-4B-Base 一致。运行资源数据集中含有
第三方数据（eMolecules 原料库，以及 USPTO-190 和 Pistachio Hard 评测集），这些数据沿用
原有条款，再分发前请先查看其 Hugging Face 页面。

## 📚 致谢

多步 fork 基于 [RetroAgent](https://huggingface.co/SXKDZ/RetroAgent) 和 [Retro-R1](https://github.com/GenSI-THUAIR/Retro-R1)。模板应用使用 [rdchiral](https://github.com/coleygroup/rdchiral)；重排改编自 [AOT*](https://github.com/ShawnKS/AOTstar)，使用 [SCScore](https://github.com/connorcoley/scscore) 打分器及其权重。原料数据来自 eMolecules 以及各框架说明中记录的来源。引用和第三方条款见各组件的 README、许可证和声明文件。
