# semantic

这是一个从现有工程中**纯复制**整理出来的独立项目目录（未修改原始工程文件）。

目标：
1. 在 `esa` 上做 Good-Turing / General-GT / Hybrid 系列估计器改良与对比。
2. 复用已经跑好的模型数据（`Mistral-7B-Instruct-v0.3`）。
3. 复用 `SIGNET` 做 semantic entropy / graph OOD 学习相关实验。

## 0. 复制来源（原路径）
- `esa` 源码：`/share/home/luenqiao/phx/esa`
- 已跑数据：`/share/home/luenqiao/phx/data/no_preprompt/Mistral-7B-Instruct-v0.3`
- 图模型代码：`/share/home/luenqiao/phx/SIGNET`

说明：本目录内容通过 `rsync/cp` 复制，不改原项目。

## 1. 目录结构

```text
semantic/
├── README.md
├── requirements.txt
├── esa/
│   ├── src/
│   │   ├── entropy/
│   │   │   ├── coverage.py
│   │   │   └── entropy.py
│   │   ├── experiments/
│   │   │   ├── collect_alphabet.py
│   │   │   ├── collect_uncertainty.py
│   │   │   ├── compare_hybrid_ggt_ueigv.py
│   │   │   └── ...
│   │   └── utils/data_utils.py
│   └── outputs/hybrid_ggt_eval/
├── data/
│   └── no_preprompt/
│       └── Mistral-7B-Instruct-v0.3/
│           ├── hotpot_qa_final_results.json
│           ├── hotpot_qa_final_labeled_results.json
│           └── uncertainty.csv
└── SIGNET/
    ├── run_semantic_entropy_signet.py
    ├── run_semantic_entropy_signet_nli.py
    ├── main.py
    ├── models.py
    └── outputs/semantic_entropy_nli_*/
```

## 2. 安装依赖（requirements）
在 `semantic/` 根目录执行。

### 2.1 快速安装（已有匹配的 torch 环境）

```bash
pip install -r requirements.txt
```

说明：
- `requirements.txt` 是项目级完整清单，覆盖 `esa + SIGNET + 当前实验脚本`。

### 2.2 推荐安装顺序（避免 PyG 轮子冲突）

1. 先安装与你机器匹配的 `torch`（CPU 或 CUDA）  
2. 再安装 `torch-geometric` 及其扩展  
3. 最后执行 `pip install -r requirements.txt`

示例（CPU）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install -r requirements.txt
```

## 3. ESA 文件说明（估计器与数据流程）
### 3.1 核心估计器
- `esa/src/entropy/coverage.py`
  - `CoverageEstimator`
  - alphabet size 估计：`gt`, `general-gt(ggt)`, `u-eigv`, `hybrid`
- `esa/src/entropy/entropy.py`
  - `EntropyEstimator`
  - entropy 估计：`plugin`, `chao-shen`, `cs-hybrid`, `chao-shen-ggt`

### 3.2 数据收集脚本
- `esa/src/experiments/collect_alphabet.py`
- `esa/src/experiments/collect_uncertainty.py`
- `esa/src/experiments/collect_data/*`（模型调用、采样、judge、批处理）
- `esa/src/experiments/semantic_labeling.py`（语义簇标注）

### 3.3 对比脚本
- `esa/src/experiments/compare_hybrid_ggt_ueigv.py`
  - 输入：`hotpot_qa_final_labeled_results.json`
  - 输出：`oracle(k@100)` vs 多种估计器的 MAE/RMSE
  - 结果目录：`esa/outputs/hybrid_ggt_eval/`

## 4. 变体解释（重点）
以下变体在 `compare_hybrid_ggt_ueigv.py` 中用于对比：

1. `hybrid_ggt`
- 定义：`max(ggt, ueigv)`

2. `miller`
- 定义：`hybrid_ggt + (k_obs - 1)/(2n)`
- 作用：小样本向上偏置修正（启发式）

3. `len_boost`
- 定义：`hybrid_ggt * (1 + len_alpha * z_len)`
- 其中：
  - `len_alpha` 是长度增强系数
  - `z_len` 是当前实验内按题目标准化后的平均回答长度 z-score
- 作用：让回答长度作为无监督辅助信号

4. `ceil_hybrid_ggt`
- 定义：`ceil(hybrid_ggt)`
- 作用：将连续估计转为整数簇数输出

## 5. 已跑数据结构（重点）
路径：`data/no_preprompt/Mistral-7B-Instruct-v0.3/`

### 5.1 `hotpot_qa_final_results.json`
每个 `id` 典型字段：
- `query`
- `true_ans`
- `context`
- `answerable`
- `responses`（通常100条）
- `log_probs`
- `single_response`

### 5.2 `hotpot_qa_final_labeled_results.json`
在有标注样本中，常见结构：
- `cluster_ids["nli-batch"]["100"]`
- `cluster_ids["nli-batch"]["entailment_prob_matrix"]`

注意：并非所有 `id` 都有 `cluster_ids`。

### 5.3 `uncertainty.csv`
当前文件列：
- `dataset, id, n`
- `plugin, cs, cs-hybrid`
- `NumSets, gt, ueigv, hybrid-alphabet`
- `snne, kle, predictive, surprise`
- `oracle, best-guess, judge-llm-score`

## 6. SIGNET 说明（当前状态）
路径：`SIGNET/`

- 关键脚本：
  - `run_semantic_entropy_signet.py`
  - `run_semantic_entropy_signet_nli.py`
  - `main.py`, `models.py`
- 已有输出：
  - `SIGNET/outputs/semantic_entropy_nli_partial/`
  - `SIGNET/outputs/semantic_entropy_nli_edge_only/`

状态说明：
- `SIGNET` 这部分在当前项目里仍在调试中，不作为“已完全稳定”的主流程。
- README 提供的是调用入口和文件定位，最终参数需按你的调试版本为准。

## 7. 运行命令（相对路径，适配 git clone）
以下命令都从项目根目录 `semantic/` 执行。

### 7.1 ESA 对比实验
```bash
python esa/src/experiments/compare_hybrid_ggt_ueigv.py \
  --json_path data/no_preprompt/Mistral-7B-Instruct-v0.3/hotpot_qa_final_labeled_results.json \
  --n 10 \
  --oracle_n 100 \
  --output_dir esa/outputs/hybrid_ggt_eval
```

### 7.2 SIGNET（CSV）
```bash
python SIGNET/run_semantic_entropy_signet.py \
  --csv_path data/no_preprompt/Mistral-7B-Instruct-v0.3/uncertainty.csv \
  --output_dir SIGNET/outputs/semantic_entropy
```

### 7.3 SIGNET（NLI 图）
```bash
python SIGNET/run_semantic_entropy_signet_nli.py \
  --json_path data/no_preprompt/Mistral-7B-Instruct-v0.3/hotpot_qa_final_labeled_results.json \
  --csv_path data/no_preprompt/Mistral-7B-Instruct-v0.3/uncertainty.csv \
  --output_dir SIGNET/outputs/semantic_entropy_nli
```

## 8. 复现实验注意事项
- `hotpot_qa_final_labeled_results.json` 体积大（约 491MB），读取较慢。
- `cluster_ids` 在部分样本缺失，脚本需先过滤。
- 若以 `k@100` 作为 oracle，要明确其是有限样本经验 oracle，不是总体真值。
- `miller` / `len_boost` / `ceil_hybrid_ggt` 都是变体后处理，论文中请单独说明。
