# Match Plants：植物图像同株判别（Siamese ResNet18）

本项目用孪生网络判断两张植物照片是否为同一株（Same / Different）。包含基线训练、消融实验、多随机种子结果，以及按图像 ID 分组的“无泄漏”验证方案。

## 项目结构

- 主 notebook（原始划分，按 pair 切分）  
  - `match_plants.ipynb`
- 按图像 ID 分组划分（避免图像泄漏）  
  - `match_plants_group_split.ipynb`
- 消融实验（原始划分）  
  - `notebooks/ablations/`
- 消融实验（图像 ID 分组划分）  
  - `notebooks/ablations_group_split/`
- 报告/结果  
  - 单种子：`reports/single_seed/ablation_summary.csv`
  - 三种子（原始划分）：`reports/multi_seed_3/ablation_summary_seeds.csv`  
    `reports/multi_seed_3/ablation_summary_stats.csv`
  - 三种子（图像 ID 分组）：`reports/multi_seed_3_group_split/ablation_summary_seeds.csv`  
    `reports/multi_seed_3_group_split/ablation_summary_stats.csv`
- 运行产物（已执行的 seed notebooks）  
  - `runs/multi_seed_3/ablation_runs/`  
  - `runs/multi_seed_3_group_split/ablation_runs_v2/`

## 模型与训练概述

- Backbone：ResNet18（ImageNet 预训练）
- 特征：去掉分类头得到 512 维
- 对比特征：拼接 `|f1-f2|` 与 `f1*f2`
- Head：MLP（1024→256→1）
- 损失：`BCEWithLogitsLoss(pos_weight=neg/pos)`
- 阈值：验证集扫描 0.1~0.9 取最佳 `best_t`

## 为什么需要“按图像 ID 分组”

原始 notebook 使用 **按 pair 的随机切分**，同一图像可能同时出现在训练与验证中，导致验证 F1 偏乐观。  
因此新增 `match_plants_group_split.ipynb` 及对应消融版本，按图像 ID 划分训练/验证，避免泄漏。

## 结果汇总（F1，均值 ± 标准差，3 seeds: 0/1/2）

### 原始划分（按 pair）

```
Baseline                0.9783 ± 0.0081
No pretrain             0.5627 ± 0.0104
Freeze backbone         0.6271 ± 0.0169
Abs diff only           0.9612 ± 0.0112
Mul only                0.9732 ± 0.0037
No augmentation         0.9917 ± 0.0018
No pos_weight           0.9897 ± 0.0017
Fixed threshold 0.5     0.9740 ± 0.0091
```

### 图像 ID 分组划分（避免泄漏）

```
Baseline                0.9604 ± 0.0250
No pretrain             0.5916 ± 0.0340
Freeze backbone         0.6728 ± 0.0348
Abs diff only           0.9174 ± 0.0139
Mul only                0.9370 ± 0.0268
No augmentation         0.9489 ± 0.0155
No pos_weight           0.9606 ± 0.0085
Fixed threshold 0.5     0.9233 ± 0.0174
```

## 主要结论（基于 3 个种子）

- **预训练与微调是决定性因素**：去掉预训练或冻结 backbone，F1 大幅下降。
- **组合特征更稳健**：单用 `|f1-f2|` 或 `f1*f2` 均略差于组合。
- **阈值搜索略优于固定 0.5**。
- **图像 ID 分组结果更保守**，更接近真实泛化。