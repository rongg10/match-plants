# Match Plants：植物图像相似度（Siamese ResNet18）

本仓库包含用于“同株/不同株”判别的孪生网络笔记本，以及一组消融实验（多随机种子）结果汇总。

## 项目内容

- 主模型：Siamese ResNet18（ImageNet 预训练）。
- 训练/验证：对成对图像进行训练，验证集上搜索最佳阈值 `best_t` 以最大化 F1。
- 输出：生成 `yourname_results.csv`（包含 `Pair_Num` 与 `Predicted_Result`）。
- 消融实验：每个消融一个独立 `ipynb`，并进行多随机种子评估。

## 主要文件

- 训练与基线：`match_plants.ipynb`
- 消融实验（已收纳）：
  - `notebooks/ablations/match_plants_baseline.ipynb`
  - `notebooks/ablations/match_plants_ablation_no_pretrain.ipynb`
  - `notebooks/ablations/match_plants_ablation_freeze_backbone.ipynb`
  - `notebooks/ablations/match_plants_ablation_absdiff_only.ipynb`
  - `notebooks/ablations/match_plants_ablation_mul_only.ipynb`
  - `notebooks/ablations/match_plants_ablation_no_augmentation.ipynb`
  - `notebooks/ablations/match_plants_ablation_no_pos_weight.ipynb`
  - `notebooks/ablations/match_plants_ablation_fixed_threshold.ipynb`
- 消融结果汇总：
  - 单种子汇总：`reports/single_seed/ablation_summary.csv`
  - 三种子明细：`reports/multi_seed_3/ablation_summary_seeds.csv`
  - 三种子统计：`reports/multi_seed_3/ablation_summary_stats.csv`
- 执行产物：
  - 三种子执行后的临时 notebook：`runs/multi_seed_3/ablation_runs/`

## 模型简介

孪生网络结构如下：

- **Backbone**：ResNet18（ImageNet 预训练）
- **Embedding**：去掉最终分类层，得到 512 维特征
- **特征对比**：拼接 `|f1 - f2|` 与 `f1 * f2`，形成 1024 维向量
- **Head**：MLP（1024→256→1）输出单个 logit

## 训练与阈值

- 损失函数：`BCEWithLogitsLoss(pos_weight=neg/pos)`
- 验证指标：正类 F1
- 阈值选择：在验证集上扫描 0.1～0.9，取最大 F1 的 `best_t`

## 多随机种子消融结果（F1）

随机种子：`0/1/2`，结果为 **均值 ± 标准差**。

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

## 结论（基于 3 个种子）

- **预训练是决定性因素**：去掉预训练后 F1 大幅下降（约 -0.41）。
- **需要微调 backbone**：冻结 backbone 明显伤害性能。
- **组合特征更好**：`|f1-f2|` 或 `f1*f2` 单独使用略差，组合更稳健。
- **阈值搜索有小幅收益**：固定 0.5 略低于验证集调阈值。
- **增强/类别权重未体现收益**：去掉增强或 `pos_weight` 的 F1 略高，且方差更小；建议进一步验证增强强度与权重策略。

## 运行方式

在项目根目录：

```
pyenv shell eoitek
jupyter lab
```

打开对应 `ipynb` 并按顺序运行。完成后会生成 `yourname_results.csv`，提交前按要求重命名。

## 备注

- 当前验证划分基于“成对样本”，可能出现同一图像同时出现在训练/验证中，F1 可能偏乐观。
- 如果需要更严格评估，可按图像 ID 分组划分训练/验证，避免图像泄漏。
