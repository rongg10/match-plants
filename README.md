# Match Plants: Same-Plant Image Identification (Siamese ResNet18)

This project uses a Siamese network to determine whether two plant photos belong to the same individual plant (Same / Different). It includes baseline training, ablation studies, multi-random-seed results, and a "leak-free" validation scheme grouped by image ID.

## Project Structure

- Main notebook (original split, split by pair)  
  - `match_plants.ipynb`
- Split grouped by image ID (to avoid image leakage)  
  - `match_plants_group_split.ipynb`
- Ablation studies (original split)  
  - `notebooks/ablations/`
- Ablation studies (image ID grouped split)  
  - `notebooks/ablations_group_split/`
- Reports / results  
  - Single seed: `reports/single_seed/ablation_summary.csv`
  - Three seeds (original split): `reports/multi_seed_3/ablation_summary_seeds.csv`  
    `reports/multi_seed_3/ablation_summary_stats.csv`
  - Three seeds (grouped by image ID): `reports/multi_seed_3_group_split/ablation_summary_seeds.csv`  
    `reports/multi_seed_3_group_split/ablation_summary_stats.csv`
- Run artifacts (executed seed notebooks)  
  - `runs/multi_seed_3/ablation_runs/`  
  - `runs/multi_seed_3_group_split/ablation_runs_v2/`

## Model and Training Overview

- Backbone: ResNet18 (ImageNet pretrained)
- Features: remove the classification head to obtain 512 dimensions
- Comparison features: concatenate `|f1-f2|` and `f1*f2`
- Head: MLP (1024→256→1)
- Loss: `BCEWithLogitsLoss(pos_weight=neg/pos)`
- Threshold: scan 0.1~0.9 on the validation set and select the best `best_t`

## Why "Grouped by Image ID" Is Needed

The original notebook uses a **random split by pair**, so the same image may appear in both training and validation, which makes validation F1 overly optimistic.  
For that reason, `match_plants_group_split.ipynb` and the corresponding ablation versions were added to split training/validation by image ID and avoid leakage.

## Results Summary (F1, mean ± standard deviation, 3 seeds: 0/1/2)

### Original Split (by pair)

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

### Image ID Grouped Split (avoid leakage)

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

## Main Conclusions (based on 3 seeds)

- **Pretraining and fine-tuning are decisive factors**: removing pretraining or freezing the backbone causes a large drop in F1.
- **Combined features are more robust**: using only `|f1-f2|` or only `f1*f2` is slightly worse than combining them.
- **Threshold search is slightly better than a fixed 0.5**.
- **Image ID grouped results are more conservative**, and closer to true generalization.
