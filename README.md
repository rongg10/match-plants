# Match Plants: Image Similarity (Siamese ResNet18)

This repo contains a Jupyter notebook that trains a pairwise image similarity model to decide whether two plant images depict the same plant. The notebook produces a CSV submission in the required format.

## What was done

- Built a Siamese neural network using a pretrained ResNet18 backbone from `torchvision`.
- Trained on the provided pair labels (`train_data.csv`) with data augmentation.
- Tuned a decision threshold on a held-out validation split to maximize F1 for the positive class.
- Retrained on the full training data and generated predictions for `test_data.csv`.
- Wrote the submission file `yourname_results.csv` with columns:
  - `Pair_Num`
  - `Predicted_Result`

The notebook is `match_plants.ipynb` and is intended to run in the `eoitek` pyenv environment on Apple Silicon (MPS is used when available).

## Model overview

The model is a Siamese network:

- **Backbone:** ResNet18 pretrained on ImageNet.
- **Embedding:** The final classification layer is removed, so each image is mapped to a 512-dim feature vector.
- **Pair fusion:** For a pair of features `(f1, f2)`, the model computes:
  - `abs(f1 - f2)`
  - `f1 * f2`
  These are concatenated into a 1024-dim vector.
- **Head:** A small MLP predicts a single logit for "same plant" vs "different plant".

This is a standard setup for image similarity: the backbone learns a robust representation and the head learns how to compare two embeddings.

## Loss function

The notebook uses **binary cross-entropy with logits**:

```
BCEWithLogitsLoss(pos_weight=neg/pos)
```

- `pos_weight` balances the classes because the dataset has more negative pairs than positive pairs.
- The model outputs logits; `torch.sigmoid` converts logits to probabilities.

## Validation F1 and threshold (`best_t`)

### `val_f1`
`val_f1` is the **F1 score for the positive class** (same plant) computed on a validation split of pairs. It is calculated from precision and recall:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

The competition scoring uses the F1 of the positive class, so this is the metric the notebook optimizes.

### `best_t`
`best_t` is the **decision threshold** applied to the predicted probabilities. Instead of using the default 0.5, the notebook scans thresholds from 0.1 to 0.9 and picks the one that yields the highest validation F1.

Example:
- If `best_t = 0.68`, then pairs with probability >= 0.68 are predicted as "same plant".

The tuned threshold is then reused when generating the final test predictions.

## Data flow

1. Read training pairs from `data/train_data.csv`.
2. Split into train/validation pairs for threshold tuning.
3. Train the Siamese model for a few epochs with augmentation.
4. Compute `val_f1` over a range of thresholds and store `best_t`.
5. Retrain on the full training data.
6. Predict for `data/test_data.csv`.
7. Write `yourname_results.csv`.

## How to run

From the project directory:

```
pyenv shell eoitek
jupyter lab
```

Then open `match_plants.ipynb` and run all cells. The notebook will:

- Print training progress and validation F1.
- Save the submission file as `yourname_results.csv`.

Rename `yourname_results.csv` to your required naming format before submission.

## Notes

- The validation split is done on pairs (not on image IDs), so some images may appear in both train and validation splits. This can make `val_f1` optimistic.
- If you want stricter evaluation, split by image IDs so pairs do not share images across splits.
- If you increase epochs or use a larger backbone, you may improve accuracy at the cost of time.
