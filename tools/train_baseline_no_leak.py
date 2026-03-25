#!/usr/bin/env python3
"""Train baseline Siamese ResNet18 with pair-based split (leakage risk) and save checkpoint.

This mirrors match_plants.ipynb baseline settings.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class PairDataset(Dataset):
    def __init__(self, df, transform, img_dir, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = Image.open(os.path.join(self.img_dir, f"{row.img_idx1}.jpg")).convert("RGB")
        img2 = Image.open(os.path.join(self.img_dir, f"{row.img_idx2}.jpg")).convert("RGB")
        x1 = self.transform(img1)
        x2 = self.transform(img2)
        if self.is_test:
            return x1, x2
        return x1, x2, torch.tensor(row["class"], dtype=torch.float32)


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        feat = torch.cat([torch.abs(f1 - f2), f1 * f2], dim=1)
        return self.head(feat).squeeze(1)


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data_dir = "data"
    img_dir = os.path.join(data_dir, "data")
    train_csv = os.path.join(data_dir, "train_data.csv")

    train_df = pd.read_csv(train_csv)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_pairs, val_pairs = train_test_split(
        train_df, test_size=0.2, stratify=train_df["class"], random_state=42
    )

    train_ds = PairDataset(train_pairs, train_tf, img_dir)
    val_ds = PairDataset(val_pairs, val_tf, img_dir)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = SiameseNet().to(device)

    pos = float(train_pairs["class"].sum())
    neg = float(len(train_pairs) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_f1 = 0.0
    best_t = 0.5

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for x1, x2, y in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        model.eval()
        all_probs = []
        all_y = []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                logits = model(x1, x2)
                probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                all_probs.append(probs)
                all_y.append(y.numpy().astype(np.float32))

        all_probs = np.concatenate(all_probs)
        all_y = np.concatenate(all_y)

        best_epoch_t, best_epoch_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 81):
            pred = (all_probs >= t).astype(int)
            f1 = f1_score(all_y, pred)
            if f1 > best_epoch_f1:
                best_epoch_f1 = f1
                best_epoch_t = t

        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_t = best_epoch_t

        print(
            f"epoch {epoch} loss {total_loss/len(train_ds):.4f} "
            f"val_f1 {best_epoch_f1:.4f} best_t {best_epoch_t:.2f}"
        )

    print("best overall", best_f1, best_t)

    # Train final model on full training set (matches notebook)
    full_ds = PairDataset(train_df, train_tf, img_dir)
    full_loader = DataLoader(full_ds, batch_size=16, shuffle=True, num_workers=0)

    final_model = SiameseNet().to(device)
    pos = float(train_df["class"].sum())
    neg = float(len(train_df) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, 6):
        final_model.train()
        total_loss = 0.0
        for x1, x2, y in full_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = final_model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
    print(f"final epoch {epoch} loss {total_loss/len(full_ds):.4f}")

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "baseline_no_leak_final.pth"
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"saved {ckpt_path}")

    metrics_path = out_dir / "baseline_no_leak_metrics.json"
    metrics_path.write_text(
        "{\n"
        f"  \"best_f1\": {best_f1},\n"
        f"  \"best_t\": {best_t}\n"
        "}\n",
        encoding="utf-8",
    )
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
