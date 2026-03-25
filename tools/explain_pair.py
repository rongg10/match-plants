#!/usr/bin/env python3
"""Explainability tool for Siamese ResNet18 plant matcher.

Usage example:
  python tools/explain_pair.py --img1 data/data/0.jpg --img2 data/data/1.jpg \
    --checkpoint /path/to/model.pth --outdir explain_outputs --layers backbone.layer4
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms


@dataclass
class AttributionTopK:
    indices: List[int]
    values: List[float]


def build_model() -> nn.Module:
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

    return SiameseNet()


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> Tuple[List[str], List[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            # Assume raw state_dict
            state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    # Strip DataParallel prefix if present
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Layer '{name}' not found. Available: {list(modules.keys())[:20]} ...")
    return modules[name]


def make_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    display_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    return val_tf, display_tf


def load_image(path: str, val_tf, display_tf) -> Tuple[torch.Tensor, Image.Image]:
    img = Image.open(path).convert("RGB")
    display_img = display_tf(img)
    tensor = val_tf(img)
    return tensor, display_img


def topk_from_vector(vec: np.ndarray, k: int) -> AttributionTopK:
    if k <= 0:
        return AttributionTopK(indices=[], values=[])
    abs_vec = np.abs(vec)
    k = min(k, abs_vec.shape[0])
    idx = np.argpartition(-abs_vec, k - 1)[:k]
    idx = idx[np.argsort(-abs_vec[idx])]
    return AttributionTopK(indices=idx.tolist(), values=vec[idx].tolist())


def _fallback_bar_image(values: np.ndarray, indices: List[int], title: str) -> Image.Image:
    from PIL import ImageDraw

    w, h = 700, 300
    pad = 30
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((10, 5), title, fill=(0, 0, 0))

    if len(indices) == 0:
        return img

    max_val = float(np.max(np.abs(values))) if np.max(np.abs(values)) > 0 else 1.0
    bar_w = max(1, (w - 2 * pad) // len(indices))
    baseline = h - pad
    for i, (idx, val) in enumerate(zip(indices, values)):
        x0 = pad + i * bar_w
        x1 = x0 + bar_w - 2
        bar_h = int((abs(val) / max_val) * (h - 2 * pad))
        y0 = baseline - bar_h
        draw.rectangle([x0, y0, x1, baseline], fill=(59, 130, 246))
        # small index label
        draw.text((x0, baseline + 2), str(idx), fill=(0, 0, 0))
    return img


def save_bar(values: np.ndarray, indices: List[int], out_path: Path, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 3))
        ax = fig.add_subplot(111)
        x = np.arange(len(indices))
        ax.bar(x, values, color="#3b82f6")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in indices], rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel("channel")
        ax.set_ylabel("score")
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
    except Exception:
        img = _fallback_bar_image(values, indices, title)
        img.save(out_path)


def _simple_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0, None)
    if cam.max() > 1e-6:
        cam = cam / cam.max()
    cam = np.clip(cam, 0, 1)
    # red-yellow heatmap approximation
    heat = np.zeros((cam.shape[0], cam.shape[1], 3), dtype=np.float32)
    heat[..., 0] = cam  # red
    heat[..., 1] = cam * 0.8  # yellow tint
    return heat


def overlay_cam(image: Image.Image, cam: np.ndarray, out_path: Path, title: str):
    # Resize cam to image size if needed
    if cam.shape[0] != image.size[1] or cam.shape[1] != image.size[0]:
        cam_norm = np.clip(cam, 0, None)
        if cam_norm.max() > 1e-6:
            cam_norm = cam_norm / cam_norm.max()
        cam_img = Image.fromarray((cam_norm * 255).astype(np.uint8))
        cam_img = cam_img.resize(image.size, resample=Image.BILINEAR)
        cam = np.array(cam_img).astype(np.float32) / 255.0

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        cam = np.clip(cam, 0, None)
        if cam.max() > 1e-6:
            cam = cam / cam.max()
        cam = np.clip(cam, 0, 1)

        img = np.array(image).astype(np.float32) / 255.0
        heat = cm.get_cmap("jet")(cam)[..., :3]
        overlay = (0.6 * img + 0.4 * heat)
        overlay = np.clip(overlay, 0, 1)

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.imshow(overlay)
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    except Exception:
        img = np.array(image).astype(np.float32) / 255.0
        heat = _simple_heatmap(cam)
        overlay = np.clip(0.6 * img + 0.4 * heat, 0, 1)
        out = Image.fromarray((overlay * 255).astype(np.uint8))
        out.save(out_path)


def forward_with_features(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor):
    f1 = model.backbone(x1)
    f2 = model.backbone(x2)
    feat = torch.cat([torch.abs(f1 - f2), f1 * f2], dim=1)
    logit = model.head(feat).squeeze(1)
    return f1, f2, feat, logit


def main():
    parser = argparse.ArgumentParser(description="Explain a Siamese ResNet18 pair prediction")
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth/.pt)")
    parser.add_argument("--outdir", default="explain_outputs", help="Output directory")
    parser.add_argument(
        "--layers",
        default="backbone.layer4",
        help="Comma-separated module names for activation analysis",
    )
    parser.add_argument("--topk", type=int, default=10, help="Top-k channels/dims to report")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    val_tf, display_tf = make_transforms()
    x1, disp1 = load_image(args.img1, val_tf, display_tf)
    x2, disp2 = load_image(args.img2, val_tf, display_tf)

    device = torch.device(args.device)

    model = build_model().to(device)
    missing, unexpected = load_checkpoint(model, args.checkpoint)
    if missing or unexpected:
        print("[warn] Missing keys:", missing)
        print("[warn] Unexpected keys:", unexpected)

    model.eval()

    layer_names = [name.strip() for name in args.layers.split(",") if name.strip()]
    hooks = []
    activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}

    def make_hook(layer_name):
        def hook(_module, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            activations[layer_name].append(out)
            out.retain_grad()
        return hook

    for name in layer_names:
        module = get_module_by_name(model, name)
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.enable_grad():
        x1b = x1.unsqueeze(0).to(device)
        x2b = x2.unsqueeze(0).to(device)
        f1, f2, feat, logit = forward_with_features(model, x1b, x2b)
        f1.retain_grad()
        f2.retain_grad()
        feat.retain_grad()
        logit.backward()

    for h in hooks:
        h.remove()

    prob = torch.sigmoid(logit).item()
    pred = int(prob >= args.threshold)

    # Feature-level attributions
    feat_attr = (feat.grad * feat).detach().cpu().numpy()[0]
    f1_attr = (f1.grad * f1).detach().cpu().numpy()[0]
    f2_attr = (f2.grad * f2).detach().cpu().numpy()[0]

    diff_attr = feat_attr[:512]
    mul_attr = feat_attr[512:]

    topk_feat_diff = topk_from_vector(diff_attr, args.topk)
    topk_feat_mul = topk_from_vector(mul_attr, args.topk)
    topk_f1 = topk_from_vector(f1_attr, args.topk)
    topk_f2 = topk_from_vector(f2_attr, args.topk)

    # Layer activation analysis
    layer_reports = {}

    for layer_name, acts in activations.items():
        if len(acts) != 2:
            raise RuntimeError(
                f"Layer {layer_name} expected 2 activations (x1/x2) but got {len(acts)}"
            )
        act1 = acts[0].detach().cpu()[0]
        act2 = acts[1].detach().cpu()[0]
        grad1 = acts[0].grad.detach().cpu()[0]
        grad2 = acts[1].grad.detach().cpu()[0]

        ch1 = act1.abs().mean(dim=(1, 2)).numpy()
        ch2 = act2.abs().mean(dim=(1, 2)).numpy()
        diff = np.abs(ch1 - ch2)

        imp1 = (grad1 * act1).mean(dim=(1, 2)).numpy()
        imp2 = (grad2 * act2).mean(dim=(1, 2)).numpy()

        topk_ch1 = topk_from_vector(ch1, args.topk)
        topk_ch2 = topk_from_vector(ch2, args.topk)
        topk_diff = topk_from_vector(diff, args.topk)
        topk_imp1 = topk_from_vector(imp1, args.topk)
        topk_imp2 = topk_from_vector(imp2, args.topk)

        # Grad-CAM heatmaps
        w1 = grad1.mean(dim=(1, 2))
        w2 = grad2.mean(dim=(1, 2))
        cam1 = (w1[:, None, None] * act1).sum(dim=0).numpy()
        cam2 = (w2[:, None, None] * act2).sum(dim=0).numpy()

        layer_dir = outdir / layer_name.replace(".", "_")
        layer_dir.mkdir(parents=True, exist_ok=True)

        save_bar(ch1[topk_ch1.indices], topk_ch1.indices, layer_dir / "active_img1.png",
                 f"{layer_name} active (img1)")
        save_bar(ch2[topk_ch2.indices], topk_ch2.indices, layer_dir / "active_img2.png",
                 f"{layer_name} active (img2)")
        save_bar(diff[topk_diff.indices], topk_diff.indices, layer_dir / "diff.png",
                 f"{layer_name} activation diff")

        overlay_cam(disp1, cam1, layer_dir / "gradcam_img1.png", f"{layer_name} Grad-CAM (img1)")
        overlay_cam(disp2, cam2, layer_dir / "gradcam_img2.png", f"{layer_name} Grad-CAM (img2)")

        layer_reports[layer_name] = {
            "active_channels_img1": {
                "indices": topk_ch1.indices,
                "values": topk_ch1.values,
            },
            "active_channels_img2": {
                "indices": topk_ch2.indices,
                "values": topk_ch2.values,
            },
            "diff_channels": {
                "indices": topk_diff.indices,
                "values": topk_diff.values,
            },
            "influence_channels_img1": {
                "indices": topk_imp1.indices,
                "values": topk_imp1.values,
            },
            "influence_channels_img2": {
                "indices": topk_imp2.indices,
                "values": topk_imp2.values,
            },
            "artifacts": {
                "active_img1": str((layer_dir / "active_img1.png").resolve()),
                "active_img2": str((layer_dir / "active_img2.png").resolve()),
                "diff": str((layer_dir / "diff.png").resolve()),
                "gradcam_img1": str((layer_dir / "gradcam_img1.png").resolve()),
                "gradcam_img2": str((layer_dir / "gradcam_img2.png").resolve()),
            },
        }

    report = {
        "inputs": {
            "img1": os.path.abspath(args.img1),
            "img2": os.path.abspath(args.img2),
            "checkpoint": os.path.abspath(args.checkpoint),
            "layers": layer_names,
            "topk": args.topk,
            "threshold": args.threshold,
            "device": str(device),
        },
        "prediction": {
            "logit": float(logit.item()),
            "prob": float(prob),
            "label": "same" if pred == 1 else "different",
        },
        "feature_attributions": {
            "diff_512": {
                "indices": topk_feat_diff.indices,
                "values": topk_feat_diff.values,
            },
            "mul_512": {
                "indices": topk_feat_mul.indices,
                "values": topk_feat_mul.values,
            },
            "f1_512": {
                "indices": topk_f1.indices,
                "values": topk_f1.values,
            },
            "f2_512": {
                "indices": topk_f2.indices,
                "values": topk_f2.values,
            },
        },
        "layers": layer_reports,
    }

    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Also write a small markdown summary
    summary_path = outdir / "report.md"
    summary_lines = []
    summary_lines.append(f"Prediction: {report['prediction']['label']}  (prob={prob:.4f}, logit={logit.item():.4f})")
    summary_lines.append("")
    summary_lines.append("Top feature dims (|f1-f2|):")
    for idx, val in zip(topk_feat_diff.indices, topk_feat_diff.values):
        summary_lines.append(f"- {idx}: {val:.6f}")
    summary_lines.append("")
    summary_lines.append("Top feature dims (f1*f2):")
    for idx, val in zip(topk_feat_mul.indices, topk_feat_mul.values):
        summary_lines.append(f"- {idx}: {val:.6f}")
    summary_lines.append("")
    summary_lines.append("Top embedding dims (f1):")
    for idx, val in zip(topk_f1.indices, topk_f1.values):
        summary_lines.append(f"- {idx}: {val:.6f}")
    summary_lines.append("")
    summary_lines.append("Top embedding dims (f2):")
    for idx, val in zip(topk_f2.indices, topk_f2.values):
        summary_lines.append(f"- {idx}: {val:.6f}")
    summary_lines.append("")

    for layer_name, layer_info in layer_reports.items():
        summary_lines.append(f"Layer: {layer_name}")
        summary_lines.append("Active channels img1:")
        for idx, val in zip(
            layer_info["active_channels_img1"]["indices"],
            layer_info["active_channels_img1"]["values"],
        ):
            summary_lines.append(f"- {idx}: {val:.6f}")
        summary_lines.append("Active channels img2:")
        for idx, val in zip(
            layer_info["active_channels_img2"]["indices"],
            layer_info["active_channels_img2"]["values"],
        ):
            summary_lines.append(f"- {idx}: {val:.6f}")
        summary_lines.append("Diff channels:")
        for idx, val in zip(
            layer_info["diff_channels"]["indices"],
            layer_info["diff_channels"]["values"],
        ):
            summary_lines.append(f"- {idx}: {val:.6f}")
        summary_lines.append("")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
