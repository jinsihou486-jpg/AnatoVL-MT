#!/usr/bin/env python3
"""
联合基线：U-Net + 四级分类头
L = L_seg (Dice+BCE) + λ_cls * CE

用法（GPU 服务器）:
  cd medica/joint_task
  python baselines/train_unet_joint.py --data-root /path/to/data --dataset RIM-ONE --organ Cup --seeds 298,1,42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from folder_dataset import FolderSegClsDataset, resolve_task_paths  # noqa: E402
from metrics_utils import DEFAULT_SEEDS, classification_metrics  # noqa: E402


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetJoint(nn.Module):
    def __init__(self, num_classes: int = 4, base: int = 32):
        super().__init__()
        self.down1 = DoubleConv(3, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)
        self.seg_head = nn.Conv2d(base, 1, 1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self._gradcam_feat = None

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bn = self.bottleneck(self.pool(d4))
        self._gradcam_feat = bn
        u4 = self.conv4(torch.cat([self.up4(bn), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        seg = self.seg_head(u1)
        cls = self.cls_head(bn)
        return seg, cls


def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dice_list, ys, preds = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        m = batch["mask"].to(device)
        y = batch["class_label"].to(device)
        seg, logits = model(x)
        pred = (torch.sigmoid(seg) > 0.5).float()
        inter = (pred * m).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3))
        dice_list.extend(((2 * inter + 1e-7) / (union + 1e-7)).cpu().tolist())
        valid = y >= 0
        if valid.any():
            ys.extend(y[valid].cpu().tolist())
            preds.extend(logits[valid].argmax(1).cpu().tolist())
    cls = classification_metrics(ys, preds)
    return {
        "n_images": len(dice_list),
        "Dice": round(float(np.mean(dice_list)) if dice_list else 0.0, 4),
        **{k: cls[k] for k in ("ClsAcc", "macro_F1", "Kappa_QWK", "n_cls", "report")},
    }


def train_one(args, seed: int) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    paths = resolve_task_paths(args.data_root, args.dataset, args.organ)

    train_ds = FolderSegClsDataset(paths["train"], paths["train_xlsx"], args.size, train=True)
    val_ds = FolderSegClsDataset(paths["val"], paths["val_xlsx"], args.size, train=False)
    test_ds = FolderSegClsDataset(paths["test"], paths["test_xlsx"], args.size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 类别权重
    grades = [p[3] for p in train_ds.pairs if p[3] > 0]
    counts = np.bincount(np.array(grades, dtype=int), minlength=5)[1:5].astype(float)
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights / weights.sum() * 4.0
    cls_crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    bce = nn.BCEWithLogitsLoss()

    model = UNetJoint(num_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out_dir) / f"{args.dataset}-{args.organ}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"UNet {args.organ} seed{seed} ep{epoch}"):
            x = batch["image"].to(device)
            m = batch["mask"].to(device)
            y = batch["class_label"].to(device)
            seg, logits = model(x)
            loss = 0.5 * dice_loss(seg, m) + 0.5 * bce(seg, m)
            valid = y >= 0
            if valid.any():
                loss = loss + args.lambda_cls * cls_crit(logits[valid], y[valid])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_m = evaluate(model, val_loader, device)
        score = 0.5 * val_m["Dice"] + 0.5 * val_m["ClsAcc"]
        print(
            f"ep{epoch} loss={np.mean(losses):.4f} "
            f"val Dice={val_m['Dice']:.4f} Acc={val_m['ClsAcc']:.4f} "
            f"F1={val_m['macro_F1']:.4f} Kappa={val_m['Kappa_QWK']:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_m, "seed": seed}, out_dir / "best.pt")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, device)
    result = {
        "method": "UNet-Joint",
        "dataset": args.dataset,
        "organ": args.organ,
        "seed": seed,
        "n_images": test_m["n_images"],
        "n_cls": test_m["n_cls"],
        "Dice": test_m["Dice"],
        "ClsAcc": test_m["ClsAcc"],
        "macro_F1": test_m["macro_F1"],
        "Kappa_QWK": test_m["Kappa_QWK"],
        "report": test_m["report"],
        "ckpt": str(out_dir / "best.pt"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TEST", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="含 RIM-ONE/ Drishti-GS1/ 的根目录")
    p.add_argument("--dataset", default="RIM-ONE", choices=["RIM-ONE", "Drishti-GS1"])
    p.add_argument("--organ", default="Cup", choices=["Cup", "Disc"])
    p.add_argument("--out-dir", default=str(ROOT / "baselines" / "runs_unet"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-cls", type=float, default=0.5)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--all-tasks", action="store_true", help="跑 2 数据集 × Cup/Disc")
    args = p.parse_args()

    tasks = (
        [("RIM-ONE", "Cup"), ("RIM-ONE", "Disc"), ("Drishti-GS1", "Cup"), ("Drishti-GS1", "Disc")]
        if args.all_tasks
        else [(args.dataset, args.organ)]
    )
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    all_rows = []
    for ds, organ in tasks:
        args.dataset, args.organ = ds, organ
        for seed in seeds:
            all_rows.append(train_one(args, seed))

    out = Path(args.out_dir) / "all_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    from metrics_utils import save_mean_std_table

    save_mean_std_table(all_rows, Path(args.out_dir) / "unet_meanstd.csv", Path(args.out_dir) / "unet_meanstd.md")
    print(f"汇总已写 {out}")


if __name__ == "__main__":
    main()
