#!/usr/bin/env python3
"""
联合基线：CLIPSeg + 四级分类头
对齐 MedCLIPSeg 论文中的较弱视觉–语言分割对比；额外加分类头做联合学习。

依赖:
  pip install transformers

用法:
  python baselines/train_clipseg_joint.py --data-root /path/to/data --all-tasks --seeds 298,1,42
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


class CLIPSegJoint(nn.Module):
    """HuggingFace CLIPSeg + 全局分类头。"""

    def __init__(self, model_name: str = "CIDAS/clipseg-rd64-refined", num_classes: int = 4):
        super().__init__()
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(model_name)
        # CLIPSeg vision hidden size
        hidden = self.clipseg.config.projection_dim
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )
        self._last_pooled = None

    def forward(self, images: torch.Tensor, texts: list[str]):
        """
        images: 已归一化的 [B,3,H,W]（ImageNet norm）。CLIPSeg 内部期望 processor 输出，
        这里用像素值近似：反归一化后喂 processor 太慢，改为直接调用 clipseg 的视觉编码路径。
        简化：用 interpolate 到 352，并调用 model(pixel_values=..., input_ids=...)。
        """
        device = images.device
        # 反归一化到 [0,1] 再按 CLIPSeg 习惯处理
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        imgs = (images * std + mean).clamp(0, 1)

        # processor 批处理文本 + 图像
        # 注意：processor 需要 PIL/numpy；为稳定训练，用 clipseg 自带 encode
        inputs = self.processor(
            text=texts,
            images=[img.detach().cpu().permute(1, 2, 0).numpy() for img in imgs],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.clipseg(**inputs)
        # logits: [B, H, W]
        seg_logits = outputs.logits
        if seg_logits.ndim == 3:
            seg_logits = seg_logits.unsqueeze(1)
        # 对齐到输入分辨率
        seg_logits = F.interpolate(seg_logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        # 用视觉 pooled 做分类：取 decoder 前的条件特征不可直接拿，改用 pixel encoder CLS 近似
        # clipseg.clip.vision_model
        vision_out = self.clipseg.clip.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_out.pooler_output  # [B, D_v]
        # 投影到 projection_dim
        pooled = self.clipseg.clip.visual_projection(pooled)
        self._last_pooled = pooled
        class_logits = self.cls_head(pooled)
        return seg_logits, class_logits


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
        texts = batch["text"]
        seg, logits = model(x, texts)
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
        "Dice": round(float(np.mean(dice_list)) if dice_list else 0.0, 4),
        **{k: cls[k] for k in ("ClsAcc", "macro_F1", "Kappa_QWK", "report")},
    }


def train_one(args, seed: int) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    paths = resolve_task_paths(args.data_root, args.dataset, args.organ)

    train_ds = FolderSegClsDataset(
        paths["train"], paths["train_xlsx"], args.size, train=True, need_text=True
    )
    val_ds = FolderSegClsDataset(paths["val"], paths["val_xlsx"], args.size, train=False, need_text=True)
    test_ds = FolderSegClsDataset(paths["test"], paths["test_xlsx"], args.size, train=False, need_text=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    grades = [p[3] for p in train_ds.pairs if p[3] > 0]
    counts = np.bincount(np.array(grades, dtype=int), minlength=5)[1:5].astype(float)
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights / weights.sum() * 4.0
    cls_crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    bce = nn.BCEWithLogitsLoss()

    model = CLIPSegJoint(model_name=args.model_name, num_classes=4).to(device)
    # 默认微调分类头 + decoder；可选全参
    if args.freeze_encoder:
        for n, p in model.named_parameters():
            if n.startswith("cls_head") or "decoder" in n or "transposed" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out_dir) / f"{args.dataset}-{args.organ}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"CLIPSeg {args.organ} seed{seed} ep{epoch}"):
            x = batch["image"].to(device)
            m = batch["mask"].to(device)
            y = batch["class_label"].to(device)
            texts = batch["text"]
            seg, logits = model(x, texts)
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
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val": val_m, "seed": seed},
                out_dir / "best.pt",
            )

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    test_m = evaluate(model, test_loader, device)
    result = {
        "method": "CLIPSeg-Joint",
        "dataset": args.dataset,
        "organ": args.organ,
        "seed": seed,
        "Dice": test_m["Dice"],
        "ClsAcc": test_m["ClsAcc"],
        "macro_F1": test_m["macro_F1"],
        "Kappa_QWK": test_m["Kappa_QWK"],
        "report": test_m["report"],
        "ckpt": str(out_dir / "best.pt"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TEST", {k: result[k] for k in result if k != "report"})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", default="RIM-ONE", choices=["RIM-ONE", "Drishti-GS1"])
    p.add_argument("--organ", default="Cup", choices=["Cup", "Disc"])
    p.add_argument("--out-dir", default=str(ROOT / "baselines" / "runs_clipseg"))
    p.add_argument("--model-name", default="CIDAS/clipseg-rd64-refined")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-cls", type=float, default=0.5)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--no-freeze-encoder", action="store_false", dest="freeze_encoder")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--all-tasks", action="store_true")
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

    save_mean_std_table(
        all_rows, Path(args.out_dir) / "clipseg_meanstd.csv", Path(args.out_dir) / "clipseg_meanstd.md"
    )


if __name__ == "__main__":
    main()
