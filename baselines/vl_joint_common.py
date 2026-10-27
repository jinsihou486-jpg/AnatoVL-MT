# -*- coding: utf-8 -*-
"""Shared training loop for DenseCLIP-style / MaPLe-style joint baselines."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from folder_dataset import FolderSegClsDataset, resolve_task_paths
from metrics_utils import classification_metrics, save_mean_std_table


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
        "n_images": len(dice_list),
        "Dice": round(float(np.mean(dice_list)) if dice_list else 0.0, 4),
        **{k: cls[k] for k in ("ClsAcc", "macro_F1", "Kappa_QWK", "n_cls", "report")},
    }


def train_one(args, seed: int, model_ctor, method_name: str) -> dict:
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

    model = model_ctor().to(device)
    if getattr(args, "freeze_backbone", True):
        for n, p in model.named_parameters():
            train_keys = ("cls_head", "seg_head", "prompt", "ctx", "refine", "logit_scale")
            p.requires_grad = any(k in n for k in train_keys)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out_dir) / f"{args.dataset}-{args.organ}" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{method_name} {args.organ} seed{seed} ep{epoch}"):
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
            losses.append(float(loss.item()))
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

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    test_m = evaluate(model, test_loader, device)
    result = {
        "method": method_name,
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
    print("TEST", {k: result[k] for k in result if k != "report"})
    return result


def run_all(args, model_ctor, method_name: str, meanstd_prefix: str):
    tasks = (
        [("RIM-ONE", "Cup"), ("RIM-ONE", "Disc"), ("Drishti-GS1", "Cup"), ("Drishti-GS1", "Disc")]
        if args.all_tasks
        else [(args.dataset, args.organ)]
    )
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    all_rows = []
    for ds, organ in tasks:
        args.dataset, args.organ = ds, organ
        try:
            resolve_task_paths(args.data_root, ds, organ)
        except Exception as e:
            print(f"[跳过] {ds}-{organ}: {e}")
            continue
        for seed in seeds:
            try:
                all_rows.append(train_one(args, seed, model_ctor, method_name))
            except Exception as e:
                print(f"[失败] {ds}-{organ} seed{seed}: {e}")
                import traceback

                traceback.print_exc()
    out = Path(args.out_dir) / "all_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_rows:
        save_mean_std_table(
            all_rows,
            Path(args.out_dir) / f"{meanstd_prefix}_meanstd.csv",
            Path(args.out_dir) / f"{meanstd_prefix}_meanstd.md",
        )
    print(f"wrote {out} n={len(all_rows)}")
