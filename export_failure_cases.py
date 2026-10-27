#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失败/成功案例导出：按 Dice 最低、等级误差最大各取若干张，保存 原图|GT|Pred 拼图。

在 MedCLIPSeg 根目录：

  python export_failure_cases.py \\
      --config-file configs/RIM-ONE-Cup.yaml \\
      --output-dir ./outputs_loss/full \\
      --seed 298 --num-fail 4 --num-ok 2 \\
      --out-dir ./viz/failure_cases_rim_cup
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text


def resolve_test_prompt(prompt_dir: str) -> str:
    for name in ("Test_text.xlsx", "Test_text_original.xlsx"):
        path = os.path.join(prompt_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(prompt_dir)


def find_best(cfg) -> Path:
    backbone = cfg.MODEL.BACKBONE.replace("/", "-")
    name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone}_joint_best.pth"
    d = Path(cfg.output_dir) / cfg.DATASET.NAME / "trained_models" / f"seed{cfg.seed}"
    p = d / name
    if p.is_file():
        return p
    cands = list(d.glob("*_best.pth"))
    if not cands:
        raise FileNotFoundError(d)
    return cands[0]


def dice_np(pred, gt, thr=0.5):
    p = (pred > thr).astype(np.float32)
    g = (gt > 0.5).astype(np.float32)
    inter = (p * g).sum()
    return float((2 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6))


def to_rgb(img_t):
    x = img_t.detach().cpu().float()
    if x.ndim == 3 and x.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x * std + mean).clamp(0, 1)
        return (x.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    arr = x.numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def contour(rgb, mask, color):
    out = rgb.copy()
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def panel(rgb, gt, pred, title):
    g = contour(rgb, gt, (0, 255, 0))
    p = contour(rgb, pred, (0, 0, 255))
    h = np.hstack([rgb, g, p])
    h = cv2.cvtColor(h, cv2.COLOR_RGB2BGR)
    cv2.putText(h, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=298)
    ap.add_argument("--num-fail", type=int, default=4)
    ap.add_argument("--num-ok", type=int, default=2)
    ap.add_argument("--out-dir", default="./viz/failure_cases")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.seed = args.seed
    cfg.output_dir = args.output_dir
    if not hasattr(cfg.MODEL, "NUM_CLASSES"):
        cfg.MODEL.NUM_CLASSES = 4

    device = cfg.MODEL.DEVICE
    test_text = read_text(resolve_test_prompt(cfg.DATASET.TEXT_PROMPT_PATH))
    tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    ds = DatasetSegmentation(
        cfg.DATASET.TEST_PATH, cfg.DATASET.NAME, test_text, tf, image_size=cfg.DATASET.SIZE
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_medclipseg_unimedclip(cfg)
    ckpt_path = find_best(cfg)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=False)
    model.to(device).eval()

    records = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="scan")):
            img = batch["image"].to(device)
            gt_t = batch["ground_truth_mask"]
            gt = gt_t[0, 0].cpu().numpy() if gt_t.ndim == 4 else gt_t[0].cpu().numpy()
            y = int(batch["class_label"].item()) if "class_label" in batch else -1
            text = batch["text_prompt"]

            # 与 eval_joint_test 一致的前向
            out = model(img, text=text, num_samples=1)
            if isinstance(out, tuple) and len(out) == 2:
                seg_samples, class_logits = out
            else:
                raise RuntimeError("期望 (seg_samples, class_logits)")

            logits = seg_samples[0]
            if logits.ndim == 4:
                logits = logits[0, 0]
            elif logits.ndim == 3:
                logits = logits[0]
            pred = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(np.float32)
            d = dice_np(pred, gt)
            yhat = int(class_logits.argmax(dim=-1).item())
            grade_err = abs(yhat - y) if y >= 0 else 0
            name = batch.get("image_name", batch.get("name", [f"idx{i}"]))
            if isinstance(name, (list, tuple)):
                name = name[0]
            records.append(
                {
                    "idx": i,
                    "name": str(name),
                    "dice": d,
                    "y": y,
                    "yhat": yhat,
                    "grade_err": grade_err,
                    "rgb": to_rgb(batch["image"][0]),
                    "gt": gt.astype(np.float32),
                    "pred": pred,
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fail = sorted(records, key=lambda r: (-r["grade_err"], r["dice"]))[: args.num_fail]
    ok = sorted(records, key=lambda r: (-r["dice"], r["grade_err"]))[: args.num_ok]

    manifest = []
    for tag, subset in (("fail", fail), ("ok", ok)):
        for j, r in enumerate(subset):
            title = f"{tag} {r['name']} Dice={r['dice']:.3f} G{r['y']}->{r['yhat']}"
            img = panel(r["rgb"], r["gt"], r["pred"], title)
            path = out_dir / f"{tag}_{j:02d}_dice{r['dice']:.3f}_g{r['y']}to{r['yhat']}.png"
            cv2.imwrite(str(path), img)
            manifest.append(
                {
                    "tag": tag,
                    "name": r["name"],
                    "dice": r["dice"],
                    "y": r["y"],
                    "yhat": r["yhat"],
                    "grade_err": r["grade_err"],
                    "file": path.name,
                }
            )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(manifest)} panels -> {out_dir}")


if __name__ == "__main__":
    main()
