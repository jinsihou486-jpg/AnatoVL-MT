#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂度测量：Params / Trainable / Infer ms / FPS
在 MedCLIPSeg 根目录执行：

  python measure_complexity.py --config-file configs/RIM-ONE-Cup.yaml
  # 可选对照 U-Net（需 joint_task/baselines）：
  python measure_complexity.py --config-file configs/RIM-ONE-Cup.yaml \\
      --unet-ckpt baselines/runs_unet/RIM-ONE-Cup/seed298/best.pt

输出：paper_table_complexity.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch


def count_params(model):
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_all, n_train


@torch.no_grad()
def bench_fps(forward_fn, warmup=20, runs=100, device="cuda"):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup):
        forward_fn()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        forward_fn()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / runs
    return ms, 1000.0 / ms


def measure_ours(cfg_path: str, device: str, size: int = 224):
    from utils.main_utils import load_cfg_from_cfg_file
    from trainers import build_medclipseg_unimedclip

    try:
        from freeze_utils import apply_freeze
    except ImportError:
        apply_freeze = None

    cfg = load_cfg_from_cfg_file(cfg_path)
    cfg.MODEL.DEVICE = device
    if not hasattr(cfg.MODEL, "NUM_CLASSES"):
        cfg.MODEL.NUM_CLASSES = 4
    model = build_medclipseg_unimedclip(cfg)
    if apply_freeze is not None:
        model = apply_freeze(model, "adapter")
    model.eval().to(device)
    n_all, n_train = count_params(model)

    img = torch.randn(1, 3, size, size, device=device)
    text = ["optic cup"]

    def fwd():
        out = model(img, text=text, num_samples=1)
        if isinstance(out, (tuple, list)):
            _ = out[0]
        return out

    with torch.no_grad():
        fwd()

    mem_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fwd()
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    ms, fps = bench_fps(fwd, device=device)
    return {
        "method": "Ours-adapter",
        "params_M": round(n_all / 1e6, 3),
        "trainable_M": round(n_train / 1e6, 3),
        "infer_ms": round(ms, 2),
        "FPS": round(fps, 2),
        "gpu_mem_MB": round(mem_mb, 1) if mem_mb is not None else "",
        "input": f"1x3x{size}x{size}",
    }


def measure_unet(ckpt: str | None, device: str, size: int = 224):
    # 尽量从 joint_task/baselines 导入
    here = Path(__file__).resolve().parent
    for p in (here / "baselines", here / "joint_task" / "baselines", Path.cwd() / "joint_task" / "baselines"):
        if (p / "train_unet_joint.py").exists():
            sys.path.insert(0, str(p))
            break
    from train_unet_joint import UNetJoint  # type: ignore

    model = UNetJoint(num_classes=4)
    if ckpt and Path(ckpt).is_file():
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(ckpt, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    n_all, n_train = count_params(model)
    img = torch.randn(1, 3, size, size, device=device)

    def fwd():
        return model(img)

    with torch.no_grad():
        fwd()
    mem_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fwd()
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    ms, fps = bench_fps(fwd, device=device)
    return {
        "method": "UNet-Joint",
        "params_M": round(n_all / 1e6, 3),
        "trainable_M": round(n_train / 1e6, 3),
        "infer_ms": round(ms, 2),
        "FPS": round(fps, 2),
        "gpu_mem_MB": round(mem_mb, 1) if mem_mb is not None else "",
        "input": f"1x3x{size}x{size}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="configs/RIM-ONE-Cup.yaml")
    ap.add_argument("--unet-ckpt", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="paper_table_complexity.csv")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-unet", action="store_true")
    args = ap.parse_args()

    rows = []
    if not args.skip_ours:
        print(">>> measure Ours...")
        rows.append(measure_ours(args.config_file, args.device))
        print(rows[-1])
    if not args.skip_unet:
        print(">>> measure UNet...")
        try:
            rows.append(measure_unet(args.unet_ckpt or None, args.device))
            print(rows[-1])
        except Exception as e:
            print(f"[跳过 UNet] {e}")

    keys = ["method", "params_M", "trainable_M", "infer_ms", "FPS", "gpu_mem_MB", "input"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
