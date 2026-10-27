#!/usr/bin/env python3
"""MedCLIPSeg Joint 多种子训练+评估（避免 bash CRLF 问题）。在 MedCLIPSeg 根目录运行。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SEEDS = [298, 1, 42]
CONFIGS = [
    "configs/RIM-ONE-Cup.yaml",
    "configs/RIM-ONE-Disc.yaml",
    "configs/Drishti-GS1-Cup.yaml",
    "configs/Drishti-GS1-Disc.yaml",
]
OUT_DIR = os.environ.get("OUT_DIR", "./outputs")
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1") == "1"


def main():
    root = Path.cwd()
    print("=== 1) 多 seed 联合训练 ===")
    for seed in SEEDS:
        for cfg in CONFIGS:
            name = Path(cfg).stem
            best = (
                Path(OUT_DIR)
                / name
                / "trained_models"
                / f"seed{seed}"
                / "MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth"
            )
            if SKIP_EXISTING and best.is_file():
                print(f"[跳过已有] {best}")
                continue
            print(f">>> train {cfg} seed={seed}")
            subprocess.check_call(
                [
                    sys.executable,
                    "train_joint.py",
                    "--config-file",
                    cfg,
                    "--output-dir",
                    OUT_DIR,
                    "--seed",
                    str(seed),
                ],
                cwd=str(root),
            )

    print("=== 2) Test 评估 ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root / 'joint_task'}:" + env.get("PYTHONPATH", "")
    subprocess.check_call(
        [
            sys.executable,
            "eval_joint_test.py",
            "--all",
            "--output-dir",
            OUT_DIR,
            "--seeds",
            "298,1,42",
            "--save-json",
            "./joint_test_all_seeds.json",
            "--method",
            "MedCLIPSeg-Joint",
        ],
        cwd=str(root),
        env=env,
    )
    print("完成。看 joint_test_all_seeds_meanstd.csv")


if __name__ == "__main__":
    main()
