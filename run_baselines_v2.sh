#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/baselines_v2
export PYTHONUNBUFFERED=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=/root/autodl-tmp/extra_py:/root/autodl-tmp/baselines_v2
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR" /root/autodl-tmp/outputs_baselines_v2
PY=/root/miniconda3/envs/medclipseg/bin/python

echo "ROOT $(pwd)"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())"

echo "===== U-Net ====="
"$PY" baselines/train_unet_joint.py \
  --data-root /root/autodl-tmp/joint_data \
  --all-tasks --seeds 298,1,42 --epochs 80 --batch-size 8 \
  --out-dir /root/autodl-tmp/outputs_baselines_v2/runs_unet \
  || echo "FAIL unet"

echo "===== CLIPSeg ====="
"$PY" baselines/train_clipseg_joint.py \
  --data-root /root/autodl-tmp/joint_data \
  --all-tasks --seeds 298,1,42 --epochs 40 --batch-size 4 --freeze-encoder \
  --out-dir /root/autodl-tmp/outputs_baselines_v2/runs_clipseg \
  || echo "FAIL clipseg"

echo "===== MaPLe ====="
"$PY" baselines/train_maple_joint.py \
  --data-root /root/autodl-tmp/joint_data \
  --all-tasks --seeds 298,1,42 --epochs 40 --batch-size 8 \
  --out-dir /root/autodl-tmp/outputs_baselines_v2/runs_maple \
  || echo "FAIL maple"

echo "===== summary ====="
"$PY" - << 'PY'
import json
from pathlib import Path
root = Path("/root/autodl-tmp/outputs_baselines_v2")
for name in ("runs_unet", "runs_clipseg", "runs_maple"):
    p = root / name / "all_metrics.json"
    print("FILE", p, p.exists())
    if not p.exists():
        continue
    rows = json.loads(p.read_text(encoding="utf-8"))
    print(name, "rows", len(rows))
    for r in rows:
        if "error" in r:
            print("ERROR", r)
            continue
        print(
            f"{r.get('method')} {r.get('dataset')} {r.get('organ')} seed={r.get('seed')} "
            f"n_images={r.get('n_images')} n_cls={r.get('n_cls')} "
            f"Dice={r.get('Dice')} Acc={r.get('ClsAcc')} QWK={r.get('Kappa_QWK')}"
        )
PY
echo "DONE"
