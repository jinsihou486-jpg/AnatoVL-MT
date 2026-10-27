#!/usr/bin/env bash
# MedCLIPSeg Joint：三种子训练 + Test 评估
# 在 MedCLIPSeg 根目录执行。已有 seed=1 可跳过该 seed 训练。
set -eu

SEEDS=(298 1 42)
CONFIGS=(
  configs/RIM-ONE-Cup.yaml
  configs/RIM-ONE-Disc.yaml
  configs/Drishti-GS1-Cup.yaml
  configs/Drishti-GS1-Disc.yaml
)
OUT_DIR=${OUT_DIR:-./outputs}
SKIP_EXISTING=${SKIP_EXISTING:-1}

echo "=== 1) 多 seed 联合训练 ==="
for seed in "${SEEDS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    best="$OUT_DIR/$name/trained_models/seed${seed}/MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth"
    if [[ "$SKIP_EXISTING" == "1" && -f "$best" ]]; then
      echo "[跳过已有] $best"
      continue
    fi
    echo ">>> train $cfg seed=$seed"
    python train_joint.py --config-file "$cfg" --output-dir "$OUT_DIR" --seed "$seed"
  done
done

echo "=== 2) Test 评估（Dice/Acc/F1/Kappa）==="
# 确保 metrics_utils.py 在 PYTHONPATH（与 train_joint 同目录或 joint_task）
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/joint_task"

python eval_joint_test.py --all --output-dir "$OUT_DIR" --seeds 298,1,42 \
  --save-json ./joint_test_all_seeds.json \
  --method MedCLIPSeg-Joint

echo "完成。主表看 joint_test_all_seeds_meanstd.csv"
