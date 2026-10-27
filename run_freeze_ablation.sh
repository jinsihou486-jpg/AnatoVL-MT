#!/usr/bin/env bash
# Freeze 消融：adapter / partial / full
# 在 MedCLIPSeg 根目录：bash run_freeze_ablation.sh
set -eu
if [[ ! -f train_joint.py ]]; then
  echo "请在 MedCLIPSeg 根目录执行"; exit 1
fi
SEEDS_CSV="${SEEDS:-298}"
MODES_CSV="${MODES:-adapter,partial,full}"
OUT_ROOT="${OUT_ROOT:-./outputs_freeze}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
if [[ -n "${CONFIGS:-}" ]]; then
  # shellcheck disable=SC2206
  CONFIG_ARR=($CONFIGS)
else
  CONFIG_ARR=(configs/RIM-ONE-Cup.yaml configs/RIM-ONE-Disc.yaml)
fi
IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS_CSV"
IFS=',' read -r -a MODES_ARR <<< "$MODES_CSV"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/joint_task"
mkdir -p "$OUT_ROOT"

for mode in "${MODES_ARR[@]}"; do
  mode=$(echo "$mode" | xargs)
  for seed in "${SEEDS_ARR[@]}"; do
    seed=$(echo "$seed" | xargs)
    for cfg in "${CONFIG_ARR[@]}"; do
      [[ -f "$cfg" ]] || continue
      name=$(basename "$cfg" .yaml)
      out_dir="${OUT_ROOT}/${mode}"
      best="${out_dir}/${name}/trained_models/seed${seed}/MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth"
      if [[ "$SKIP_EXISTING" == "1" && -f "$best" ]]; then
        echo "[跳过] $best"; continue
      fi
      echo ">>> freeze=$mode seed=$seed cfg=$cfg"
      python train_joint.py --config-file "$cfg" --output-dir "$out_dir" \
        --seed "$seed" --freeze-mode "$mode"
    done
  done
done

echo "=== eval ==="
for mode in "${MODES_ARR[@]}"; do
  mode=$(echo "$mode" | xargs)
  for cfg in "${CONFIG_ARR[@]}"; do
    [[ -f "$cfg" ]] || continue
    name=$(basename "$cfg" .yaml)
    python eval_joint_test.py --config-file "$cfg" \
      --output-dir "${OUT_ROOT}/${mode}" --seeds "$SEEDS_CSV" \
      --method "Freeze-${mode}" \
      --save-json "${OUT_ROOT}/eval_${mode}_${name}.json" || true
  done
done
echo "完成: ${OUT_ROOT}/eval_*.json"
