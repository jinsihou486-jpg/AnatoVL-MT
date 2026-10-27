#!/usr/bin/env bash
# Loss 消融：seg_only / seg_clip / seg_cls / full
# 协议与 Freeze 一致：默认 RIM Cup+Disc、seed=298
# 在 MedCLIPSeg 根目录：
#   sed -i 's/\r$//' run_loss_ablation.sh
#   bash run_loss_ablation.sh
set -eu
if [[ ! -f train_joint.py ]]; then
  echo "请在 MedCLIPSeg 根目录执行"; exit 1
fi

SEEDS_CSV="${SEEDS:-298}"
# 可用 SETTINGS=seg_only,seg_clip,seg_cls,full 子集
SETTINGS_CSV="${SETTINGS:-seg_only,seg_clip,seg_cls,full}"
OUT_ROOT="${OUT_ROOT:-./outputs_loss}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ -n "${CONFIGS:-}" ]]; then
  # shellcheck disable=SC2206
  CONFIG_ARR=($CONFIGS)
else
  CONFIG_ARR=(configs/RIM-ONE-Cup.yaml configs/RIM-ONE-Disc.yaml)
fi

IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS_CSV"
IFS=',' read -r -a SETTINGS_ARR <<< "$SETTINGS_CSV"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/joint_task"
mkdir -p "$OUT_ROOT"

# 返回该 setting 的 CLIP_WEIGHT CLS_WEIGHT
weights_for() {
  case "$1" in
    seg_only) echo "0 0" ;;
    seg_clip) echo "0.1 0" ;;
    seg_cls)  echo "0 0.5" ;;
    full)     echo "0.1 0.5" ;;
    *) echo "未知 SETTINGS: $1" >&2; exit 1 ;;
  esac
}

for setting in "${SETTINGS_ARR[@]}"; do
  setting=$(echo "$setting" | xargs)
  read -r CLIP_W CLS_W <<< "$(weights_for "$setting")"
  for seed in "${SEEDS_ARR[@]}"; do
    seed=$(echo "$seed" | xargs)
    for cfg in "${CONFIG_ARR[@]}"; do
      [[ -f "$cfg" ]] || continue
      name=$(basename "$cfg" .yaml)
      out_dir="${OUT_ROOT}/${setting}"
      best="${out_dir}/${name}/trained_models/seed${seed}/MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth"
      if [[ "$SKIP_EXISTING" == "1" && -f "$best" ]]; then
        echo "[跳过] $best"; continue
      fi
      echo ">>> loss=$setting CLIP=$CLIP_W CLS=$CLS_W seed=$seed cfg=$cfg"
      python train_joint.py --config-file "$cfg" --output-dir "$out_dir" \
        --seed "$seed" \
        TRAIN.CLIP_WEIGHT "$CLIP_W" TRAIN.CLS_WEIGHT "$CLS_W"
    done
  done
done

echo "=== eval ==="
for setting in "${SETTINGS_ARR[@]}"; do
  setting=$(echo "$setting" | xargs)
  for cfg in "${CONFIG_ARR[@]}"; do
    [[ -f "$cfg" ]] || continue
    name=$(basename "$cfg" .yaml)
    python eval_joint_test.py --config-file "$cfg" \
      --output-dir "${OUT_ROOT}/${setting}" --seeds "$SEEDS_CSV" \
      --method "Loss-${setting}" \
      --save-json "${OUT_ROOT}/eval_${setting}_${name}.json" || true
  done
done
echo "完成: ${OUT_ROOT}/eval_*.json"

# 真正写出论文表（之前只 echo 了示例，不会生成 csv）
python - "$OUT_ROOT" <<'PY'
import json, glob, csv, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for p in sorted(root.glob("eval_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.extend(d if isinstance(d, list) else [d])
out = root / "paper_table_loss.csv"
keys = ["method", "dataset", "organ", "seed", "Dice", "ClsAcc", "macro_F1", "Kappa_QWK", "ckpt_epoch"]
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"wrote {out} ({len(rows)} rows)")
PY
