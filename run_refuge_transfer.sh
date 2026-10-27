#!/usr/bin/env bash
# =============================================================================
# REFUGE 外验（协议 A：RIM 权重 → REFUGE Test 直接评测）
# 可选协议 B：在 REFUGE 上微调后再测（DO_FINETUNE=1）
# -----------------------------------------------------------------------------
# 前置：
#   1) 把官方 REFUGE 解压到例如 data/REFUGE_raw/
#   2) 拷本脚本与 prepare_refuge_dataset.py 到 MedCLIPSeg 根目录
#   3) 保留 RIM 权重：outputs_loss/full 或 outputs/ 下 seed298 best
#
# 用法：
#   sed -i 's/\r$//' run_refuge_transfer.sh
#   nohup bash run_refuge_transfer.sh > run_refuge.log 2>&1 &
#   tail -f run_refuge.log
# =============================================================================
set -u
if [[ ! -f train_joint.py ]]; then
  echo "请在 MedCLIPSeg 根目录执行"; exit 1
fi
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/joint_task"
export SAVE_LATEST=0 CKPT_WEIGHTS_ONLY=1 PYTHONUNBUFFERED=1

SRC="${REFUGE_SRC:-data/REFUGE_raw}"
OUT_DATA="${REFUGE_OUT:-data/REFUGE}"
SEED="${SEED:-298}"
LOSS_OUT="${LOSS_OUT:-./outputs_loss/full}"
OURS_OUT="${OURS_OUT:-./outputs}"
RESULT_DIR="${RESULT_DIR:-./outputs_refuge}"
DO_PREPARE="${DO_PREPARE:-1}"
DO_FINETUNE="${DO_FINETUNE:-0}"   # 1=协议 B，耗时长
mkdir -p "$RESULT_DIR"

log() { echo ""; echo "======== $* ========"; }

PREP=prepare_refuge_dataset.py
[[ -f $PREP ]] || PREP=joint_task/prepare_refuge_dataset.py
[[ -f $PREP ]] || { echo "找不到 prepare_refuge_dataset.py"; exit 1; }

# ---------- 0) 准备数据 ----------
if [[ "$DO_PREPARE" == "1" ]]; then
  log "0) prepare REFUGE dataset"
  if [[ ! -d "$SRC" ]]; then
    echo "[错误] 找不到原始数据目录: $SRC"
    echo "请把 REFUGE 解压到该路径，或: REFUGE_SRC=/path/to/refuge bash run_refuge_transfer.sh"
    exit 1
  fi
  if [[ -f configs/REFUGE-Cup.yaml && -d "$OUT_DATA/REFUGE-Cup/Test_Folder/img" && "${FORCE_PREPARE:-0}" != "1" ]]; then
    echo "[跳过准备] 已存在 $OUT_DATA 与 configs/REFUGE-*.yaml （FORCE_PREPARE=1 可强制重建）"
  else
    python "$PREP" --src "$SRC" --out "$OUT_DATA" --cfg-dir configs
  fi
fi

if [[ ! -f configs/REFUGE-Cup.yaml ]]; then
  echo "[错误] 无 configs/REFUGE-Cup.yaml，准备步骤失败"; exit 1
fi

find_ckpt() {
  local organ="$1"
  local cand
  for cand in \
    "$LOSS_OUT/RIM-ONE-${organ}/trained_models/seed${SEED}/MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth" \
    "$OURS_OUT/RIM-ONE-${organ}/trained_models/seed${SEED}/MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth"
  do
    if [[ -f "$cand" ]]; then echo "$cand"; return 0; fi
  done
  return 1
}

# ---------- A) 迁移评测 ----------
log "A) Transfer eval RIM → REFUGE"
for organ in Cup Disc; do
  cfg="configs/REFUGE-${organ}.yaml"
  ckpt=$(find_ckpt "$organ" || true)
  if [[ -z "${ckpt:-}" ]]; then
    echo "[错误] 无 RIM-$organ 权重。请确认 outputs_loss/full 或 outputs/ 下 seed${SEED} best 还在。"
    echo "       若腾盘删了，需从本机重新上传，或先在 RIM 上重训 full。"
    continue
  fi
  echo ">>> $organ ckpt=$ckpt"
  python eval_joint_test.py --config-file "$cfg" \
    --output-dir "$LOSS_OUT" --seed "$SEED" \
    --ckpt "$ckpt" \
    --method "Transfer-RIM→REFUGE" \
    --save-json "$RESULT_DIR/eval_transfer_${organ}.json" \
    || echo "[警告] transfer $organ 失败"
done

# ---------- B) 可选微调 ----------
if [[ "$DO_FINETUNE" == "1" ]]; then
  log "B) Fine-tune on REFUGE (seed=$SEED)"
  FT_OUT="${FT_OUT:-./outputs_refuge_ft}"
  for organ in Cup Disc; do
    cfg="configs/REFUGE-${organ}.yaml"
    out_dir="$FT_OUT"
    best="$out_dir/REFUGE-${organ}/trained_models/seed${SEED}/MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth"
    if [[ -f "$best" ]]; then
      echo "[跳过训练] $best"
    else
      python train_joint.py --config-file "$cfg" --output-dir "$out_dir" --seed "$SEED" \
        --freeze-mode adapter \
        || echo "[警告] finetune $organ 失败"
    fi
    python eval_joint_test.py --config-file "$cfg" \
      --output-dir "$out_dir" --seed "$SEED" \
      --method "Finetune-REFUGE" \
      --save-json "$RESULT_DIR/eval_finetune_${organ}.json" \
      || true
  done
fi

# ---------- 汇总 CSV ----------
log "汇总 CSV"
python - "$RESULT_DIR" <<'PY'
import json, csv, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for p in sorted(root.glob("eval_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.extend(d if isinstance(d, list) else [d])
rows = [r for r in rows if "error" not in r]
out = root / "paper_table_refuge.csv"
keys = ["method","dataset","organ","seed","Dice","ClsAcc","macro_F1","Kappa_QWK","n_images"]
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("wrote", out)
for r in rows:
    print(f"{r.get('method')},{r.get('organ')},Dice={r.get('Dice')},Acc={r.get('ClsAcc')},QWK={r.get('Kappa_QWK')}")
PY

echo ""
echo "拉回本机: $RESULT_DIR/paper_table_refuge.csv 与 eval_*.json"
df -h . 2>/dev/null | tail -n 1 || true
