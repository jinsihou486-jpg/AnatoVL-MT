#!/usr/bin/env bash
# 提示词消融一键跑（务必在 MedCLIPSeg 根目录执行）
#
# 推荐先小规模试跑：
#   SEEDS=298 MODES=empty,random,swap \
#   CONFIGS="configs/RIM-ONE-Cup.yaml configs/RIM-ONE-Disc.yaml" \
#   bash joint_task/run_prompt_ablation.sh
#
# 正式四种子×四配置：
#   SEEDS=298,1,42 MODES=correct,empty,random,swap bash joint_task/run_prompt_ablation.sh
set -eu

if [[ ! -f train_joint.py ]]; then
  echo "请在 MedCLIPSeg 根目录执行本脚本（需存在 train_joint.py）"
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-./data}"
SEEDS_CSV="${SEEDS:-298}"
MODES_CSV="${MODES:-correct,empty,random,swap}"
OUT_ROOT="${OUT_ROOT:-./outputs_ablate}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ -n "${CONFIGS:-}" ]]; then
  # shellcheck disable=SC2206
  CONFIG_ARR=($CONFIGS)
else
  CONFIG_ARR=(
    configs/RIM-ONE-Cup.yaml
    configs/RIM-ONE-Disc.yaml
    configs/Drishti-GS1-Cup.yaml
    configs/Drishti-GS1-Disc.yaml
  )
fi

IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS_CSV"
IFS=',' read -r -a MODES_ARR <<< "$MODES_CSV"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/joint_task"
mkdir -p "$OUT_ROOT"

echo "=== 0) 生成 *_ablate_*.xlsx ==="
ABLATOR=""
if [[ -f joint_task/make_prompt_ablation.py ]]; then
  ABLATOR=joint_task/make_prompt_ablation.py
elif [[ -f make_prompt_ablation.py ]]; then
  ABLATOR=make_prompt_ablation.py
fi
if [[ -n "$ABLATOR" ]]; then
  python "$ABLATOR" --data-root "$DATA_ROOT" --only RIM-ONE,Drishti
else
  echo "[warn] 未找到 make_prompt_ablation.py，请确认已手工生成消融 prompt"
fi

echo "=== 1) 训练 ==="
for mode in "${MODES_ARR[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -z "$mode" ]] && continue
  for seed in "${SEEDS_ARR[@]}"; do
    seed="$(echo "$seed" | xargs)"
    for cfg in "${CONFIG_ARR[@]}"; do
      [[ -f "$cfg" ]] || { echo "[跳过] 无配置 $cfg"; continue; }
      name=$(basename "$cfg" .yaml)
      out_dir="${OUT_ROOT}/${mode}"
      best="${out_dir}/${name}/trained_models/seed${seed}/MedCLIPSeg_unimedclip_ViT-B-16_joint_best.pth"
      if [[ "$SKIP_EXISTING" == "1" && -f "$best" ]]; then
        echo "[跳过已有] $best"
        continue
      fi
      echo ">>> train mode=$mode seed=$seed cfg=$cfg"
      python train_joint.py \
        --config-file "$cfg" \
        --output-dir "$out_dir" \
        --seed "$seed" \
        --prompt-ablate "$mode"
    done
  done
done

echo "=== 2) Test 评测 ==="
MERGE_LIST=()
for mode in "${MODES_ARR[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -z "$mode" ]] && continue
  out_dir="${OUT_ROOT}/${mode}"
  save_json="${OUT_ROOT}/prompt_ablation_${mode}.json"
  echo ">>> eval mode=$mode"
  # 逐配置评测，避免 --all 遇到缺失配置中断
  : > "${save_json}.parts"
  first=1
  for cfg in "${CONFIG_ARR[@]}"; do
    [[ -f "$cfg" ]] || continue
    part="${OUT_ROOT}/_tmp_${mode}_$(basename "$cfg" .yaml).json"
    python eval_joint_test.py \
      --config-file "$cfg" \
      --output-dir "$out_dir" \
      --seeds "$SEEDS_CSV" \
      --prompt-ablate "$mode" \
      --method "Prompt-${mode}" \
      --save-json "$part" || echo "[warn] eval failed $mode $cfg"
    if [[ "$first" -eq 1 ]]; then
      cp "$part" "$save_json" 2>/dev/null || echo "[]" > "$save_json"
      first=0
    else
      python - "$save_json" "$part" <<'PY'
import json, sys
a, b = sys.argv[1], sys.argv[2]
ra = json.loads(open(a, encoding="utf-8").read())
rb = json.loads(open(b, encoding="utf-8").read())
if isinstance(ra, dict):
    ra = [ra]
if isinstance(rb, dict):
    rb = [rb]
open(a, "w", encoding="utf-8").write(json.dumps(ra + rb, indent=2, ensure_ascii=False))
PY
    fi
  done
  MERGE_LIST+=("$save_json")
done

python - "$OUT_ROOT/prompt_ablation_all.json" "${MERGE_LIST[@]}" <<'PY'
import json, sys
out = sys.argv[1]
rows = []
for p in sys.argv[2:]:
    try:
        data = json.loads(open(p, encoding="utf-8").read())
    except Exception:
        continue
    if isinstance(data, dict):
        data = [data]
    rows.extend(data)
open(out, "w", encoding="utf-8").write(json.dumps(rows, indent=2, ensure_ascii=False))
print("merged", len(rows), "rows ->", out)
PY

echo "=== 3) 汇总 ==="
AGG=""
if [[ -f joint_task/aggregate_ablation.py ]]; then
  AGG=joint_task/aggregate_ablation.py
elif [[ -f aggregate_ablation.py ]]; then
  AGG=aggregate_ablation.py
fi
if [[ -n "$AGG" ]]; then
  python "$AGG" \
    --input "${OUT_ROOT}/prompt_ablation_all.json" \
    --out "${OUT_ROOT}/paper_table_prompt_ablation.csv" \
    --group-key prompt_ablate
fi

echo "完成："
echo "  ${OUT_ROOT}/prompt_ablation_all.json"
echo "  ${OUT_ROOT}/paper_table_prompt_ablation.csv"
