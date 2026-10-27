"""
用 best.pth 在 TEST 集上评估联合模型：Dice / Acc / macro-F1 / Quadratic Kappa

放到 MedCLIPSeg 根目录后运行。

单 seed:
  python eval_joint_test.py --all --output-dir ./outputs --seed 298 \\
      --save-json ./joint_test_seed298.json

三种子一次评估:
  python eval_joint_test.py --all --output-dir ./outputs --seeds 298,1,42 \\
      --save-json ./joint_test_all_seeds.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text

# 允许从 medica/joint_task 导入 metrics_utils（拷到服务器时可一并放根目录或 PYTHONPATH）
_HERE = Path(__file__).resolve().parent
for cand in [_HERE, _HERE.parent, Path.cwd() / "joint_task", Path.cwd()]:
    if (cand / "metrics_utils.py").exists():
        sys.path.insert(0, str(cand))
        break

from metrics_utils import DEFAULT_SEEDS, classification_metrics  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/RIM-ONE-Cup.yaml",
    "configs/RIM-ONE-Disc.yaml",
    "configs/Drishti-GS1-Cup.yaml",
    "configs/Drishti-GS1-Disc.yaml",
]


def get_cfg(config_file: str, output_dir: str, seed: int, opts, prompt_ablate: str = ""):
    cfg = load_cfg_from_cfg_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.seed = seed
    cfg.output_dir = output_dir
    cfg.prompt_ablate = prompt_ablate or ""
    if not hasattr(cfg.MODEL, "NUM_CLASSES"):
        cfg.MODEL.NUM_CLASSES = 4
    if not hasattr(cfg.TRAIN, "CLS_WEIGHT"):
        cfg.TRAIN.CLS_WEIGHT = 0.5
    return cfg


def resolve_test_prompt(prompt_dir: str, prompt_ablate: str = "") -> str:
    if prompt_ablate:
        cand = os.path.join(prompt_dir, f"Test_text_ablate_{prompt_ablate}.xlsx")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"缺少测试消融 prompt: {cand}")
    for name in ("Test_text.xlsx", "Test_text_original.xlsx"):
        path = os.path.join(prompt_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"缺少测试 prompt，目录: {prompt_dir}")


def build_model(cfg):
    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        return build_medclipseg_unimedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "biomedclip":
        return build_medclipseg_biomedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "clip":
        return build_medclipseg_clip(cfg)
    if cfg.MODEL.CLIP_MODEL == "pubmedclip":
        return build_medclipseg_pubmedclip(cfg)
    raise ValueError(cfg.MODEL.CLIP_MODEL)


def find_best_ckpt(cfg) -> Path:
    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}_joint"
    save_dir = Path(cfg.output_dir) / cfg.DATASET.NAME / "trained_models" / f"seed{cfg.seed}"
    candidates = [
        save_dir / f"{results_name}_best.pth",
        save_dir / f"{results_name}_latest.pth",
    ]
    if save_dir.exists():
        candidates.extend(sorted(save_dir.glob("*joint*_best.pth")))
        candidates.extend(sorted(save_dir.glob("*_best.pth")))
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"找不到 checkpoint，已查: {save_dir}")


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    dice_scores = []
    ys, preds = [], []

    for batch in tqdm(loader, desc="Test"):
        images = batch["image"].to(device)
        masks = batch["ground_truth_mask"].to(device)
        text = batch["text_prompt"]
        labels = batch["class_label"].to(device)

        out = model(images, text=text, num_samples=1)
        if isinstance(out, tuple) and len(out) == 2:
            seg_samples, class_logits = out
        else:
            raise RuntimeError("模型未返回 (seg_samples, class_logits)，请确认已替换带 cls_head 的 unimedclip")

        logits = seg_samples[0]
        probs = torch.sigmoid(logits)
        pred_masks = (probs > 0.5).float()
        if pred_masks.ndim == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        intersection = (pred_masks * masks).sum(dim=(1, 2, 3))
        union = pred_masks.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice_scores.extend(dice.detach().cpu().numpy().tolist())

        valid = labels >= 0
        if valid.any():
            ys.extend(labels[valid].cpu().tolist())
            preds.extend(class_logits[valid].argmax(1).cpu().tolist())

    avg_dice = float(mean(dice_scores)) if dice_scores else 0.0
    cls = classification_metrics(ys, preds)
    return {
        "n_images": len(dice_scores),
        "Dice": round(avg_dice, 4),
        **{k: cls[k] for k in ("ClsAcc", "macro_F1", "Kappa_QWK", "n_cls", "report")},
    }


def eval_one(
    config_file: str,
    output_dir: str,
    seed: int,
    opts,
    ckpt: str | None = None,
    method: str = "MedCLIPSeg-Joint",
    prompt_ablate: str = "",
):
    cfg = get_cfg(config_file, output_dir, seed, opts, prompt_ablate=prompt_ablate)
    device = cfg.MODEL.DEVICE

    prompt_dir = cfg.DATASET.TEXT_PROMPT_PATH
    test_text_path = resolve_test_prompt(prompt_dir, prompt_ablate)
    print(f"[prompt] ablate={prompt_ablate or 'default'} -> {test_text_path}")
    test_text = read_text(test_text_path)

    tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    ds = DatasetSegmentation(
        cfg.DATASET.TEST_PATH, cfg.DATASET.NAME, test_text, tf, image_size=cfg.DATASET.SIZE
    )
    loader = DataLoader(ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(cfg)
    ckpt_path = Path(ckpt) if ckpt else find_best_ckpt(cfg)
    # PyTorch>=2.6 默认 weights_only=True，会无法反序列化含 numpy scalar 的旧 checkpoint
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
        meta = {
            "ckpt_epoch": int(state["epoch"]) if state.get("epoch") is not None else None,
            "ckpt_best_dice_val": float(state["best_dice"]) if state.get("best_dice") is not None else None,
            "ckpt_best_cls_acc_val": float(state["best_cls_acc"]) if state.get("best_cls_acc") is not None else None,
            "ckpt_best_score_val": float(state["best_score"]) if state.get("best_score") is not None else None,
        }
    else:
        model.load_state_dict(state, strict=False)
        meta = {}

    model.to(device)
    metrics = evaluate_test(model, loader, device)
    name = cfg.DATASET.NAME
    organ = "Cup" if "Cup" in name else ("Disc" if "Disc" in name else "?")
    dataset = "RIM-ONE" if "RIM" in name else ("Drishti-GS1" if "Drishti" in name else name)

    result = {
        "method": method,
        "dataset": dataset,
        "organ": organ,
        "seed": seed,
        "prompt_ablate": prompt_ablate or "default",
        "config": config_file,
        "ckpt": str(ckpt_path),
        **meta,
        **metrics,
    }
    print("=" * 60)
    print(f"{method} | {dataset} | {organ} | seed={seed}")
    print(f"ckpt: {ckpt_path}")
    print(
        f"n={result['n_images']}  Dice={result['Dice']:.4f}  "
        f"Acc={result['ClsAcc']:.4f}  F1={result['macro_F1']:.4f}  Kappa={result['Kappa_QWK']:.4f}"
    )
    print(result["report"])
    return result


def to_jsonable(obj):
    """把 numpy / torch 标量转成 JSON 可写的 Python 类型。"""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return obj


def parse_seeds(args) -> list[int]:
    if getattr(args, "seeds", ""):
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return [int(args.seed)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", type=str, default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=str, default="", help="逗号分隔，如 298,1,42；优先于 --seed")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--method", type=str, default="MedCLIPSeg-Joint")
    p.add_argument(
        "--prompt-ablate",
        type=str,
        default="",
        choices=["", "correct", "empty", "random", "swap"],
        help="评测时使用 Test_text_ablate_{mode}.xlsx（须与训练一致）",
    )
    p.add_argument("--save-json", type=str, default="joint_test_metrics.json")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = p.parse_args()

    configs = DEFAULT_CONFIGS if args.all else [args.config_file]
    if not args.all and not args.config_file:
        raise SystemExit("请指定 --config-file 或使用 --all")

    seeds = parse_seeds(args)
    rows = []
    for seed in seeds:
        for cfg_file in configs:
            if not os.path.isfile(cfg_file):
                print(f"[跳过] 无配置文件: {cfg_file}")
                continue
            try:
                rows.append(
                    eval_one(
                        cfg_file,
                        args.output_dir,
                        seed,
                        args.opts,
                        ckpt=args.ckpt or None,
                        method=args.method,
                        prompt_ablate=args.prompt_ablate,
                    )
                )
            except Exception as e:
                print(f"[失败] seed={seed} {cfg_file}: {e}")
                rows.append({"method": args.method, "seed": seed, "config": cfg_file, "error": str(e)})

    print("\n======= TEST 明细 ======")
    print("method,dataset,organ,seed,Dice,ClsAcc,macro_F1,Kappa_QWK")
    for r in rows:
        if "error" in r:
            print(f"{r.get('config')},ERROR,{r['error']}")
        else:
            print(
                f"{r['method']},{r['dataset']},{r['organ']},{r['seed']},"
                f"{r['Dice']},{r['ClsAcc']},{r['macro_F1']},{r['Kappa_QWK']}"
            )

    out = Path(args.save_json)
    out.write_text(
        json.dumps(to_jsonable(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n已写入 {out.resolve()}")

    # 若多 seed，顺带写 mean±std
    try:
        from metrics_utils import save_mean_std_table

        if len(seeds) > 1:
            table = save_mean_std_table(
                rows,
                out.with_name(out.stem + "_meanstd.csv"),
                out.with_name(out.stem + "_meanstd.md"),
            )
            if not table.empty:
                print("\n======= Mean ± Std ======")
                print(table[["dataset", "organ", "Dice", "ClsAcc", "macro_F1", "Kappa_QWK"]].to_string(index=False))
    except Exception as e:
        print(f"[warn] mean±std 汇总失败: {e}")


if __name__ == "__main__":
    main()
