#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把官方 REFUGE 目录整理成 MedCLIPSeg 可用的 Cup/Disc 数据，并生成：
  - Train/Val/Test_Folder/{img,label}
  - Prompts_Folder/{Train,Val,Test}_text.xlsx（含 Grade）
  - class_labels.csv
  - configs/REFUGE-Cup.yaml / REFUGE-Disc.yaml

官方掩膜约定（BMP）：0=cup，128=disc，255=背景。
垂直 CDR 与正文相同阈值映射到 grade 1–4。

用法（MedCLIPSeg 根目录）：
  # 1) 下载并解压 REFUGE 到例如 data/REFUGE_raw/
  python prepare_refuge_dataset.py --src data/REFUGE_raw --out data/REFUGE
  # 2) 迁移评测
  bash run_refuge_transfer.sh
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

GRADE_LABEL = {
    1: "健康基线",
    2: "临界基线",
    3: "重度结构损伤",
    4: "极重度结构损伤",
}

CFG_TEMPLATE = """DATASET:
  NAME: "{name}"
  TRAIN_PATH: "{train}"
  VAL_PATH: "{val}"
  TEST_PATH: "{test}"
  TEXT_PROMPT_PATH: "{prompt}"
  SIZE: 224

TRAIN:
  BATCH_SIZE: 4
  NUM_EPOCHS: 100
  LEARNING_RATE: 0.0003
  DICE_WEIGHT: 0.5
  CE_WEIGHT: 0.5
  CLIP_WEIGHT: 0.1
  CLS_WEIGHT: 0.5

TEST:
  NUM_SAMPLES: 30
  USE_LATEST: True

MODEL:
  CLIP_MODEL: unimedclip
  BACKBONE: "ViT-B/16"
  ADAPTER_DIM: 256
  NUM_UPSCALE: 2
  BETA: 2.35
  GATE_INIT: 0.0
  LAYERS: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  TEMPERATURE: 0.2
  DEVICE: "cuda"
  NUM_CLASSES: 4
"""


def grade_from_vcdr(v: float) -> int:
    if v <= 0.43:
        return 1
    if v <= 0.60:
        return 2
    if v <= 0.74:
        return 3
    return 4


def mask_metrics(mask: np.ndarray) -> tuple[int, int, int]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, 0, 0
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1), int(mask.sum())


def find_first(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = list(root.glob(pat))
        if hits:
            return hits[0]
    return None


def discover_splits(src: Path) -> dict[str, dict[str, Path]]:
    """
    返回 {train|val|test: {images: Path, masks: Path}}
    兼容常见 REFUGE / REFUGE2 解压布局。
    """
    src = src.resolve()
    candidates = {
        "train": [
            ("**/Training400", "**/Annotation-Training400/**/Disc_Cup_Masks", "**/Annotation-Training400/Disc_Cup_Masks"),
            ("**/REFUGE-Training400/**/Training400", "**/Disc_Cup_Masks"),
            ("**/train/images", "**/train/mask"),
            ("**/Train/Images", "**/Train/Masks"),
        ],
        "val": [
            ("**/REFUGE-Validation400", "**/REFUGE-Validation400-GT/Disc_Cup_Masks"),
            ("**/Validation400", "**/Validation*/Disc_Cup_Masks"),
            ("**/val/images", "**/val/mask"),
            ("**/Val/Images", "**/Val/Masks"),
        ],
        "test": [
            ("**/REFUGE-Test400/Test400", "**/REFUGE-Test-GT/Disc_Cup_Masks"),
            ("**/Test400", "**/REFUGE-Test-GT/Disc_Cup_Masks"),
            ("**/test/images", "**/test/mask"),
            ("**/Test/Images", "**/Test/Masks"),
        ],
    }

    # 扁平搜索：任意含 Glaucoma/Non-Glaucoma 的训练图像根
    found: dict[str, dict[str, Path]] = {}

    # 更稳健：直接扫目录名
    def pick_images(split_key: str) -> Path | None:
        keys = {
            "train": ["Training400", "train", "Train"],
            "val": ["Validation400", "REFUGE-Validation400", "val", "Val"],
            "test": ["Test400", "test", "Test"],
        }[split_key]
        for k in keys:
            for p in src.rglob(k):
                if not p.is_dir():
                    continue
                # Training400 下可能是 Glaucoma/Non-Glaucoma
                jpgs = list(p.rglob("*.jpg")) + list(p.rglob("*.JPG")) + list(p.rglob("*.png"))
                if len(jpgs) >= 10:
                    return p
        return None

    def pick_masks(near: Path | None) -> Path | None:
        search_roots = [src]
        if near is not None:
            search_roots = [near, near.parent, near.parent.parent, src]
        for root in search_roots:
            for p in root.rglob("Disc_Cup_Masks"):
                if p.is_dir() and (list(p.glob("*.bmp")) or list(p.glob("*.BMP")) or list(p.glob("*.png"))):
                    return p
            for p in root.rglob("*mask*"):
                if p.is_dir() and (list(p.glob("*.bmp")) or list(p.glob("*.png"))):
                    # 避免误匹配
                    if "illustration" in p.name.lower():
                        continue
                    return p
        return None

    for split in ("train", "val", "test"):
        img_root = pick_images(split)
        mask_root = pick_masks(img_root)
        if img_root and mask_root:
            found[split] = {"images": img_root, "masks": mask_root}
            print(f"[发现] {split}: images={img_root} | masks={mask_root}")
        else:
            print(f"[缺] {split}: images={img_root} masks={mask_root}")

    if "test" not in found and "val" in found:
        print("[提示] 无 test，将用 val 作为 Test_Folder（协议 A 仍可跑）")
        found["test"] = found["val"]
    if "val" not in found and "train" in found:
        found["val"] = found["train"]
        print("[提示] 无 val，Val_Folder 暂指向 train")
    if "train" not in found and "test" in found:
        found["train"] = found["test"]
        print("[提示] 无 train，Train_Folder 暂指向 test（仅迁移评测够用）")

    if "test" not in found:
        raise FileNotFoundError(
            f"在 {src} 下找不到带 Disc_Cup_Masks 的 REFUGE 划分。\n"
            "请确认已解压 Training/Validation/Test 与对应 Annotation。"
        )
    return found


def index_files(root: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for ext in exts:
        for p in root.rglob(f"*{ext}"):
            if p.is_file():
                out[p.stem] = p
    return out


def load_combined_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def split_cup_disc(combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 uint8 0/255 的 cup、disc 二值图。disc=杯+盘。"""
    cup = (combined == 0).astype(np.uint8) * 255
    disc = ((combined == 0) | (combined == 128)).astype(np.uint8) * 255
    # 有的版本 cup 为暗、disc 为灰，容错：
    if cup.sum() == 0 and disc.sum() == 0:
        # 可能已是二值或其它编码：取非白为前景
        fg = (combined < 200).astype(np.uint8) * 255
        disc = fg
        cup = (combined < 64).astype(np.uint8) * 255
    return cup, disc


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_prompt_xlsx(path: Path, stems: list[str], organ: str, grade_map: dict[str, int]):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        pass
    desc = (
        "A central optic cup located in the center region of the color fundus image."
        if organ == "Cup"
        else "A central optic disc located in the center region of the color fundus image."
    )
    # 与 RIM 一致用短提示亦可；eval 用 Description 列
    short = "optic cup" if organ == "Cup" else "optic disc"
    rows = []
    for s in stems:
        rows.append(
            {
                "Image": f"{s}.png",
                "Ground Truth": f"{s}.png",
                "Description": short,
                "Grade": int(grade_map.get(s, -1)),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(path, index=False)


def convert(src: Path, out: Path, cfg_dir: Path):
    splits = discover_splits(src)
    out = out.resolve()
    ensure_dir(out)

    # 先收集全部 stem 的 cup/disc 与图像
    split_items: dict[str, list[str]] = {k: [] for k in ("train", "val", "test")}
    grade_map: dict[str, int] = {}
    vcdr_map: dict[str, float] = {}

    for organ in ("Cup", "Disc"):
        organ_root = out / f"REFUGE-{organ}"
        for split, folders in splits.items():
            folder_name = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}[split]
            img_dir = organ_root / folder_name / "img"
            lab_dir = organ_root / folder_name / "label"
            ensure_dir(img_dir)
            ensure_dir(lab_dir)

    # 建 mask/image 索引（各 split）
    for split, folders in splits.items():
        imgs = index_files(folders["images"], (".jpg", ".JPG", ".jpeg", ".png", ".PNG"))
        masks = index_files(folders["masks"], (".bmp", ".BMP", ".png", ".PNG"))
        common = sorted(set(imgs) & set(masks))
        print(f"[{split}] paired images={len(common)}")
        if not common:
            raise RuntimeError(f"{split} 图像与掩膜无法按文件名 stem 对齐")

        for stem in common:
            combined = load_combined_mask(masks[stem])
            cup_m, disc_m = split_cup_disc(combined)
            cv, _, _ = mask_metrics(cup_m > 0)
            dv, _, _ = mask_metrics(disc_m > 0)
            if dv <= 0:
                continue
            vcdr = float(cv / dv) if cv > 0 else 0.0
            g = grade_from_vcdr(vcdr)
            grade_map[stem] = g
            vcdr_map[stem] = vcdr

            # 读图并统一存 png
            im = cv2.imread(str(imgs[stem]))
            if im is None:
                im = np.array(Image.open(imgs[stem]).convert("RGB"))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            for organ, mask in (("Cup", cup_m), ("Disc", disc_m)):
                folder_name = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}[split]
                organ_root = out / f"REFUGE-{organ}"
                img_path = organ_root / folder_name / "img" / f"{stem}.png"
                lab_path = organ_root / folder_name / "label" / f"{stem}.png"
                cv2.imwrite(str(img_path), im)
                cv2.imwrite(str(lab_path), mask)
            split_items[split].append(stem)

    # class_labels + prompts + configs
    all_rows = []
    for split, stems in split_items.items():
        for s in stems:
            g = grade_map[s]
            all_rows.append(
                {
                    "image_id": s,
                    "split": split,
                    "grade_4": g,
                    "grade_4_label": GRADE_LABEL[g],
                    "vertical_cdr": round(vcdr_map[s], 6),
                }
            )
    labels_df = pd.DataFrame(all_rows).drop_duplicates("image_id")

    for organ in ("Cup", "Disc"):
        organ_root = out / f"REFUGE-{organ}"
        labels_df.to_csv(organ_root / "class_labels.csv", index=False)
        for split in ("train", "val", "test"):
            # 也写到各 Folder
            folder_name = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}[split]
            sub = labels_df[labels_df["split"] == split]
            sub.to_csv(organ_root / folder_name / "class_labels.csv", index=False)
            xname = {"train": "Train_text.xlsx", "val": "Val_text.xlsx", "test": "Test_text.xlsx"}[split]
            write_prompt_xlsx(
                organ_root / "Prompts_Folder" / xname,
                sorted(sub["image_id"].tolist()),
                organ,
                grade_map,
            )

        name = f"REFUGE-{organ}"
        cfg = CFG_TEMPLATE.format(
            name=name,
            train=str(organ_root / "Train_Folder").replace("\\", "/") + "/",
            val=str(organ_root / "Val_Folder").replace("\\", "/") + "/",
            test=str(organ_root / "Test_Folder").replace("\\", "/") + "/",
            prompt=str(organ_root / "Prompts_Folder").replace("\\", "/") + "/",
        )
        # 相对 MedCLIPSeg 根的路径更稳
        rel = Path("data") / "REFUGE" / name
        cfg = CFG_TEMPLATE.format(
            name=name,
            train=f"./{rel.as_posix()}/Train_Folder/",
            val=f"./{rel.as_posix()}/Val_Folder/",
            test=f"./{rel.as_posix()}/Test_Folder/",
            prompt=f"./{rel.as_posix()}/Prompts_Folder/",
        )
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / f"REFUGE-{organ}.yaml").write_text(cfg, encoding="utf-8")
        print(f"[ok] config -> {cfg_dir / f'REFUGE-{organ}.yaml'}")

    # 汇总
    labels_df.to_csv(out / "refuge_grades_summary.csv", index=False)
    print("\n=== Grade 分布 ===")
    print(labels_df.groupby(["split", "grade_4"]).size().unstack(fill_value=0))
    print(f"\n完成。输出目录: {out}")
    print("下一步: bash run_refuge_transfer.sh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="REFUGE 原始解压根目录")
    ap.add_argument("--out", default="data/REFUGE", help="整理后的 MedCLIPSeg 风格目录")
    ap.add_argument("--cfg-dir", default="configs")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"找不到 --src {src}")
    convert(src, Path(args.out), Path(args.cfg_dir))


if __name__ == "__main__":
    main()
