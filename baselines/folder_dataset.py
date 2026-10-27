"""
从 MedCLIPSeg 风格目录读取：Train/Val/Test_Folder + Prompts Grade / class_labels.csv
供 U-Net / CLIPSeg 联合基线共用（不依赖 MedCLIPSeg 包）。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _load_grade_map(folder: Path) -> dict[str, int]:
    grade_map: dict[str, int] = {}
    for cand in [folder / "class_labels.csv", folder.parent / "class_labels.csv"]:
        if cand.exists():
            df = pd.read_csv(cand)
            for r in df.itertuples(index=False):
                grade_map[Path(str(r.image_id)).stem] = int(r.grade_4)
            break
    return grade_map


def _read_prompt_xlsx(path: Path) -> list[dict]:
    df = pd.read_excel(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({k: r[k] for k in df.columns})
    return rows


def list_pairs(split_folder: Path, prompt_xlsx: Path | None) -> list[tuple[str, str, str, int]]:
    """返回 (img_name, mask_name, text, grade_1to4)。"""
    img_dir = split_folder / "img"
    label_dir = split_folder / "label"
    grade_map = _load_grade_map(split_folder)

    if prompt_xlsx is not None and prompt_xlsx.exists():
        pairs = []
        for row in _read_prompt_xlsx(prompt_xlsx):
            image = str(row["Image"])
            mask = str(row["Ground Truth"])
            desc = str(row.get("Description", "optic structure"))
            if "Grade" in row and pd.notna(row["Grade"]) and int(row["Grade"]) > 0:
                grade = int(row["Grade"])
            else:
                grade = grade_map.get(Path(image).stem, -1)
            pairs.append((image, mask, desc, grade))
        return sorted(pairs, key=lambda x: x[0])

    # fallback：直接扫目录
    pairs = []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}:
            continue
        mask_path = label_dir / img_path.name
        if not mask_path.exists():
            # 常见：mask 同 stem
            cands = list(label_dir.glob(img_path.stem + ".*"))
            if not cands:
                continue
            mask_path = cands[0]
        grade = grade_map.get(img_path.stem, -1)
        pairs.append((img_path.name, mask_path.name, "optic disc or cup", grade))
    return pairs


class FolderSegClsDataset(Dataset):
    def __init__(
        self,
        split_folder: str | Path,
        prompt_xlsx: str | Path | None,
        image_size: int = 224,
        train: bool = False,
        need_text: bool = False,
    ):
        self.split_folder = Path(split_folder)
        self.image_size = image_size
        self.train = train
        self.need_text = need_text
        self.pairs = list_pairs(
            self.split_folder,
            Path(prompt_xlsx) if prompt_xlsx else None,
        )
        if not self.pairs:
            raise FileNotFoundError(f"空数据集: {self.split_folder}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_name, mask_name, text, grade = self.pairs[idx]
        img_path = self.split_folder / "img" / image_name
        mask_path = self.split_folder / "label" / mask_name

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), 0)
        if mask is None:
            raise FileNotFoundError(mask_path)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask >= 127).astype(np.uint8)

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask * 255)

        if self.train and torch.rand(1).item() > 0.5:
            angle = float(torch.randint(-20, 21, (1,)).item())
            image_pil = TF.rotate(image_pil, angle)
            mask_pil = TF.rotate(mask_pil, angle)

        image_t = TF.to_tensor(image_pil)
        image_t = IMAGENET_NORM(image_t)
        mask_t = (TF.to_tensor(mask_pil) > 0.5).float()  # [1,H,W]

        class_label = int(grade) - 1 if int(grade) > 0 else -1
        out = {
            "image": image_t,
            "mask": mask_t,
            "class_label": class_label,
            "image_name": image_name,
        }
        if self.need_text:
            out["text"] = text
        return out


def resolve_task_paths(data_root: str | Path, dataset: str, organ: str) -> dict[str, Path]:
    """
    dataset: RIM-ONE | Drishti-GS1
    organ: Cup | Disc
    支持 data_root/RIM-ONE/... 或 data_root/data/RIM-ONE/...
    """
    root = Path(data_root)
    name = f"{dataset}-{organ}"
    candidates = [
        root / dataset / name,
        root / "data" / dataset / name,
        root / name,
    ]
    base = None
    for c in candidates:
        if (c / "Train_Folder").exists():
            base = c
            break
    if base is None:
        raise FileNotFoundError(f"找不到 {dataset}-{organ}，已查: {candidates}")

    prompts = base / "Prompts_Folder"
    return {
        "train": base / "Train_Folder",
        "val": base / "Val_Folder",
        "test": base / "Test_Folder",
        "train_xlsx": prompts / "Train_text.xlsx",
        "val_xlsx": prompts / "Val_text.xlsx",
        "test_xlsx": prompts / "Test_text.xlsx",
        "base": base,
    }
