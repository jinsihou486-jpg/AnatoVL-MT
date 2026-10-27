#!/usr/bin/env python3
"""
把每张图的 grade_4 写入 MedCLIPSeg 的 prompt xlsx（新增 Grade 列）。

在 medica 目录下运行:
  python joint_task/medclipseg_cls_patch/inject_grades_into_prompts.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MEDICA = Path(__file__).resolve().parents[2]

TARGETS = [
    MEDICA / "Drishti-GS1" / "Drishti-GS1-Cup",
    MEDICA / "Drishti-GS1" / "Drishti-GS1-Disc",
    MEDICA / "RIM-ONE" / "RIM-ONE-Cup",
    MEDICA / "RIM-ONE" / "RIM-ONE-Disc",
]


def stem(name: str) -> str:
    return Path(str(name)).stem


def inject_one(dataset_root: Path) -> None:
    label_csv = dataset_root / "class_labels.csv"
    if not label_csv.exists():
        print(f"[skip] no class_labels.csv: {dataset_root}")
        return

    grade_map = {
        stem(r.image_id): int(r.grade_4)
        for r in pd.read_csv(label_csv).itertuples(index=False)
    }

    prompt_dir = dataset_root / "Prompts_Folder"
    xlsx_files = sorted(prompt_dir.glob("*.xlsx"))
    if not xlsx_files:
        print(f"[skip] no xlsx: {prompt_dir}")
        return

    for xlsx in xlsx_files:
        df = pd.read_excel(xlsx)
        if "Image" not in df.columns:
            print(f"[skip] no Image column: {xlsx.name}")
            continue
        grades = []
        missing = 0
        for img in df["Image"]:
            g = grade_map.get(stem(img))
            if g is None:
                missing += 1
                grades.append(-1)
            else:
                grades.append(g)
        df["Grade"] = grades
        df.to_excel(xlsx, index=False)
        print(f"[ok] {xlsx.relative_to(MEDICA)}  rows={len(df)} missing={missing}")


def main():
    for root in TARGETS:
        if root.exists():
            inject_one(root)
        else:
            print(f"[skip] missing {root}")
    print("Done. Prompt xlsx now have column: Grade (1~4).")


if __name__ == "__main__":
    main()
