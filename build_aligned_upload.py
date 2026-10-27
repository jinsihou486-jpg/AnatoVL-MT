#!/usr/bin/env python3
"""Build an upload folder whose RIM-ONE cup and disc splits match per_image_class_labels.csv.

Original folders are left unchanged. Destination:
  medica/upload_joint/RIM-ONE/RIM-ONE-{Cup,Disc}/
  medica/upload_joint/Drishti-GS1/Drishti-GS1-{Cup,Disc}/   (copied only if IDs already match)
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MEDICA = Path(__file__).resolve().parents[1]
LABELS = MEDICA / "joint_task" / "per_image_class_labels.csv"
DEST = MEDICA / "upload_joint"
SERVER_ROOT = "/root/autodl-tmp/joint_data"


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def vertical_diameter(path: Path) -> int:
    ys = np.where(load_mask(path))[0]
    if len(ys) == 0:
        return 0
    return int(ys.max() - ys.min() + 1)


def index_files(organ_root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """stem -> split -> {'img': path, 'label': path}."""
    found: dict[str, dict[str, dict[str, Path]]] = {}
    for split in ("train", "val", "test"):
        folder = organ_root / f"{split.capitalize()}_Folder"
        for kind in ("img", "label"):
            directory = folder / kind
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if path.is_file():
                    found.setdefault(path.stem, {}).setdefault(split, {})[kind] = path
    return found


def grade_from_vcdr(value: float) -> int:
    if value <= 0.43:
        return 1
    if value <= 0.60:
        return 2
    if value <= 0.74:
        return 3
    return 4


def choose_mask_split(
    stem: str,
    cup_locs: dict[str, dict[str, Path]],
    disc_locs: dict[str, dict[str, Path]],
    stored_cdr: float,
    official_split: str,
) -> tuple[str, str]:
    scored: list[tuple[float, str, str]] = []
    for cup_split, cup_pair in cup_locs.items():
        for disc_split, disc_pair in disc_locs.items():
            if not all(kind in cup_pair and kind in disc_pair for kind in ("img", "label")):
                continue
            disc_v = vertical_diameter(disc_pair["label"])
            if disc_v == 0:
                continue
            cdr = vertical_diameter(cup_pair["label"]) / disc_v
            scored.append((abs(cdr - stored_cdr), cup_split, disc_split))
    if not scored:
        raise RuntimeError(f"no cup/disc mask pair for {stem}")
    official = [
        item for item in scored if item[1] == official_split and item[2] == official_split and item[0] <= 1e-3
    ]
    error, cup_split, disc_split = official[0] if official else min(scored)
    if error > 1e-3:
        raise RuntimeError(
            f"CDR mismatch for {stem}: best cup={cup_split} disc={disc_split} error {error}"
        )
    return cup_split, disc_split


def description_for(df: pd.DataFrame) -> str:
    counts = Counter(str(x) for x in df["Description"].tolist())
    return counts.most_common(1)[0][0]


def write_prompt(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Image", "Ground Truth", "Description", "Grade"]).to_excel(
        path, index=False
    )


def rebuild_organ(
    organ_name: str,
    source_root: Path,
    dest_root: Path,
    records: pd.DataFrame,
    mask_split_of: dict[str, tuple[str, str]],
    file_index: dict[str, dict[str, dict[str, Path]]],
    which: int,
    lines: list[str],
) -> None:
    if dest_root.exists():
        shutil.rmtree(dest_root)
    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for record in records.itertuples(index=False):
        stem = str(record.image_id)
        dest_split = str(record.split)
        source_split = mask_split_of[stem][which]
        pair = file_index[stem][source_split]
        for kind in ("img", "label"):
            source = pair[kind]
            target = dest_root / f"{dest_split.capitalize()}_Folder" / kind / f"{stem}{source.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        split_rows[dest_split].append(
            {
                "image_id": stem,
                "split": dest_split,
                "grade_4": int(record.grade_4),
                "grade_4_label": record.grade_4_label,
            }
        )

    labels = pd.DataFrame(
        [row for rows in split_rows.values() for row in rows],
        columns=["image_id", "split", "grade_4", "grade_4_label"],
    )
    labels.to_csv(dest_root / "class_labels.csv", index=False)
    for split, rows in split_rows.items():
        pd.DataFrame(rows).to_csv(
            dest_root / f"{split.capitalize()}_Folder" / "class_labels.csv", index=False
        )

    prompt_src = source_root / "Prompts_Folder"
    prompt_dst = dest_root / "Prompts_Folder"
    by_split = {
        split: {row["image_id"]: row for row in rows} for split, rows in split_rows.items()
    }
    for xlsx in sorted(prompt_src.glob("*.xlsx")):
        old = pd.read_excel(xlsx)
        if "Description" not in old.columns:
            lines.append(f"[skip prompt] {organ_name}/{xlsx.name}: no Description")
            continue
        text = description_for(old)
        name = xlsx.name
        if name.startswith("Train_text"):
            split = "train"
        elif name.startswith("Val_text"):
            split = "val"
        elif name.startswith("Test_text"):
            split = "test"
        else:
            lines.append(f"[skip prompt] {organ_name}/{name}")
            continue
        allowed = by_split[split]
        if name in {f"{split.capitalize()}_text.xlsx", "Test_text_original.xlsx"} or name.startswith("Test_text_"):
            chosen = [allowed[stem] for stem in sorted(allowed)]
        else:
            chosen = []
            seen = set()
            for image in old["Image"].tolist():
                stem = Path(str(image)).stem
                if stem in allowed and stem not in seen:
                    chosen.append(allowed[stem])
                    seen.add(stem)
        rows = [
            {
                "Image": f"{row['image_id']}.png",
                "Ground Truth": f"{row['image_id']}.png",
                "Description": text,
                "Grade": int(row["grade_4"]),
            }
            for row in chosen
        ]
        write_prompt(prompt_dst / name, rows)
        lines.append(f"[prompt] {organ_name}/{name}: {len(old)} -> {len(rows)}")

    # Training looks for Test_text.xlsx before Test_text_original.xlsx.
    original = prompt_dst / "Test_text_original.xlsx"
    primary = prompt_dst / "Test_text.xlsx"
    if original.exists() and not primary.exists():
        shutil.copy2(original, primary)
        lines.append(f"[prompt] {organ_name}/Test_text.xlsx copied from original")


def mirror_subset_prompts(lines: list[str]) -> None:
    """Cup and disc percentage files must list the same image ids."""
    cup_root = DEST / "RIM-ONE" / "RIM-ONE-Cup" / "Prompts_Folder"
    disc_root = DEST / "RIM-ONE" / "RIM-ONE-Disc" / "Prompts_Folder"
    grades = {
        str(row.image_id): int(row.grade_4)
        for row in pd.read_csv(DEST / "RIM-ONE" / "RIM-ONE-Cup" / "class_labels.csv").itertuples(index=False)
    }
    for name in (
        "Train_text_10.xlsx",
        "Train_text_25.xlsx",
        "Train_text_50.xlsx",
        "Val_text_10.xlsx",
        "Val_text_25.xlsx",
        "Val_text_50.xlsx",
    ):
        disc = pd.read_excel(disc_root / name)
        cup = pd.read_excel(cup_root / name) if (cup_root / name).exists() else None
        if cup is not None and len(cup) > 0 and "Description" in cup.columns:
            text = description_for(cup)
        else:
            parent = "Train_text.xlsx" if name.startswith("Train") else "Val_text.xlsx"
            text = description_for(pd.read_excel(cup_root / parent))
        rows = []
        for image in disc["Image"].tolist():
            stem = Path(str(image)).stem
            if stem not in grades:
                raise RuntimeError(f"{name} contains {stem}, which is outside the aligned split")
            rows.append(
                {
                    "Image": f"{stem}.png",
                    "Ground Truth": f"{stem}.png",
                    "Description": text,
                    "Grade": grades[stem],
                }
            )
        write_prompt(cup_root / name, rows)
        lines.append(f"[prompt] RIM-ONE-Cup/{name} aligned to disc ids: {len(rows)}")


def verify_rim(dest_cup: Path, dest_disc: Path, records: pd.DataFrame, lines: list[str]) -> None:
    expected = {"train": 249, "val": 36, "test": 37}
    cup_ids: dict[str, set[str]] = {}
    disc_ids: dict[str, set[str]] = {}
    for split, n_expected in expected.items():
        for organ, bucket in ((dest_cup, cup_ids), (dest_disc, disc_ids)):
            img = {p.stem for p in (organ / f"{split.capitalize()}_Folder" / "img").iterdir() if p.is_file()}
            lab = {p.stem for p in (organ / f"{split.capitalize()}_Folder" / "label").iterdir() if p.is_file()}
            bucket[split] = img
            if img != lab or len(img) != n_expected:
                raise RuntimeError(f"{organ.name} {split}: img {len(img)} label {len(lab)} expected {n_expected}")
        if cup_ids[split] != disc_ids[split]:
            raise RuntimeError(f"cup/disc id mismatch in {split}")
        official = set(records.loc[records["split"] == split, "image_id"].astype(str))
        if cup_ids[split] != official:
            raise RuntimeError(f"{split} ids differ from the label file")
    grades = records.loc[records["split"] == "test", "grade_4"].astype(int)
    counts = grades.value_counts().sort_index().to_dict()
    if counts != {1: 21, 2: 10, 3: 3, 4: 3}:
        raise RuntimeError(f"RIM test grades {counts}")
    lines.append(f"[verify] RIM cup/disc 249/36/37, test grades {counts}")


def copy_drishti(records: pd.DataFrame, lines: list[str]) -> None:
    for organ in ("Drishti-GS1-Cup", "Drishti-GS1-Disc"):
        source = MEDICA / "Drishti-GS1" / organ
        dest = DEST / "Drishti-GS1" / organ
        if dest.exists():
            shutil.rmtree(dest)
        official = records.loc[records["dataset"] == "Drishti-GS1", ["image_id", "split", "grade_4"]]
        expected = {"train": 71, "val": 10, "test": 20}
        for split, n_expected in expected.items():
            stems = {p.stem for p in (source / f"{split.capitalize()}_Folder" / "img").iterdir() if p.is_file()}
            wanted = set(official.loc[official["split"] == split, "image_id"].astype(str))
            if stems != wanted or len(stems) != n_expected:
                raise RuntimeError(f"{organ} {split} does not match the label file")
        shutil.copytree(source, dest)
        grade_map = {
            str(row.image_id): int(row.grade_4)
            for row in official.itertuples(index=False)
        }
        for xlsx in (dest / "Prompts_Folder").glob("*.xlsx"):
            frame = pd.read_excel(xlsx)
            if "Image" not in frame.columns:
                continue
            frame["Grade"] = [grade_map.get(Path(str(name)).stem, -1) for name in frame["Image"]]
            if (frame["Grade"] < 1).any():
                raise RuntimeError(f"missing grade in {organ}/{xlsx.name}")
            frame.to_excel(xlsx, index=False)
        lines.append(f"[copy] {organ} already matched 71/10/20")


def write_configs() -> None:
    config_dir = DEST / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        MEDICA / "RIM-ONE" / "configs" / "RIM-ONE-Cup.yaml": "RIM-ONE/RIM-ONE-Cup",
        MEDICA / "RIM-ONE" / "configs" / "RIM-ONE-Disc.yaml": "RIM-ONE/RIM-ONE-Disc",
        MEDICA / "Drishti-GS1" / "configs" / "Drishti-GS1-Cup.yaml": "Drishti-GS1/Drishti-GS1-Cup",
        MEDICA / "Drishti-GS1" / "configs" / "Drishti-GS1-Disc.yaml": "Drishti-GS1/Drishti-GS1-Disc",
    }
    for source, relative in mapping.items():
        text = source.read_text(encoding="utf-8")
        text = text.replace("./data/" + relative + "/", f"{SERVER_ROOT}/{relative}/")
        (config_dir / source.name).write_text(text, encoding="utf-8")


def main() -> None:
    labels = pd.read_csv(LABELS)
    rim = labels.loc[labels["dataset"] == "RIM-ONE"].copy()
    if len(rim) != 322:
        raise RuntimeError(f"expected 322 RIM rows, got {len(rim)}")
    rim["grade_check"] = rim["vertical_cdr"].map(grade_from_vcdr)
    if not (rim["grade_check"] == rim["grade_4"].astype(int)).all():
        raise RuntimeError("grade_4 does not match the CDR cut-offs")

    cup_index = index_files(MEDICA / "RIM-ONE" / "RIM-ONE-Cup")
    disc_index = index_files(MEDICA / "RIM-ONE" / "RIM-ONE-Disc")
    mask_split_of: dict[str, tuple[str, str]] = {}
    moved: list[str] = []
    for record in rim.itertuples(index=False):
        stem = str(record.image_id)
        if stem not in cup_index or stem not in disc_index:
            raise RuntimeError(f"missing files for {stem}")
        cup_split, disc_split = choose_mask_split(
            stem,
            cup_index[stem],
            disc_index[stem],
            float(record.vertical_cdr),
            str(record.split),
        )
        mask_split_of[stem] = (cup_split, disc_split)
        if cup_split != str(record.split) or disc_split != str(record.split):
            moved.append(
                f"{stem}: cup mask from {cup_split}, disc mask from {disc_split}, assigned to {record.split}"
            )

    lines = ["RIM-ONE aligned to per_image_class_labels.csv", *moved]
    rebuild_organ(
        "RIM-ONE-Cup",
        MEDICA / "RIM-ONE" / "RIM-ONE-Cup",
        DEST / "RIM-ONE" / "RIM-ONE-Cup",
        rim,
        mask_split_of,
        cup_index,
        0,
        lines,
    )
    rebuild_organ(
        "RIM-ONE-Disc",
        MEDICA / "RIM-ONE" / "RIM-ONE-Disc",
        DEST / "RIM-ONE" / "RIM-ONE-Disc",
        rim,
        mask_split_of,
        disc_index,
        1,
        lines,
    )
    mirror_subset_prompts(lines)
    verify_rim(DEST / "RIM-ONE" / "RIM-ONE-Cup", DEST / "RIM-ONE" / "RIM-ONE-Disc", rim, lines)
    copy_drishti(labels, lines)
    write_configs()
    report = DEST / "COUNTS.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {DEST}")


if __name__ == "__main__":
    main()
