"""
统一分类/分割指标：Accuracy、macro-F1、Quadratic Weighted Kappa、Dice。
供 Cls-only / Joint / 基线 / 多 seed 汇总共用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)

DEFAULT_SEEDS = (298, 1, 42)
CLASS_NAMES = ["G1", "G2", "G3", "G4"]
LABELS = [0, 1, 2, 3]


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] = LABELS,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {
            "ClsAcc": 0.0,
            "macro_F1": 0.0,
            "Kappa_QWK": 0.0,
            "n_cls": 0,
            "report": "no valid class labels",
        }
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    # 四级有序：二次加权 Kappa
    kappa = float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(labels))
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    return {
        "ClsAcc": round(acc, 4),
        "macro_F1": round(macro_f1, 4),
        "Kappa_QWK": round(kappa, 4),
        "n_cls": int(len(y_true)),
        "report": report,
    }


def batch_dice(
    pred_masks: np.ndarray | Any,
    gt_masks: np.ndarray | Any,
    eps: float = 1e-7,
) -> list[float]:
    """pred/gt: [B,1,H,W] or [B,H,W]，值为 0/1。"""
    pred = np.asarray(pred_masks, dtype=np.float64)
    gt = np.asarray(gt_masks, dtype=np.float64)
    if pred.ndim == 3:
        pred = pred[:, None]
    if gt.ndim == 3:
        gt = gt[:, None]
    axes = tuple(range(1, pred.ndim))
    inter = (pred * gt).sum(axis=axes)
    union = pred.sum(axis=axes) + gt.sum(axis=axes)
    dice = (2.0 * inter + eps) / (union + eps)
    return dice.tolist()


def mean_std_str(values: Iterable[float], decimals: int = 4) -> str:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return "—"
    if arr.size == 1:
        return f"{arr[0]:.{decimals}f}"
    return f"{arr.mean():.{decimals}f}±{arr.std(ddof=0):.{decimals}f}"


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_keys: Sequence[str] = ("method", "dataset", "organ"),
    metric_keys: Sequence[str] = ("Dice", "ClsAcc", "macro_F1", "Kappa_QWK"),
) -> pd.DataFrame:
    """把多 seed 明细聚合成 mean±std 表。"""
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        return pd.DataFrame()

    out_rows = []
    for keys, g in df.groupby(list(group_keys), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["n_seeds"] = int(len(g))
        if "seed" in g.columns:
            row["seeds"] = ",".join(str(int(s)) for s in sorted(g["seed"].unique()))
        for m in metric_keys:
            if m not in g.columns:
                row[m] = "—"
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                continue
            vals = g[m].astype(float)
            row[m] = mean_std_str(vals)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def load_metric_jsons(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.append(data)
    return rows


def save_mean_std_table(
    rows: list[dict[str, Any]],
    out_csv: str | Path,
    out_md: str | Path | None = None,
) -> pd.DataFrame:
    table = aggregate_rows(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if out_md is not None and not table.empty:
        cols = [c for c in ("method", "dataset", "organ", "Dice", "ClsAcc", "macro_F1", "Kappa_QWK", "n_seeds") if c in table.columns]
        Path(out_md).write_text(table[cols].to_markdown(index=False), encoding="utf-8")
    return table
