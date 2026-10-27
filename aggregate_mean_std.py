#!/usr/bin/env python3
"""
汇总多 seed / 多方法的 JSON 指标 → 论文用 Mean±Std 表。

用法:
  python aggregate_mean_std.py --inputs joint_test_all_seeds.json baselines/*/metrics_seed*.json \\
      --out paper_table_joint_meanstd.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics_utils import load_metric_jsons, save_mean_std_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="一个或多个 json（文件或 glob）")
    p.add_argument("--out", type=str, default="paper_table_joint_meanstd.csv")
    p.add_argument("--out-md", type=str, default="")
    args = p.parse_args()

    paths: list[Path] = []
    for item in args.inputs:
        pth = Path(item)
        if any(ch in item for ch in "*?[]"):
            paths.extend(sorted(Path().glob(item)))
        elif pth.is_dir():
            paths.extend(sorted(pth.rglob("*.json")))
        else:
            paths.append(pth)

    rows = load_metric_jsons(paths)
    # 展平：有的 json 是 list
    flat = []
    for r in rows:
        if isinstance(r, list):
            flat.extend(r)
        else:
            flat.append(r)

    out_md = args.out_md or str(Path(args.out).with_suffix(".md"))
    table = save_mean_std_table(flat, args.out, out_md)
    print(f"加载 {len(flat)} 条明细 → {args.out}")
    if table.empty:
        print("无有效行")
    else:
        print(table.to_string(index=False))
        # 额外写一份明细
        detail = Path(args.out).with_name(Path(args.out).stem + "_detail.json")
        detail.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"明细: {detail}")


if __name__ == "__main__":
    main()
