#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DenseCLIP-style + 四级分类头（联合基线）

核心思想对齐 DenseCLIP：用 CLIP 视觉 patch 与文本嵌入做稠密相似度，再细化成掩膜。
在相同小样本联合协议下与 MedCLIPSeg-Ours 对比（非 Detectron2 全量官方栈复刻）。

  pip install open_clip_torch
  python baselines/train_denseclip_joint.py --data-root ./data --all-tasks --seeds 298,1,42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_utils import DEFAULT_SEEDS  # noqa: E402
from vl_joint_common import run_all  # noqa: E402


class DenseCLIPJoint(nn.Module):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai", num_classes: int = 4):
        super().__init__()
        import open_clip

        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        width = self.clip.visual.output_dim
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(width, num_classes),
        )
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def encode_text(self, texts, device):
        tokens = self.tokenizer(texts).to(device)
        return self.clip.encode_text(tokens)

    def encode_image_dense(self, images):
        # open_clip ViT forward features: get patch tokens
        visual = self.clip.visual
        x = images
        # Expect CLIP normalization; our loader uses ImageNet — close enough for baseline
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat(
            [
                visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        cls = x[:, 0, :]
        patches = x[:, 1:, :]
        if visual.proj is not None:
            cls = cls @ visual.proj
            patches = patches @ visual.proj
        return cls, patches

    def forward(self, images: torch.Tensor, texts: list[str]):
        device = images.device
        # resize to CLIP 224 if needed
        if images.shape[-1] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        cls, patches = self.encode_image_dense(images)
        text = self.encode_text(texts, device)
        cls = F.normalize(cls.float(), dim=-1)
        patches = F.normalize(patches.float(), dim=-1)
        text = F.normalize(text.float(), dim=-1)
        # dense sim [B, N]
        sim = torch.einsum("bnc,bc->bn", patches, text) * self.logit_scale.exp().clamp(max=100)
        h = int(sim.shape[1] ** 0.5)
        dense = sim.view(sim.shape[0], 1, h, h)
        dense = F.interpolate(dense, size=images.shape[-2:], mode="bilinear", align_corners=False)
        seg_logits = self.refine(dense)
        class_logits = self.cls_head(cls)
        return seg_logits, class_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", default="RIM-ONE", choices=["RIM-ONE", "Drishti-GS1"])
    p.add_argument("--organ", default="Cup", choices=["Cup", "Disc"])
    p.add_argument("--out-dir", default=str(ROOT / "baselines" / "runs_denseclip"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-cls", type=float, default=0.5)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--all-tasks", action="store_true")
    args = p.parse_args()

    def ctor():
        return DenseCLIPJoint()

    run_all(args, ctor, "DenseCLIP-style-Joint", "denseclip")


if __name__ == "__main__":
    main()
