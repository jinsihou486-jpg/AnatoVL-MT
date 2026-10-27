# -*- coding: utf-8 -*-
"""
覆盖到 MedCLIPSeg/trainers/medclipseg_unimedclip.py 的 build 函数冻结逻辑，
或在 train 时通过环境变量/配置控制。

更稳妥：本文件作为补丁说明 + train_joint 增加 --freeze-mode。

freeze_mode:
  adapter  = 默认（只训 PVL/mask/upscale/cls）← 正文设定
  partial  = adapter + 最后若干视觉 block（可选简化：adapter + ln_post/proj）
  full     = 全参数微调
"""

FREEZE_MODES = ("adapter", "partial", "full")


def apply_freeze(model, mode: str = "adapter"):
    mode = (mode or "adapter").lower()
    if mode not in FREEZE_MODES:
        raise ValueError(f"freeze_mode must be one of {FREEZE_MODES}")

    for _, p in model.named_parameters():
        p.requires_grad_(False)

    if mode == "full":
        for _, p in model.named_parameters():
            p.requires_grad_(True)
        return model

    # adapter / partial: always train heads + adapters
    for name, p in model.named_parameters():
        if any(k in name for k in ("pvl_adapters", "mask_head", "upscale", "cls_head")):
            p.requires_grad_(True)

    if mode == "partial":
        # 解冻视觉塔后半段 resblocks（约后 1/3）+ ln_post/proj
        vis = getattr(model, "vision_model", None)
        if vis is not None and hasattr(vis, "transformer"):
            blocks = list(vis.transformer.resblocks)
            n = max(1, len(blocks) // 3)
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        for name, p in model.named_parameters():
            if any(k in name for k in ("ln_post", "visual.proj", "vision_model.proj")):
                p.requires_grad_(True)

    return model
