"""
联合训练脚本：分割 + 四分类
用法与官方 train.py 相同，额外使用 TRAIN.CLS_WEIGHT / MODEL.NUM_CLASSES

覆盖用法：把本文件放到 MedCLIPSeg 根目录后运行:
  python train_joint.py --config-file configs/xxx.yaml --output-dir ./outputs

或在官方 train.py 同目录直接跑（需已替换 dataloader / unimedclip）。
"""

import os
import argparse
import random
import logging
from statistics import mean

import numpy as np
import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers import *
from utils.main_utils import load_cfg_from_cfg_file, read_text


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_prompt_file(prompt_dir: str, split: str, data_percentage: int, prompt_ablate: str) -> str:
    """
    split: Train / Val / Test
    prompt_ablate: '' | correct | empty | random | swap
    """
    if data_percentage != 100 and split in ("Train", "Val"):
        base = f"{split}_text_{data_percentage}"
    else:
        base = f"{split}_text"
    if prompt_ablate:
        cand = os.path.join(prompt_dir, f"{base}_ablate_{prompt_ablate}.xlsx")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(
            f"缺少消融 prompt: {cand}。请先运行 make_prompt_ablation.py 生成。"
        )
    # 默认：Train_text.xlsx / Val_text.xlsx；Test 兼容 original
    primary = os.path.join(prompt_dir, f"{base}.xlsx")
    if os.path.isfile(primary):
        return primary
    if split == "Test":
        alt = os.path.join(prompt_dir, "Test_text_original.xlsx")
        if os.path.isfile(alt):
            return alt
    raise FileNotFoundError(f"缺少 prompt: {primary}")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--prompt-ablate",
        type=str,
        default="",
        choices=["", "correct", "empty", "random", "swap"],
        help="提示词消融：使用 Train/Val_text_ablate_{mode}.xlsx；空=原文 prompt",
    )
    parser.add_argument(
        "--freeze-mode",
        type=str,
        default="adapter",
        choices=["adapter", "partial", "full"],
        help="冻结策略消融：adapter=只训适配器/头；partial=再解冻后部视觉块；full=全参微调",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    # 默认分类配置
    if not hasattr(cfg.MODEL, "NUM_CLASSES"):
        cfg.MODEL.NUM_CLASSES = 4
    if not hasattr(cfg.TRAIN, "CLS_WEIGHT"):
        cfg.TRAIN.CLS_WEIGHT = 0.5
    if not hasattr(cfg, "prompt_ablate"):
        cfg.prompt_ablate = ""
    if not hasattr(cfg, "freeze_mode"):
        cfg.freeze_mode = "adapter"
    return cfg


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def calc_seg_loss(low_res_logits, low_res_label_batch, ce_loss, dice_loss, cfg):
    loss_ce = ce_loss(low_res_logits, low_res_label_batch.float())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch)
    return cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce


def calc_cls_loss(class_logits, class_labels, cls_crit):
    # 过滤无效标签 (-1)
    valid = class_labels >= 0
    if valid.sum() == 0:
        return class_logits.new_zeros(())
    return cls_crit(class_logits[valid], class_labels[valid])


@torch.no_grad()
def evaluate(model, val_dataloader, device, ce_loss, dice_loss, cls_crit, cfg):
    model.eval()
    val_losses, dice_scores = [], []
    ys, preds = [], []

    for batch in tqdm(val_dataloader, desc="Validation"):
        images = batch["image"].to(device)
        masks = batch["ground_truth_mask"].to(device)
        text = batch["text_prompt"]
        labels = batch["class_label"].to(device)

        seg_samples, class_logits = model(images, text=text, num_samples=1)
        logits = seg_samples[0]
        loss = calc_seg_loss(logits, masks, ce_loss, dice_loss, cfg)
        loss = loss + cfg.TRAIN.CLS_WEIGHT * calc_cls_loss(class_logits, labels, cls_crit)
        val_losses.append(loss.item())

        probs = torch.sigmoid(logits)
        pred_masks = (probs > 0.5).float()
        if pred_masks.ndim == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        intersection = (pred_masks * masks).sum(dim=(1, 2, 3))
        union = pred_masks.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice_scores.extend(dice.cpu().numpy())

        valid = labels >= 0
        if valid.any():
            ys.extend(labels[valid].cpu().tolist())
            preds.extend(class_logits[valid].argmax(1).cpu().tolist())

    avg_loss = mean(val_losses) if val_losses else 0.0
    avg_dice = mean(dice_scores) if dice_scores else 0.0
    if ys:
        acc = float(np.mean(np.array(ys) == np.array(preds)))
    else:
        acc = 0.0
    model.train()
    return avg_loss, avg_dice, acc


cfg = get_arguments()
cfg.DATASET.NAME = (
    cfg.DATASET.NAME + f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
)
os.makedirs(
    os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}"),
    exist_ok=True,
)

logger = logger_config(
    os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log_joint.txt")
)
logger.info(cfg)
if cfg.seed >= 0:
    set_random_seed(cfg.seed)


def worker_init_fn(worker_id):
    seed = cfg.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


ce_loss = BCEWithLogitsLoss()
dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")
cls_crit = CrossEntropyLoss()

train_tf = transforms.Compose([RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])])
val_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])

prompt_ablate = getattr(cfg, "prompt_ablate", "") or ""
train_text_path = resolve_prompt_file(
    cfg.DATASET.TEXT_PROMPT_PATH, "Train", cfg.data_percentage, prompt_ablate
)
val_text_path = resolve_prompt_file(
    cfg.DATASET.TEXT_PROMPT_PATH, "Val", cfg.data_percentage, prompt_ablate
)
logger.info(f"Prompt ablate={prompt_ablate or 'default'} | train={train_text_path} | val={val_text_path}")
train_text = read_text(train_text_path)
val_text = read_text(val_text_path)

train_dataset = DatasetSegmentation(
    cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text, train_tf, image_size=cfg.DATASET.SIZE
)
val_dataset = DatasetSegmentation(
    cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, val_text, val_tf, image_size=cfg.DATASET.SIZE
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    num_workers=8,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    shuffle=False,
    worker_init_fn=worker_init_fn,
    num_workers=8,
    pin_memory=True,
)

if cfg.MODEL.CLIP_MODEL == "unimedclip":
    model = build_medclipseg_unimedclip(cfg)
elif cfg.MODEL.CLIP_MODEL == "biomedclip":
    model = build_medclipseg_biomedclip(cfg)
elif cfg.MODEL.CLIP_MODEL == "clip":
    model = build_medclipseg_clip(cfg)
elif cfg.MODEL.CLIP_MODEL == "pubmedclip":
    model = build_medclipseg_pubmedclip(cfg)
else:
    raise ValueError(cfg.MODEL.CLIP_MODEL)

# 冻结策略消融（覆盖 build_* 里的默认冻结）
try:
    from freeze_utils import apply_freeze
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from freeze_utils import apply_freeze
    except ImportError:
        apply_freeze = None

freeze_mode = getattr(cfg, "freeze_mode", "adapter") or "adapter"
if apply_freeze is not None:
    model = apply_freeze(model, freeze_mode)
    logger.info(f"Freeze mode = {freeze_mode}")
else:
    logger.info("freeze_utils not found; keep default adapter freeze from build_*")

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
logger.info(f"Trainable params: {n_train} / {n_all} ({100.0 * n_train / max(n_all,1):.2f}%)")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LEARNING_RATE)
num_epochs = cfg.TRAIN.NUM_EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}_joint"

save_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
os.makedirs(save_dir, exist_ok=True)
best_path = os.path.join(save_dir, f"{results_name}_best.pth")

start_epoch = 0
best_dice = 0.0
best_score = -1.0

if getattr(cfg, "resume", False):
    ckpt_path = best_path
    if not os.path.isfile(ckpt_path):
        alt = os.path.join(save_dir, f"{results_name}_latest.pth")
        if os.path.isfile(alt):
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"--resume 但找不到 ckpt: {best_path}")
    # PyTorch>=2.6 默认 weights_only=True，旧 ckpt 含 numpy scalar 会 UnpicklingError
    try:
        ckpt = torch.load(ckpt_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_dice = float(ckpt.get("best_dice", 0.0))
    best_score = float(ckpt.get("best_score", -1.0))
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        # 仅权重续训时，把 cosine 推到对应 epoch
        for _ in range(start_epoch):
            scheduler.step()
    logger.info(
        f"Resume from {ckpt_path} | start_epoch={start_epoch + 1}/{num_epochs} "
        f"| best_score={best_score:.4f}"
    )
    if start_epoch >= num_epochs:
        logger.info("Already finished all epochs; nothing to do.")
        raise SystemExit(0)

model.train()
model.to(cfg.MODEL.DEVICE)

def _free_bytes(path):
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except Exception:
        return None


def save_checkpoint(payload, path):
    """省盘保存：优先原子写；空间紧或失败则删旧后直写。"""
    tmp_path = path + ".tmp"
    if os.path.isfile(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    free = _free_bytes(os.path.dirname(path) or ".")
    # ~1.2GB 安全余量不够双倍写时，直接覆盖
    need_atomic = free is None or free > 3 * 1024 ** 3
    if need_atomic:
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            logger.warning(f"Atomic save failed ({e}); fallback overwrite.")
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
    if os.path.isfile(path):
        os.remove(path)
    torch.save(payload, path)


for epoch in range(start_epoch, num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
        seg_logits, clip_loss, class_logits = model(
            image=batch["image"].to(cfg.MODEL.DEVICE),
            text=batch["text_prompt"],
        )
        labels = batch["class_label"].to(cfg.MODEL.DEVICE)
        loss = calc_seg_loss(
            seg_logits, batch["ground_truth_mask"].to(cfg.MODEL.DEVICE), ce_loss, dice_loss, cfg
        )
        loss = loss + cfg.TRAIN.CLIP_WEIGHT * clip_loss
        loss = loss + cfg.TRAIN.CLS_WEIGHT * calc_cls_loss(class_logits, labels, cls_crit)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    scheduler.step()
    mean_epoch_loss = mean(epoch_losses)
    mean_val_loss, mean_val_dice, mean_val_acc = evaluate(
        model, val_dataloader, cfg.MODEL.DEVICE, ce_loss, dice_loss, cls_crit, cfg
    )
    score = 0.5 * mean_val_dice + 0.5 * mean_val_acc
    logger.info(
        f"EPOCH: {epoch+1} | TrainLoss: {mean_epoch_loss:.4f} | "
        f"ValLoss: {mean_val_loss:.4f} | Dice: {mean_val_dice:.4f} | ClsAcc: {mean_val_acc:.4f}"
    )

    # 默认只存 best，避免每个 epoch 写 latest 把磁盘写满（可用 SAVE_LATEST=1 打开）
    save_latest = os.environ.get("SAVE_LATEST", "0") == "1"
    # 续训需要 optimizer：设 CKPT_WEIGHTS_ONLY=0；默认仍只存权重省盘
    weights_only = os.environ.get("CKPT_WEIGHTS_ONLY", "1") == "1"

    if score > best_score:
        best_score = score
        best_dice = mean_val_dice
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "best_cls_acc": mean_val_acc,
            "best_score": best_score,
            "freeze_mode": getattr(cfg, "freeze_mode", "adapter"),
        }
        if not weights_only:
            payload["optimizer"] = optimizer.state_dict()
            payload["scheduler"] = scheduler.state_dict()
        try:
            save_checkpoint(payload, best_path)
            logger.info(f"Saved best joint model score={best_score:.4f} -> {best_path}")
        except Exception as e:
            logger.warning(
                f"Save best failed (disk full?): {e}. "
                "Training continues; free disk then --resume."
            )
            try:
                if os.path.isfile(best_path + ".tmp"):
                    os.remove(best_path + ".tmp")
            except OSError:
                pass

    if save_latest and ((epoch + 1) % 10 == 0 or epoch + 1 == num_epochs):
        latest_path = os.path.join(save_dir, f"{results_name}_latest.pth")
        try:
            save_checkpoint({"model": model.state_dict(), "epoch": epoch}, latest_path)
        except Exception as e:
            logger.warning(f"Save latest failed: {e}")
