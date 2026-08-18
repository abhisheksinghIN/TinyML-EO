#!/usr/bin/env python3
import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import utilsint as utils
from networks.soaint import get_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchinfo import summary

from brevitas.export import export_onnx_qcdq
from sklearn.metrics import jaccard_score, accuracy_score, cohen_kappa_score
import onnx

#Before Evaluation export the onnx model to OpenVino 
# ovc /home/absingh/downloads/TinyML/log_l2_qnn/qat_v2/qnn_qnn_BN.onnx --output_model openvino_qnn

# ------------------------
# Streaming S1/S2 dataset
# ------------------------
class StreamingS1S2Dataset(IterableDataset):
    """
    mode = "s1s2"  → concatenate S1 + S2
    mode = "s2only" → use only S2
    mode = "s1only" → use only S1
    """

    def __init__(self,
                 s1_fns,
                 s2_fns,
                 hr_label_fns=None,
                 chip_size=256,
                 num_chips_per_tile=100,
                 mode="s1s2",
                 image_transform=None,
                 label_transform=None):

        assert len(s1_fns) == len(s2_fns), "S1 and S2 lists must match"

        self.fns = list(zip(
            s1_fns,
            s2_fns,
            hr_label_fns if hr_label_fns is not None else [None] * len(s1_fns),
        ))

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.mode = mode.lower()
        assert self.mode in ["s1s2", "s2only", "s1only"], \
            "mode must be one of: s1s2, s2only, s1only"

        self.image_transform = image_transform
        self.label_transform = label_transform

    def __iter__(self):
        for s1_fn, s2_fn, hr_fn in self.fns:

            with rasterio.open(s1_fn, "r") as s1_fp, \
                 rasterio.open(s2_fn, "r") as s2_fp:

                hr_fp = rasterio.open(hr_fn, "r") if hr_fn is not None else None

                height, width = s1_fp.height, s1_fp.width
                chips_yielded = 0

                while chips_yielded < self.num_chips_per_tile:

                    if width <= self.chip_size or height <= self.chip_size:
                        x, y = 0, 0
                    else:
                        x = np.random.randint(0, width - self.chip_size)
                        y = np.random.randint(0, height - self.chip_size)

                    # S1
                    s1 = np.rollaxis(
                        s1_fp.read(window=Window(x, y, self.chip_size, self.chip_size)),
                        0, 3
                    ).astype(np.float32)

                    # S2
                    s2 = np.rollaxis(
                        s2_fp.read(window=Window(x, y, self.chip_size, self.chip_size)),
                        0, 3
                    ).astype(np.float32)

                    # Mode selection
                    if self.mode == "s1s2":
                        img = np.concatenate([s1, s2], axis=-1)
                    elif self.mode == "s2only":
                        img = s2
                    else:  # s1only
                        img = s1

                    img_raw = torch.from_numpy(img).permute(2, 0, 1)

                    if self.image_transform is not None:
                        img_norm = self.image_transform(img)
                    else:
                        img_norm = torch.from_numpy(
                            ((img - utils.IMAGE_MEANS) / utils.IMAGE_STDS).astype(np.float32)
                        ).permute(2, 0, 1)

                    hr_labels = None
                    if hr_fp is not None:
                        hr_labels = hr_fp.read(
                            window=Window(x, y, self.chip_size, self.chip_size)
                        ).squeeze()
                        if self.label_transform is not None:
                            hr_labels = self.label_transform(hr_labels)

                    if hr_labels is not None:
                        yield img_norm, hr_labels, img_raw
                    else:
                        yield img_norm, img_raw

                    chips_yielded += 1


# ------------------------
# Validation
# ------------------------
@torch.no_grad()
def validate_seg(model, val_loader, device, num_classes, epoch, save_dir):
    model.eval()
    y_true_all, y_pred_all = [], []

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        if len(batch) == 3:
            imgs_norm, labs, _ = batch
        else:
            # no labels → skip
            continue

        imgs_norm, labs = imgs_norm.to(device), labs.to(device)
        out = model(imgs_norm)

        if isinstance(out, (tuple, list)):
            seg_logits = out[0]
        elif isinstance(out, dict):
            seg_logits = out.get("logits", list(out.values())[0])
        else:
            seg_logits = out

        preds = torch.argmax(seg_logits, dim=1)
        valid_mask = (labs != 0)
        if valid_mask.sum() == 0:
            continue

        y_true_all.append(labs[valid_mask].cpu().numpy().flatten())
        y_pred_all.append(preds[valid_mask].cpu().numpy().flatten())

    if len(y_true_all) == 0:
        print("[Warning] No valid pixels in validation set.")
        return 0.0

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    oa = accuracy_score(y_true_all, y_pred_all)
    ious = jaccard_score(
        y_true_all, y_pred_all,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    miou = np.nanmean(ious)
    kappa = cohen_kappa_score(y_true_all, y_pred_all)
    
    
    # Class-wise accuracy with label names
    print("Class-wise accuracy:")
    for c in range(1, 5):   # explicitly 1..4
        mask = (y_true_all == c)
        acc = accuracy_score(y_true_all[mask], y_pred_all[mask]) if mask.sum() > 0 else 0.0
        class_name = utils.LABEL_NAMES[c]
        print(f"  {class_name}: {acc:.4f}")

      

    print(f"[Val] Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}, Kappa={kappa:.4f}")

    os.makedirs(os.path.join(save_dir, "val_logs"), exist_ok=True)
    log_path = os.path.join(save_dir, "val_logs", "val_metrics.txt")
    with open(log_path, "a") as f:
        f.write(f"Epoch {epoch}: OA={oa:.4f}, mIoU={miou:.4f}, Kappa={kappa:.4f}\n"),
        f.write(f"  {class_name}: {acc:.4f}\n")

    return miou


# ------------------------
# Train function
# ------------------------
def train_seg(args, model, device):

    df = pd.read_csv(args.list_dir)
    
    
    # ----------------------------------------------------
    # SELECT CORRECT MEAN/STDs BASED ON S1/S2 MODE
    # ----------------------------------------------------
    if args.s_mode == "s1only":
        IMAGE_MEANS = utils.IMAGE_MEANS[:2]
        IMAGE_STDS  = utils.IMAGE_STDS[:2]

    elif args.s_mode == "s2only":
        IMAGE_MEANS = utils.IMAGE_MEANS[2:15]   # adjust if your S2 bands differ
        IMAGE_STDS  = utils.IMAGE_STDS[2:15]

    elif args.s_mode == "s1s2":
        IMAGE_MEANS = utils.IMAGE_MEANS
        IMAGE_STDS  = utils.IMAGE_STDS


    train_dataset = StreamingS1S2Dataset(
        s1_fns=df["S1image_fn"].values,
        s2_fns=df["S2image_fn"].values,
        hr_label_fns=df["hr_label_fn"].values,
        chip_size=args.chip_size,
        num_chips_per_tile=args.num_chips_per_tile,
        mode=args.s_mode,
        image_transform=lambda x: torch.from_numpy(
            ((x - IMAGE_MEANS) / IMAGE_STDS).astype(np.float32)
        ).permute(2, 0, 1),
        label_transform=lambda y: torch.from_numpy(
            np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, y, mode="clip").astype(np.int64)
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4
    )

    # Validation loader
    val_loader = None
    if args.val_list and os.path.exists(args.val_list):
        df_val = pd.read_csv(args.val_list)
        val_dataset = StreamingS1S2Dataset(
            s1_fns=df_val["S1image_fn"].values,
            s2_fns=df_val["S2image_fn"].values,
            hr_label_fns=df_val["hr_label_fn"].values,
            chip_size=args.chip_size,
            num_chips_per_tile=args.num_chips_per_tile_val,
            mode=args.s_mode,
            image_transform=lambda x: torch.from_numpy(
                ((x - IMAGE_MEANS) / IMAGE_STDS).astype(np.float32)
            ).permute(2, 0, 1),
            label_transform=lambda y: torch.from_numpy(
                np.take(utils.LABEL_CLASS_TO_IDX_MAP_GT, y, mode="clip").astype(np.int64)
            ),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            num_workers=4
        )

    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

    best_miou = -1.0
    os.makedirs(args.savepath, exist_ok=True)

    for epoch in range(args.max_epochs):
        model.train()
        seg_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                imgs_norm, labs, _ = batch
            else:
                # no labels → skip
                continue

            imgs_norm = imgs_norm.to(device)
            labs = labs.to(device).long()
            labs = torch.nan_to_num(labs, nan=0.0).clamp_(0, args.num_classes - 1)

            out = model(imgs_norm)
            if isinstance(out, (tuple, list)):
                seg_logits = out[0]
            elif isinstance(out, dict):
                seg_logits = out.get("logits", list(out.values())[0])
            else:
                seg_logits = out

            if epoch == 0 and batch_idx == 0:
                print("logits shape:", seg_logits.shape)
                print("logits min/max:", seg_logits.min().item(), seg_logits.max().item())
                print("pred unique:", torch.unique(seg_logits.argmax(1)))

            seg_logits = torch.nan_to_num(seg_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_seg = seg_loss_fn(seg_logits, labs)

            optimizer.zero_grad(set_to_none=True)
            loss_seg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            seg_losses.append(loss_seg.item())
            pbar.set_postfix({"seg_loss": f"{np.mean(seg_losses):.4f}"})

        # Save checkpoint
        ckpt = os.path.join(args.savepath, f"{args.model}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")

        # Validation
        if val_loader is not None:
            miou = validate_seg(model, val_loader, device, args.num_classes, epoch, args.savepath)
            if miou > best_miou:
                best_miou = miou
                best_ckpt = os.path.join(args.savepath, f"{args.model}_best.pth")
                torch.save(model.state_dict(), best_ckpt)
                print(f"Saved BEST checkpoint: {best_ckpt} (mIoU={miou:.4f})")

    print("Training finished.")


# ------------------------
# BN folding helpers
# ------------------------
def fold_bn_into_conv(conv, bn):
    W = conv.weight
    if conv.bias is None:
        b = torch.zeros(W.size(0), device=W.device)
    else:
        b = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    alpha = gamma / torch.sqrt(var + eps)

    W_fold = W * alpha.reshape(-1, 1, 1, 1)
    b_fold = beta + (b - mean) * alpha

    return W_fold, b_fold


def replace_conv_weights(conv, W_fold, b_fold):
    conv.weight.data.copy_(W_fold)
    if conv.bias is None:
        conv.bias = torch.nn.Parameter(b_fold)
    else:
        conv.bias.data.copy_(b_fold)


def remove_bn_from_block(block, bn_index):
    block[bn_index] = torch.nn.Identity()


def fold_all_batchnorms(model):
    for name, module in model.named_modules():
        if hasattr(module, "block") and isinstance(module.block, torch.nn.Sequential):

            if (len(module.block) > 1 and
                isinstance(module.block[0], torch.nn.Conv2d) and
                isinstance(module.block[1], torch.nn.BatchNorm2d)):
                W_fold, b_fold = fold_bn_into_conv(module.block[0], module.block[1])
                replace_conv_weights(module.block[0], W_fold, b_fold)
                remove_bn_from_block(module.block, 1)
                print(f"[BN-FOLD] Folded BN at {name}.block.1")

            if (len(module.block) > 4 and
                isinstance(module.block[3], torch.nn.Conv2d) and
                isinstance(module.block[4], torch.nn.BatchNorm2d)):
                W_fold, b_fold = fold_bn_into_conv(module.block[3], module.block[4])
                replace_conv_weights(module.block[3], W_fold, b_fold)
                remove_bn_from_block(module.block, 4)
                print(f"[BN-FOLD] Folded BN at {name}.block.4")


# ------------------------
# QNN ONNX export with BN folding
# ------------------------
def export_qnn_onnx_bn(model, args, device):
    print("\n[INFO] Exporting Brevitas QNN UNet to ONNX (BN folded)")

    best_ckpt = os.path.join(args.savepath, f"{args.model}_epoch_30.pth")
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, args.in_channels, args.chip_size, args.chip_size).to(device)

    with torch.no_grad():
        _ = model(dummy)

    fold_all_batchnorms(model)

    export_path = os.path.join(args.savepath, f"{args.model}_qnn_BN.onnx")

    export_onnx_qcdq(
        model,
        args=dummy,
        export_path=export_path,
        opset_version=13
    )

    print(f"[INFO] FINN-compatible ONNX exported to: {export_path}")


# ------------------------
# ONNX inspection
# ------------------------
def inspect_qdq(path):
    m = onnx.load(path)
    q = [n for n in m.graph.node if n.op_type == "QuantizeLinear"]
    dq = [n for n in m.graph.node if n.op_type == "DequantizeLinear"]
    print(path)
    print("  QuantizeLinear:", len(q))
    print("  DequantizeLinear:", len(dq))


# ------------------------
# CLI
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_dir", type=str, default="./dataset/CSV_list/C_NYC-train_split.csv")
    parser.add_argument("--val_list", type=str, default="./dataset/CSV_list/C_NYC-val_split.csv")

    parser.add_argument("--savepath", type=str, default="./log_INT8/unettS1/")
    parser.add_argument("--model", type=str, default="qnn",
                        choices=["unet", "qnn"])
                        
                        
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--in_channels", type=int, default=2)  # will be overwritten by s_mode
    
    
    parser.add_argument("--chip_size", type=int, default=256)
    parser.add_argument("--num_chips_per_tile", type=int, default=1)
    parser.add_argument("--num_chips_per_tile_val", type=int, default=1)
    
    parser.add_argument("--max_epochs", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=16)
    
    
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=0.00001)
    parser.add_argument("--max_grad_norm", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_dataparallel", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")

    parser.add_argument("--s_mode", type=str, default="s1only",
                        choices=["s1s2", "s2only", "s1only"],
                        help="Choose which Sentinel inputs to use")


    return parser.parse_args()


# ------------------------
# Model preparation
# ------------------------
def prepare_model(args, device):
    if args.s_mode == "s1s2":
        args.in_channels = 15   # S1(4) + S2(4)
    elif args.s_mode == "s2only":
        args.in_channels = 13   # S2 only
    elif args.s_mode == "s1only":
        args.in_channels = 2   # S1 only

    print(f"Building model '{args.model}' with mode={args.s_mode}, in_channels={args.in_channels}")

    model = get_model(
        args.model,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        img_size=args.chip_size
    )
    model = model.to(device)

    if args.use_dataparallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    args = parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
        print(f"Using CUDA device(s): {args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.savepath, exist_ok=True)

    model = prepare_model(args, device)
    summary(model)
    train_seg(args, model, device)
    export_qnn_onnx_bn(model, args, device)


# Example inspection (adjust filenames as needed)
    onnx_qnn_bn = os.path.join(args.savepath, f"{args.model}_qnn_BN.onnx")
    if os.path.exists(onnx_qnn_bn):
        inspect_qdq(onnx_qnn_bn)
