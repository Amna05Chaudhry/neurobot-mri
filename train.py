"""Train the VGG-19 U-Net on the LGG dataset (CPU-friendly).

Usage:  python train.py --epochs 8 --size 128
Saves the best (validation Dice) weights to weights/neurobot_unet_vgg19.pt and a
training log to results/train_log.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neurobot.data import load_split, normalize
from neurobot.model import ENCODER, build_model, save_weights

ROOT = Path(__file__).resolve().parent


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip, small shift and intensity jitter (uint8 in, float out)."""
    x = x.astype(np.float32)
    flip = rng.random(len(x)) < 0.5
    x[flip] = x[flip, :, ::-1]
    y = y.copy()
    y[flip] = y[flip, :, ::-1]
    for i in range(len(x)):
        dy, dx = rng.integers(-8, 9, size=2)
        x[i] = np.roll(x[i], (dy, dx), axis=(0, 1))
        y[i] = np.roll(y[i], (dy, dx), axis=(0, 1))
    gain = rng.uniform(0.85, 1.15, size=(len(x), 1, 1, 1)).astype(np.float32)
    x = np.clip(x * gain, 0, 255)
    return x, y


def evaluate(model, x: np.ndarray, y: np.ndarray, bs: int = 32) -> dict:
    model.eval()
    inter = union = 0.0
    slice_d = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(normalize(x[i:i + bs]))
            pred = (torch.sigmoid(model(xb))[:, 0].numpy() > 0.5)
            gt = y[i:i + bs].astype(bool)
            inter += (pred & gt).sum()
            union += pred.sum() + gt.sum()
            for p, g in zip(pred, gt):
                if g.any():
                    slice_d.append(2 * (p & g).sum() / (p.sum() + g.sum()))
    return {"global_dice": float(2 * inter / max(union, 1)), "mean_pos_slice_dice": float(np.mean(slice_d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--neg-ratio", type=float, default=0.5,
                    help="tumour-free slices sampled per tumour slice each epoch")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    tr, va = load_split(args.size, "train"), load_split(args.size, "val")
    pos = np.nonzero(tr["y"].reshape(len(tr["y"]), -1).any(1))[0]
    neg = np.setdiff1d(np.arange(len(tr["y"])), pos)
    print(f"train slices {len(tr['x'])} ({len(pos)} with tumour), val slices {len(va['x'])}; "
          f"patients train {len(set(tr['pid']))} / val {len(set(va['pid']))}")

    model = build_model(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_neg = int(len(pos) * args.neg_ratio)
    steps_per_epoch = (len(pos) + n_neg + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15)
    log, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
        rng.shuffle(idx)
        t0, losses = time.time(), []
        for s in range(0, len(idx), args.batch):
            b = idx[s:s + args.batch]
            xb, yb = augment(tr["x"][b], tr["y"][b], rng)
            xt = torch.from_numpy(normalize(xb.astype(np.uint8)))
            yt = torch.from_numpy(yb[:, None].astype(np.float32))
            logits = model(xt)
            loss = F.binary_cross_entropy_with_logits(logits, yt) + dice_loss(logits, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if (s // args.batch) % 25 == 0:
                print(f"  ep{ep} step {s // args.batch}/{steps_per_epoch} loss {np.mean(losses[-25:]):.3f}",
                      flush=True)
        m = evaluate(model, va["x"], va["y"])
        rec = {"epoch": ep, "loss": float(np.mean(losses)), **m, "minutes": round((time.time() - t0) / 60, 1)}
        log.append(rec)
        print(rec, flush=True)
        if m["global_dice"] > best:
            best = m["global_dice"]
            save_weights(model, meta={"encoder": ENCODER, "img_size": args.size, "epoch": ep,
                                      "val": m, "seed": args.seed})
            print(f"  saved best (val global Dice {best:.3f})")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=1))


if __name__ == "__main__":
    main()
