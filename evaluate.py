"""Evaluate the trained model on the held-out TEST patients at native 256x256.

Usage:  python evaluate.py
Writes results/metrics.json, results/test_examples.png and refreshes examples/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neurobot.data import list_pairs, load_splits, normalize
from neurobot.findings import clean_mask
from neurobot.model import load_model

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
EX = ROOT / "examples"


def dice(p: np.ndarray, g: np.ndarray) -> float:
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * (p & g).sum() / s)


def iou(p: np.ndarray, g: np.ndarray) -> float:
    u = (p | g).sum()
    return 1.0 if u == 0 else float((p & g).sum() / u)


@torch.no_grad()
def predict_batch(model, imgs256: np.ndarray, size: int) -> np.ndarray:
    small = np.stack([np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR)) for im in imgs256])
    logits = model(torch.from_numpy(normalize(small)))
    up = torch.nn.functional.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
    return torch.sigmoid(up)[:, 0].numpy()


def main() -> None:
    torch.set_num_threads(8)
    model, meta = load_model()
    size = meta.get("img_size", 128)
    test_pts = set(load_splits()["test"])
    pairs = [p for p in list_pairs() if p["patient"] in test_pts]
    pairs.sort(key=lambda p: (p["patient"], p["slice"]))

    imgs = np.stack([np.asarray(Image.open(p["image"]).convert("RGB")) for p in pairs])
    gts = np.stack([np.asarray(Image.open(p["mask"]).convert("L")) > 127 for p in pairs])
    probs = np.concatenate([predict_batch(model, imgs[i:i + 8], size) for i in range(0, len(imgs), 8)])
    preds = np.stack([clean_mask(pr) for pr in probs])

    inter = (preds & gts).sum()
    g_dice = float(2 * inter / (preds.sum() + gts.sum()))
    g_iou = float(inter / (preds | gts).sum())
    pos = gts.reshape(len(gts), -1).any(1)
    pos_d = [dice(p, g) for p, g in zip(preds[pos], gts[pos])]
    pos_i = [iou(p, g) for p, g in zip(preds[pos], gts[pos])]

    by_pt = defaultdict(list)
    for i, p in enumerate(pairs):
        by_pt[p["patient"]].append(i)
    vol_d = {pt: dice(preds[ix], gts[ix]) for pt, ix in by_pt.items()}
    vol_i = {pt: iou(preds[ix], gts[ix]) for pt, ix in by_pt.items()}

    det = preds.reshape(len(preds), -1).any(1)
    tp, tn = int((det & pos).sum()), int((~det & ~pos).sum())
    fp, fn = int((det & ~pos).sum()), int((~det & pos).sum())

    metrics = {
        "test_patients": len(by_pt), "test_slices": len(pairs), "tumour_slices": int(pos.sum()),
        "model": meta,
        "segmentation": {
            "per_patient_volume_dice_mean": float(np.mean(list(vol_d.values()))),
            "per_patient_volume_dice_median": float(np.median(list(vol_d.values()))),
            "per_patient_volume_iou_mean": float(np.mean(list(vol_i.values()))),
            "tumour_slice_dice_mean": float(np.mean(pos_d)),
            "tumour_slice_iou_mean": float(np.mean(pos_i)),
            "global_pixel_dice": g_dice, "global_pixel_iou": g_iou,
        },
        "slice_detection": {
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / len(pairs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
        "per_patient_volume_dice": {k: round(v, 3) for k, v in sorted(vol_d.items())},
    }
    RES.mkdir(exist_ok=True)
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: metrics[k] for k in ("test_patients", "test_slices", "tumour_slices",
                                              "segmentation", "slice_detection")}, indent=1))

    # Examples for the app: tumour slices of small / medium / large size + a clean slice,
    # chosen from test patients so the demo never shows training data.
    rng = np.random.default_rng(0)
    areas = gts.reshape(len(gts), -1).sum(1)
    pos_idx = np.nonzero(pos)[0]
    order = pos_idx[np.argsort(areas[pos_idx])]
    picks = {"large": order[int(len(order) * 0.95)], "medium": order[int(len(order) * 0.6)],
             "small": order[int(len(order) * 0.25)]}
    neg_idx = np.nonzero(~pos & (imgs.reshape(len(imgs), -1).mean(1) > 25))[0]
    picks["no_tumour"] = rng.choice(neg_idx)
    picks["medium_2"] = order[int(len(order) * 0.75)] if pairs[order[int(len(order) * 0.75)]]["patient"] != \
        pairs[picks["medium"]]["patient"] else order[int(len(order) * 0.8)]
    EX.mkdir(exist_ok=True)
    for f in EX.glob("*.png"):
        f.unlink()
    for name, i in picks.items():
        p = pairs[i]
        Image.fromarray(imgs[i]).save(EX / f"{name}_{p['patient']}_s{p['slice']}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, len(picks), figsize=(3 * len(picks), 9))
    for c, (name, i) in enumerate(picks.items()):
        ax[0, c].imshow(imgs[i]); ax[0, c].set_title(name)
        ax[1, c].imshow(gts[i], cmap="gray"); ax[1, c].set_title("ground truth")
        ax[2, c].imshow(preds[i], cmap="gray"); ax[2, c].set_title(f"prediction (Dice {dice(preds[i], gts[i]):.2f})")
        for r in range(3):
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(RES / "test_examples.png", dpi=90)
    print(f"saved {RES / 'metrics.json'}, {RES / 'test_examples.png'} and {len(picks)} examples")


if __name__ == "__main__":
    main()
