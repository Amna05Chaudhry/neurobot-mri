"""Dataset indexing, patient-level split and a cached, resized array version.

The LGG dataset has one folder per patient with `<case>_<slice>.tif` (RGB:
pre-contrast, FLAIR, post-contrast) and `<case>_<slice>_mask.tif` (0/255).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
SPLITS = ROOT / "results" / "splits.json"  # committed so the split is reproducible

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SLICE_RE = re.compile(r"_(\d+)\.tif$")


def list_pairs() -> list[dict]:
    """All (image, mask) pairs with their patient id and slice number."""
    pairs, seen = [], set()
    # The archive ships kaggle_3m twice (as on Kaggle); keep the shortest path per file.
    for mask in sorted(RAW.rglob("*_mask.tif"), key=lambda p: (len(p.parts), str(p))):
        img = mask.with_name(mask.name.replace("_mask.tif", ".tif"))
        if mask.name in seen or not img.exists():
            continue
        seen.add(mask.name)
        m =_SLICE_RE.search(img.name)
        pairs.append({"image": img, "mask": mask, "patient": mask.parent.name,
                      "slice": int(m.group(1)) if m else -1})
    pairs.sort(key=lambda p: (p["patient"], p["slice"]))
    if not pairs:
        raise FileNotFoundError(f"No *_mask.tif under {RAW}. Run scripts/download_data.py first.")
    return pairs


def make_splits(patients: list[str], seed: int = 42, val: float = 0.15, test: float = 0.15) -> dict:
    """Patient-level split: no patient's slices appear in two splits."""
    pts = sorted(set(patients))
    random.Random(seed).shuffle(pts)
    n_test, n_val = round(len(pts) * test), round(len(pts) * val)
    return {"test": sorted(pts[:n_test]), "val": sorted(pts[n_test:n_test + n_val]),
            "train": sorted(pts[n_test + n_val:])}


def load_splits() -> dict:
    if SPLITS.exists():
        return json.loads(SPLITS.read_text())
    splits = make_splits([p["patient"] for p in list_pairs()])
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    SPLITS.write_text(json.dumps(splits, indent=1))
    return splits


def build_cache(size: int) -> Path:
    """Resize every slice once and store uint8 arrays per split in an .npz."""
    out = CACHE / f"lgg_{size}.npz"
    if out.exists():
        return out
    pairs = list_pairs()
    splits = load_splits()
    where = {p: s for s, pts in splits.items() for p in pts}
    arrays: dict[str, dict[str, list]] = {s: {"x": [], "y": [], "pid": [], "sl": []} for s in splits}
    for i, p in enumerate(pairs):
        img = Image.open(p["image"]).convert("RGB").resize((size, size), Image.BILINEAR)
        msk = Image.open(p["mask"]).convert("L").resize((size, size), Image.NEAREST)
        a = arrays[where[p["patient"]]]
        a["x"].append(np.asarray(img, dtype=np.uint8))
        a["y"].append((np.asarray(msk) > 127).astype(np.uint8))
        a["pid"].append(p["patient"])
        a["sl"].append(p["slice"])
        if i % 500 == 0:
            print(f"  cached {i}/{len(pairs)}")
    CACHE.mkdir(parents=True, exist_ok=True)
    flat = {}
    for s, a in arrays.items():
        flat[f"{s}_x"] = np.stack(a["x"])
        flat[f"{s}_y"] = np.stack(a["y"])
        flat[f"{s}_pid"] = np.array(a["pid"])
        flat[f"{s}_sl"] = np.array(a["sl"])
    np.savez(out, **flat)
    return out


def load_split(size: int, split: str) -> dict[str, np.ndarray]:
    d = np.load(build_cache(size))
    return {k: d[f"{split}_{k}"] for k in ("x", "y", "pid", "sl")}


def normalize(x_uint8: np.ndarray) -> np.ndarray:
    """(..., H, W, 3) uint8 -> (..., 3, H, W) float32 ImageNet-normalised."""
    x = (x_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.moveaxis(x, -1, -3).copy()
