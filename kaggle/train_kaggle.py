# NeuroBot: GPU training on Kaggle (one notebook cell).
# 1. Kaggle -> New Notebook -> "Add Input" -> dataset "mateuszbuda/lgg-mri-segmentation"
# 2. Settings: Accelerator = GPU T4, Internet = On
# 3. Paste this whole file into one cell and run (about 20-30 min).
# 4. Download /kaggle/working/neurobot_unet_vgg19.pt (and metrics_kaggle.json) into weights/ and results/.
# Same architecture, patient split and checkpoint format as the repo (neurobot/model.py).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0"], check=True)

import json, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import ndimage

SIZE, EPOCHS, BATCH, LR, SEED = 256, 30, 16, 3e-4, 42
SPLITS = json.loads(r'''{"test":["TCGA_CS_4942_19970222","TCGA_CS_6188_20010812","TCGA_CS_6669_20020102","TCGA_DU_6400_19830518","TCGA_DU_6407_19860514","TCGA_DU_7008_19830723","TCGA_DU_8165_19970205","TCGA_DU_A5TS_19970726","TCGA_DU_A5TU_19980312","TCGA_EZ_7264_20010816","TCGA_FG_5964_20010511","TCGA_FG_6692_20020606","TCGA_FG_7637_20000922","TCGA_HT_7473_19970826","TCGA_HT_8105_19980826","TCGA_HT_8114_19981030"],"val":["TCGA_CS_6186_20000601","TCGA_DU_5855_19951217","TCGA_DU_5871_19941206","TCGA_DU_7010_19860307","TCGA_DU_7019_19940908","TCGA_DU_8162_19961029","TCGA_DU_8164_19970111","TCGA_DU_8167_19970402","TCGA_DU_A5TR_19970726","TCGA_FG_5962_20000626","TCGA_FG_6691_20020405","TCGA_HT_7602_19951103","TCGA_HT_7608_19940304","TCGA_HT_7680_19970202","TCGA_HT_7686_19950629","TCGA_HT_7690_19960312"],"train":["TCGA_CS_4941_19960909","TCGA_CS_4943_20000902","TCGA_CS_4944_20010208","TCGA_CS_5393_19990606","TCGA_CS_5395_19981004","TCGA_CS_5396_20010302","TCGA_CS_5397_20010315","TCGA_CS_6290_20000917","TCGA_CS_6665_20010817","TCGA_CS_6666_20011109","TCGA_CS_6667_20011105","TCGA_CS_6668_20011025","TCGA_DU_5849_19950405","TCGA_DU_5851_19950428","TCGA_DU_5852_19950709","TCGA_DU_5853_19950823","TCGA_DU_5854_19951104","TCGA_DU_5872_19950223","TCGA_DU_5874_19950510","TCGA_DU_6399_19830416","TCGA_DU_6401_19831001","TCGA_DU_6404_19850629","TCGA_DU_6405_19851005","TCGA_DU_6408_19860521","TCGA_DU_7013_19860523","TCGA_DU_7014_19860618","TCGA_DU_7018_19911220","TCGA_DU_7294_19890104","TCGA_DU_7298_19910324","TCGA_DU_7299_19910417","TCGA_DU_7300_19910814","TCGA_DU_7301_19911112","TCGA_DU_7302_19911203","TCGA_DU_7304_19930325","TCGA_DU_7306_19930512","TCGA_DU_7309_19960831","TCGA_DU_8163_19961119","TCGA_DU_8166_19970322","TCGA_DU_8168_19970503","TCGA_DU_A5TP_19970614","TCGA_DU_A5TT_19980318","TCGA_DU_A5TW_19980228","TCGA_DU_A5TY_19970709","TCGA_FG_6688_20020215","TCGA_FG_6689_20020326","TCGA_FG_6690_20020226","TCGA_FG_7634_20000128","TCGA_FG_7643_20021104","TCGA_FG_8189_20030516","TCGA_FG_A4MT_20020212","TCGA_FG_A4MU_20030903","TCGA_FG_A60K_20040224","TCGA_HT_7475_19970918","TCGA_HT_7605_19950916","TCGA_HT_7616_19940813","TCGA_HT_7684_19950816","TCGA_HT_7692_19960724","TCGA_HT_7693_19950520","TCGA_HT_7694_19950404","TCGA_HT_7855_19951020","TCGA_HT_7856_19950831","TCGA_HT_7860_19960513","TCGA_HT_7874_19950902","TCGA_HT_7877_19980917","TCGA_HT_7879_19981009","TCGA_HT_7881_19981015","TCGA_HT_7882_19970125","TCGA_HT_7884_19980913","TCGA_HT_8018_19970411","TCGA_HT_8106_19970727","TCGA_HT_8107_19980708","TCGA_HT_8111_19980330","TCGA_HT_8113_19930809","TCGA_HT_8563_19981209","TCGA_HT_A5RC_19990831","TCGA_HT_A616_19991226","TCGA_HT_A61A_20000127","TCGA_HT_A61B_19991127"]}''')
OUT = Path("/kaggle/working")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
print("device", dev)

# ---------------------------------------------------------------- data
masks = sorted(Path("/kaggle/input").rglob("*_mask.tif"), key=lambda p: (len(p.parts), str(p)))
seen, pairs = set(), []
for m in masks:
    if m.name in seen:
        continue
    seen.add(m.name)
    pairs.append((m.parent.name, int(re.search(r"_(\d+)_mask\.tif$", m.name).group(1)),
                  m.with_name(m.name.replace("_mask.tif", ".tif")), m))
pairs.sort(key=lambda t: (t[0], t[1]))
print(len(pairs), "slices,", len({p[0] for p in pairs}), "patients")
assert len(pairs) == 3929, "unexpected dataset size"

def load(split):
    sel = [p for p in pairs if p[0] in set(SPLITS[split])]
    x = np.stack([np.asarray(Image.open(p[2]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)) for p in sel])
    y = np.stack([(np.asarray(Image.open(p[3]).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127) for p in sel])
    return x, y.astype(np.uint8), [p[0] for p in sel]

tr_x, tr_y, _ = load("train"); va_x, va_y, _ = load("val")
print("train", tr_x.shape, "val", va_x.shape)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def to_tensor(xb):
    x = torch.from_numpy(xb).to(dev).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD

# ---------------------------------------------------------------- model
model = smp.Unet(encoder_name="vgg19_bn", encoder_weights="imagenet", in_channels=3, classes=1,
                 decoder_channels=(256, 128, 64, 32, 16)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
pos = np.nonzero(tr_y.reshape(len(tr_y), -1).any(1))[0]
neg = np.setdiff1d(np.arange(len(tr_y)), pos)
n_neg = len(pos) // 2
steps = (len(pos) + n_neg + BATCH - 1) // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps, pct_start=0.15)
scaler = torch.amp.GradScaler(enabled=dev == "cuda")

def dice_loss(logits, t, eps=1.0):
    p = torch.sigmoid(logits)
    return (1 - (2 * (p * t).sum((1, 2, 3)) + eps) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps)).mean()

def augment(x, y):
    flip = torch.rand(len(x), device=dev) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x); y = torch.where(flip.view(-1, 1, 1, 1), y.flip(-1), y)
    dy, dx = (int(v) for v in rng.integers(-16, 17, size=2))
    x = torch.roll(x, (dy, dx), (2, 3)); y = torch.roll(y, (dy, dx), (2, 3))
    return x * torch.empty(len(x), 1, 1, 1, device=dev).uniform_(0.85, 1.15), y

@torch.no_grad()
def predict(x_uint8, bs=32):
    model.eval(); out = []
    for i in range(0, len(x_uint8), bs):
        with torch.autocast(dev, enabled=dev == "cuda"):
            out.append(torch.sigmoid(model(to_tensor(x_uint8[i:i + bs])).float())[:, 0].cpu().numpy())
    return np.concatenate(out)

def save(meta):
    sd = {k: (v.detach().cpu().half() if v.is_floating_point() else v.cpu()) for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "meta": meta}, OUT / "neurobot_unet_vgg19.pt")

best, log = -1, []
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); losses = []
    idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]); rng.shuffle(idx)
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        x = to_tensor(tr_x[b]); y = torch.from_numpy(tr_y[b]).to(dev)[:, None].float()
        x, y = augment(x, y)
        with torch.autocast(dev, enabled=dev == "cuda"):
            logits = model(x)
        logits = logits.float()
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        losses.append(loss.item())
    pr = predict(va_x) > 0.5; gt = va_y.astype(bool)
    vd = float(2 * (pr & gt).sum() / (pr.sum() + gt.sum()))
    rec = {"epoch": ep, "loss": round(float(np.mean(losses)), 4), "val_global_dice": round(vd, 4),
           "sec": round(time.time() - t0)}
    log.append(rec); print(rec, flush=True)
    if vd > best:
        best = vd
        save({"encoder": "vgg19_bn", "img_size": SIZE, "epoch": ep, "val": {"global_dice": vd},
              "seed": SEED, "trained_on": "Kaggle GPU"})
json.dump(log, open(OUT / "train_log_kaggle.json", "w"), indent=1)
print("best val global Dice", round(best, 4))

# ---------------------------------------------------------------- quick test-set check (full eval: evaluate.py)
ck = torch.load(OUT / "neurobot_unet_vgg19.pt", weights_only=False)
model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
te_x, te_y, te_pid = load("test")

def clean(p, min_px=40 * (SIZE / 256) ** 2):
    m = p >= 0.5; lab, n = ndimage.label(m)
    if n == 0: return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_px])

pr = np.stack([clean(p) for p in predict(te_x)]); gt = te_y.astype(bool)
d = lambda a, b: 1.0 if a.sum() + b.sum() == 0 else 2 * (a & b).sum() / (a.sum() + b.sum())
by = defaultdict(list)
for i, p in enumerate(te_pid): by[p].append(i)
vol = [d(pr[ix], gt[ix]) for ix in by.values()]
posm = gt.reshape(len(gt), -1).any(1)
res = {"test_patients": len(by), "per_patient_volume_dice_mean": float(np.mean(vol)),
       "tumour_slice_dice_mean": float(np.mean([d(a, b) for a, b in zip(pr[posm], gt[posm])])),
       "global_pixel_dice": float(d(pr, gt))}
json.dump(res, open(OUT / "metrics_kaggle.json", "w"), indent=1)
print(res)
