"""U-Net with a VGG-19 (batch-norm) encoder, plus inference helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import normalize

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "neurobot_unet_vgg19.pt"
ENCODER = "vgg19_bn"
IMG_SIZE = 128  # training/inference resolution (CPU budget); outputs are upsampled


def build_model(pretrained: bool = True) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1, decoder_channels=(256, 128, 64, 32, 16))


def save_weights(model: torch.nn.Module, path: Path = WEIGHTS, meta: dict | None = None) -> None:
    """Store weights as fp16 to keep the file small enough for git."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "meta": meta or {}}, path)


def load_model(path: Path = WEIGHTS) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                           for k, v in ckpt["state_dict"].items()})
    model.eval()
    return model, ckpt.get("meta", {})


def to_rgb256(img: Image.Image) -> np.ndarray:
    """Any uploaded image -> 256x256x3 uint8 (grayscale is replicated)."""
    return np.asarray(img.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)


@torch.no_grad()
def predict_proba(model: torch.nn.Module, rgb256: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Return a 256x256 float probability map for one RGB slice."""
    small = np.asarray(Image.fromarray(rgb256).resize((size, size), Image.BILINEAR))
    x = torch.from_numpy(normalize(small)).unsqueeze(0)
    logits = model(x)
    up = torch.nn.functional.interpolate(logits, size=(256, 256), mode="bilinear", align_corners=False)
    return torch.sigmoid(up)[0, 0].numpy()
