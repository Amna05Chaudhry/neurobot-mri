"""Turn a probability map into structured, human-readable findings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

THRESHOLD = 0.5
MIN_REGION_PX = 40  # at 256x256; smaller blobs are treated as noise


@dataclass
class Findings:
    tumour_present: bool
    area_pct: float = 0.0
    area_px: int = 0
    n_regions: int = 0
    hemisphere: str = "n/a"
    quadrant: str = "n/a"
    centroid_xy: tuple[int, int] | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    mean_prob_in_mask: float = 0.0
    max_prob: float = 0.0
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if not self.tumour_present:
            return (f"No FLAIR abnormality above the size threshold on this slice "
                    f"(max probability {self.max_prob:.2f}).")
        return (f"FLAIR abnormality detected: {self.area_pct:.1f}% of the slice, "
                f"{self.hemisphere} hemisphere, {self.quadrant} quadrant, "
                f"{self.n_regions} region(s), mean confidence {self.mean_prob_in_mask:.2f}.")


def clean_mask(prob: np.ndarray, thr: float = THRESHOLD, min_px: int = MIN_REGION_PX) -> np.ndarray:
    """Binarise and drop connected components smaller than `min_px`."""
    mask = prob >= thr
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_px])
    return keep


def extract(prob: np.ndarray, thr: float = THRESHOLD) -> tuple[Findings, np.ndarray]:
    """Findings for one slice. Assumes radiological convention: patient's left
    is on the image right, anterior is at the top."""
    mask = clean_mask(prob, thr)
    h, w = prob.shape
    max_p = float(prob.max())
    if not mask.any():
        f = Findings(tumour_present=False, max_prob=max_p, confidence=round(1 - max_p, 3))
        return f, mask
    _, n = ndimage.label(mask)
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    hemi = "right" if cx < w / 2 else "left"           # radiological convention
    ap = "anterior" if cy < h / 2 else "posterior"
    notes = []
    if abs(cx - w / 2) < w * 0.05:
        notes.append("Centre lies close to the midline; hemisphere call is uncertain.")
    mean_p = float(prob[mask].mean())
    f = Findings(
        tumour_present=True,
        area_pct=round(100.0 * mask.sum() / mask.size, 2),
        area_px=int(mask.sum()),
        n_regions=int(n),
        hemisphere=hemi,
        quadrant=f"{ap}-{hemi}",
        centroid_xy=(int(round(cx)), int(round(cy))),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        mean_prob_in_mask=round(mean_p, 3),
        max_prob=round(max_p, 3),
        confidence=round(mean_p, 3),
        notes=notes,
    )
    return f, mask
