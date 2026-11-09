"""Structured, templated slice report (deterministic; no LLM needed)."""
from __future__ import annotations

from datetime import datetime

from .findings import Findings

SIZE_BANDS = [(1.0, "small"), (4.0, "moderate"), (100.0, "large")]


def _size_word(pct: float) -> str:
    return next(w for lim, w in SIZE_BANDS if pct <= lim)


def build_report(f: Findings, filename: str = "uploaded slice", metrics: dict | None = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    head = [f"### NeuroBot slice report", f"*{filename} · generated {ts}*", ""]
    if f.tumour_present:
        x0, y0, x1, y1 = f.bbox_xyxy
        body = [
            "**Findings**",
            f"- A {_size_word(f.area_pct)} region of FLAIR-type abnormality is segmented, covering "
            f"**{f.area_pct:.1f}%** of the slice ({f.area_px} px) in {f.n_regions} connected region(s).",
            f"- Centre of mass lies in the **{f.hemisphere} hemisphere**, {f.quadrant.replace('-', ' ')} quadrant "
            f"(radiological convention).",
            f"- Bounding box (x0, y0, x1, y1) = ({x0}, {y0}, {x1}, {y1}) on the 256x256 image "
            f"({x1 - x0 + 1} x {y1 - y0 + 1} px).",
            f"- Model confidence: mean probability inside the mask **{f.mean_prob_in_mask:.2f}**, "
            f"slice maximum {f.max_prob:.2f}.",
            *[f"- Note: {n}" for n in f.notes],
            "",
            "**Impression (automated)**",
            f"- Appearance is compatible with a FLAIR-hyperintense lesion such as a lower-grade glioma, "
            f"but FLAIR signal is not specific (oedema, demyelination, ischaemia and post-treatment change "
            f"can look similar).",
            "",
            "**Suggested next steps for the clinical team**",
            "- Neuroradiologist review of the full multi-sequence study and comparison with prior imaging.",
            "- If a glioma is suspected: neurosurgical / neuro-oncology referral; tissue diagnosis with "
            "molecular markers (IDH, 1p/19q) per WHO CNS5.",
        ]
    else:
        body = [
            "**Findings**",
            f"- No region of FLAIR-type abnormality above the size threshold on this slice "
            f"(highest pixel probability {f.max_prob:.2f}).",
            "",
            "**Impression (automated)**",
            "- No abnormality detected on this single slice. This does not exclude disease on other "
            "slices or sequences.",
        ]
    tail = ["", "**Limitations**",
            "- 2-D, single-slice analysis by a model trained on 110 TCGA lower-grade glioma patients only.",
            "- Location assumes anterior at top and patient's left on the image right."]
    if metrics:
        s = metrics.get("segmentation", {})
        tail.append(f"- Held-out test performance: per-patient Dice {s.get('per_patient_volume_dice_mean', 0):.2f}, "
                    f"tumour-slice Dice {s.get('tumour_slice_dice_mean', 0):.2f}.")
    tail += ["", "> Research demo, **not for clinical use**. Not a medical device."]
    return "\n".join(head + body + tail)
