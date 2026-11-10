"""NeuroBot Gradio app: brain MRI FLAIR-abnormality segmentation + RAG consultation.

Run:  python app.py   (then open http://127.0.0.1:7860)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import llm
from neurobot.findings import Findings, extract
from neurobot.model import WEIGHTS, load_model, predict_proba, to_rgb256
from neurobot.rag import KnowledgeBase, answer, sources_markdown
from neurobot.report import build_report

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted((ROOT / "examples").glob("*.png"))
METRICS_PATH = ROOT / "results" / "metrics.json"
TEAL = "#0F766E"
AMBER = (245, 158, 11)

print("Loading segmentation model...")
MODEL, META = load_model(WEIGHTS)
print("Building knowledge-base index...")
KB = KnowledgeBase()
METRICS = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


# --------------------------------------------------------------------------- helpers
def make_overlay(rgb: np.ndarray, mask: np.ndarray, f: Findings) -> Image.Image:
    base = rgb.astype(np.float32)
    out = base.copy()
    if mask.any():
        fill = np.array(AMBER, dtype=np.float32)
        out[mask] = 0.55 * base[mask] + 0.45 * fill
        edge = mask & ~ndimage.binary_erosion(mask, iterations=1)
        out[edge] = fill
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    if f.bbox_xyxy:
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = f.bbox_xyxy
        d.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=(20, 184, 166), width=2)
    return img.resize((384, 384), Image.NEAREST)


def findings_card(f: Findings) -> str:
    if f.tumour_present:
        status = '<span class="nb-pill nb-pos">Abnormality detected</span>'
        x0, y0, x1, y1 = f.bbox_xyxy
        tiles = [
            ("Area of slice", f"{f.area_pct:.1f}%", f"{f.area_px:,} px · {f.n_regions} region(s)"),
            ("Location", f"{f.hemisphere.title()} hemisphere", f.quadrant.replace("-", " ")),
            ("Confidence", f"{f.mean_prob_in_mask:.2f}", f"mean prob. in mask · max {f.max_prob:.2f}"),
            ("Bounding box", f"{x1 - x0 + 1}×{y1 - y0 + 1} px", f"({x0}, {y0}) → ({x1}, {y1})"),
        ]
    else:
        status = '<span class="nb-pill nb-neg">No abnormality detected</span>'
        tiles = [
            ("Area of slice", "0%", "below size threshold"),
            ("Location", "n/a", "no region"),
            ("Confidence", f"{f.confidence:.2f}", f"1 − max prob. ({f.max_prob:.2f})"),
            ("Bounding box", "n/a", ""),
        ]
    tiles_html = "".join(
        f'<div class="nb-tile"><div class="nb-tile-label">{a}</div>'
        f'<div class="nb-tile-value">{b}</div><div class="nb-tile-sub">{c}</div></div>'
        for a, b, c in tiles)
    notes = "".join(f'<div class="nb-note">⚠ {n}</div>' for n in f.notes)
    return f'<div class="nb-card"><div class="nb-card-head">Findings {status}</div>' \
           f'<div class="nb-tiles">{tiles_html}</div>{notes}</div>'


EMPTY_CARD = ('<div class="nb-card nb-empty">Upload a brain MRI slice or pick an example '
              'to see findings here.</div>')


# --------------------------------------------------------------------------- callbacks
def analyze(img: Image.Image | None, progress=gr.Progress()):
    if img is None:
        raise gr.Error("Please upload an MRI slice (PNG, JPG or TIF) or click an example.")
    progress(0.2, desc="Segmenting…")
    rgb = to_rgb256(img)
    prob = predict_proba(MODEL, rgb, size=META.get("img_size", 128))
    f, mask = extract(prob)
    progress(0.8, desc="Writing report…")
    name = Path(getattr(img, "filename", "") or "uploaded slice").name
    report = build_report(f, name, METRICS)
    tmp = Path(tempfile.gettempdir()) / "neurobot_report.md"
    tmp.write_text(report, encoding="utf-8")
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((384, 384), Image.NEAREST)
    state = {"summary": f.summary(), "findings": f.to_dict()}
    banner = f"**Current scan:** {f.summary()}"
    return (Image.fromarray(rgb).resize((384, 384)), make_overlay(rgb, mask, f), mask_img,
            findings_card(f), report, str(tmp), state, banner, [])


def chat(message: str, history: list, state: dict | None):
    message = (message or "").strip()
    if not message:
        return history, gr.update(), ""
    findings_text = state["summary"] if state else "No scan has been analysed yet."
    reply, hits, used = answer(KB, message, findings_text, history)
    tag = "" if used else "\n\n<span class='nb-offline'>LLM offline</span>"
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": reply + tag}]
    return history, sources_markdown(hits), ""


# --------------------------------------------------------------------------- UI
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(button_primary_background_fill=TEAL, button_primary_background_fill_hover="#115E59")

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto;}
#nb-header {background: linear-gradient(120deg, #0F766E 0%, #115E59 55%, #134E4A 100%);
  color: white; border-radius: 18px; padding: 22px 28px; margin-bottom: 6px;}
#nb-header h1 {margin: 0; font-size: 30px; font-weight: 800; color: white; letter-spacing: -0.5px;}
#nb-header p {margin: 4px 0 0 0; opacity: .92; font-size: 15px; color: white;}
#nb-header .nb-badges {margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;}
#nb-header .nb-badge {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
  padding: 3px 10px; border-radius: 999px; font-size: 12px; color: white;}
.nb-disclaimer {border-left: 4px solid #F59E0B; background: rgba(245,158,11,.10);
  padding: 8px 14px; border-radius: 10px; font-size: 13px;}
.nb-card {border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  border-radius: 16px; padding: 16px 18px;}
.nb-empty {color: var(--body-text-color-subdued); text-align: center; padding: 34px;}
.nb-card-head {font-weight: 700; font-size: 16px; display: flex; gap: 10px; align-items: center;
  margin-bottom: 12px;}
.nb-pill {font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px;}
.nb-pos {background: rgba(245,158,11,.18); color: #B45309;}
.nb-neg {background: rgba(15,118,110,.15); color: #0F766E;}
.dark .nb-pos {color: #FBBF24;} .dark .nb-neg {color: #5EEAD4;}
.nb-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px;}
.nb-tile {border-radius: 12px; padding: 10px 12px; background: var(--background-fill-secondary);}
.nb-tile-label {font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--body-text-color-subdued);}
.nb-tile-value {font-size: 20px; font-weight: 700; margin-top: 2px;}
.nb-tile-sub {font-size: 12px; color: var(--body-text-color-subdued);}
.nb-note {margin-top: 10px; font-size: 13px; color: #B45309;}
.nb-offline {font-size: 11px; background: rgba(100,116,139,.18); padding: 2px 8px; border-radius: 999px;}
footer {display: none !important;}
"""

llm_badge = (f"LLM online · {llm.MODEL}" if llm.available() else "LLM offline · templated answers")
if METRICS:
    seg = METRICS["segmentation"]
    metric_badge = (f"Test Dice {seg['per_patient_volume_dice_mean']:.2f} (per patient) · "
                    f"{METRICS['test_patients']} held-out patients")
else:
    metric_badge = "Run evaluate.py for test metrics"

HEADER = f"""
<div id="nb-header">
  <h1>🧠 NeuroBot</h1>
  <p>AI brain-MRI abnormality segmentation with a retrieval-augmented consultation assistant.</p>
  <div class="nb-badges">
    <span class="nb-badge">U-Net · VGG-19 encoder</span>
    <span class="nb-badge">{metric_badge}</span>
    <span class="nb-badge">RAG · MiniLM + FAISS</span>
    <span class="nb-badge">{llm_badge}</span>
  </div>
</div>
<div class="nb-disclaimer"><b>Research demo, not for clinical use.</b> Outputs are produced by a model
trained on one public dataset and must not be used for diagnosis or treatment decisions.</div>
"""


def about_md() -> str:
    rows = ""
    if METRICS:
        s, d = METRICS["segmentation"], METRICS["slice_detection"]
        rows = f"""
| Metric (held-out test set: {METRICS['test_patients']} patients, {METRICS['test_slices']} slices) | Value |
|---|---|
| Per-patient volumetric Dice (mean / median) | {s['per_patient_volume_dice_mean']:.3f} / {s['per_patient_volume_dice_median']:.3f} |
| Per-patient volumetric IoU (mean) | {s['per_patient_volume_iou_mean']:.3f} |
| Dice on tumour slices (mean) | {s['tumour_slice_dice_mean']:.3f} |
| Global pixel Dice / IoU | {s['global_pixel_dice']:.3f} / {s['global_pixel_iou']:.3f} |
| Slice detection sensitivity / specificity | {d['sensitivity']:.3f} / {d['specificity']:.3f} |
"""
    return f"""
### How it works
1. **Segmentation.** The uploaded slice (expected: the 3-channel pre-contrast / FLAIR / post-contrast
   composite used by the LGG dataset) is resized to {META.get('img_size', 128)} px and passed through a
   **U-Net with an ImageNet-pretrained VGG-19 encoder**. The probability map is upsampled to 256 px.
2. **Findings extraction.** Threshold 0.5, connected components < 40 px removed, then area %,
   hemisphere / quadrant (radiological convention), bounding box and confidence are computed.
3. **Report.** A deterministic, templated report summarises the findings and limitations.
4. **RAG consultation.** Your question + the findings are embedded with `all-MiniLM-L6-v2`; the top
   passages from a curated knowledge base (NCI, WHO CNS5, Radiopaedia, FDA, dataset papers) are retrieved
   with FAISS and given to an open-weight LLM on Groq (gpt-oss-120b) that must cite them. Without an API key NeuroBot returns
   the retrieved passages with a templated summary.

### Results
{rows or '_No metrics yet: run `python evaluate.py`._'}

### Data
LGG Segmentation Dataset: Buda, Saha & Mazurowski (2019), *Computers in Biology and Medicine* 109:218-225,
110 TCGA-LGG patients from The Cancer Imaging Archive. Split by **patient** (70/15/15).
"""


with gr.Blocks(title="NeuroBot · Brain MRI AI") as demo:
    scan_state = gr.State(None)
    gr.HTML(HEADER)
    with gr.Tabs():
        with gr.Tab("🔬 Analyze scan"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300):
                    inp = gr.Image(type="pil", label="MRI slice", height=300,
                                   sources=["upload", "clipboard"])
                    btn = gr.Button("Analyze scan", variant="primary", size="lg")
                    ex_box = gr.Column()
                with gr.Column(scale=8):
                    with gr.Row():
                        out_orig = gr.Image(label="Original", height=260, interactive=False)
                        out_overlay = gr.Image(label="Overlay", height=260, interactive=False)
                        out_mask = gr.Image(label="Predicted mask", height=260, interactive=False)
                    card = gr.HTML(EMPTY_CARD)
            with gr.Accordion("📝 Auto-generated report", open=True):
                report_md = gr.Markdown("_The report appears here after analysis._")
                report_file = gr.File(label="Download report (.md)", interactive=False)
        with gr.Tab("💬 Ask NeuroBot about this scan"):
            scan_banner = gr.Markdown("**Current scan:** none yet. Analyse a scan first for "
                                      "scan-specific answers.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    bot = gr.Chatbot(height=430, label="NeuroBot", buttons=["copy"],
                                     placeholder="Ask about the current scan, MRI sequences, next steps or "
                                                 "lower-grade glioma. Answers cite the knowledge base.")
                    with gr.Row():
                        msg = gr.Textbox(placeholder="e.g. What does a left anterior FLAIR lesion mean?",
                                         show_label=False, scale=8, container=False)
                        send = gr.Button("Ask", variant="primary", scale=1)
                    gr.Examples(examples=["What does this finding mean?",
                                          "Which MRI sequences should the radiologist review next?",
                                          "What are the usual next steps after a scan like this?",
                                          "What treatments exist for lower-grade glioma?",
                                          "How reliable is this model?"],
                                inputs=msg, label="Suggested questions")
                with gr.Column(scale=5):
                    gr.Markdown("#### 📚 Sources retrieved")
                    sources = gr.Markdown("_Retrieved knowledge-base passages appear here._")
        with gr.Tab("ℹ️ About / How it works"):
            gr.Markdown(about_md())

    outs = [out_orig, out_overlay, out_mask, card, report_md, report_file, scan_state, scan_banner, bot]
    with ex_box:
        gr.Examples(examples=[str(p) for p in EXAMPLES], inputs=inp, outputs=outs, fn=analyze,
                    run_on_click=True, cache_examples=False,
                    label="Example slices (held-out test patients) · click to analyse")
    btn.click(analyze, inputs=inp, outputs=outs)
    inp.upload(analyze, inputs=inp, outputs=outs)
    send.click(chat, [msg, bot, scan_state], [bot, sources, msg])
    msg.submit(chat, [msg, bot, scan_state], [bot, sources, msg])


if __name__ == "__main__":
    demo.queue().launch(server_name=os.getenv("HOST", "127.0.0.1"),
                        server_port=int(os.getenv("PORT", "7860")), theme=theme, css=CSS)
