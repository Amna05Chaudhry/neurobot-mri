"""Retrieval-augmented consultation over the markdown knowledge base in kb/.

Chunks = paragraphs of each note (the note's title is prepended for context);
each chunk carries the note's source URLs. Embeddings: sentence-transformers
all-MiniLM-L6-v2, index: FAISS inner product on normalised vectors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import llm

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM = (
    "You are NeuroBot, an assistant that explains the output of a research brain-MRI "
    "segmentation model to doctors and patients. Answer ONLY from the scan findings and the "
    "numbered knowledge-base passages provided. Cite passages inline as [1], [2]. If the "
    "passages do not cover the question, say so. Never give a diagnosis, prognosis or "
    "treatment decision: recommend discussing with the treating clinician. Be concise "
    "(under 180 words) and use plain language."
)


@dataclass
class Chunk:
    doc: str
    title: str
    text: str
    sources: list[tuple[str, str]]


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        body, _, src = raw.partition("\nSources:")
        if not src:
            body, _, src = raw.partition("\nSource:")
        sources = []  # (short name, url) from lines like "- NCI, Some page: https://..."
        for line in src.splitlines():
            m = re.search(r"https?://\S+", line)
            if m:
                name = line[:m.start()].strip(" -:.").strip()
                sources.append((name if len(name) <= 70 else name[:67] + "...", m.group(0)))
        lines = body.strip().splitlines()
        title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else path.stem
        paras = [p.strip() for p in "\n".join(lines[1:]).split("\n\n") if p.strip()]
        merged: list[str] = []
        for p in paras:  # attach lead-in lines ("... include:") to the list that follows
            if merged and merged[-1].endswith(":"):
                merged[-1] += "\n" + p
            else:
                merged.append(p)
        for p in merged:
            chunks.append(Chunk(doc=path.name, title=title, text=" ".join(p.split()), sources=sources))
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.chunks = load_chunks(kb_dir)
        self.encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
        emb = self._embed([f"{c.title}. {c.text}" for c in self.chunks])
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        scores, ids = self.index.search(self._embed([query]), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{n}] ({c.title}) {c.text}" for n, (c, _) in enumerate(hits, 1))


def sources_markdown(hits: list[tuple[Chunk, float]]) -> str:
    out = []
    for n, (c, s) in enumerate(hits, 1):
        links = " · ".join(f"[{name or u.split('/')[2]}]({u})" for name, u in c.sources[:3])
        snippet = c.text if len(c.text) < 260 else c.text[:257] + "..."
        out.append(f"**[{n}] {c.title}** · `{c.doc}` · relevance {s:.2f}  \n{snippet}  \n{links}")
    return "\n\n".join(out)


def _fallback_answer(question: str, findings_text: str, hits: list[tuple[Chunk, float]]) -> str:
    lines = ["**Scan summary.** " + findings_text, "",
             "**Most relevant knowledge-base passages for your question:**"]
    for n, (c, _) in enumerate(hits[:3], 1):
        text = re.sub(r"(?:^|\s)\d+\.\s", " ", c.text).strip()  # drop "1. 2." list markers
        picked = ""
        for sent in re.split(r"(?<=[.!?:])\s", text):
            picked = f"{picked} {sent}".strip()
            if len(picked) >= 240:
                break
        lines.append(f"- {picked} [{n}]")
    lines += ["", "_LLM offline: this answer is assembled from retrieved passages without "
              "generation. Please discuss the scan with the treating clinician._"]
    return "\n".join(lines)


def answer(kb: KnowledgeBase, question: str, findings_text: str,
           history: list[dict] | None = None, k: int = 4) -> tuple[str, list[tuple[Chunk, float]], bool]:
    """Return (answer_markdown, retrieved hits, used_llm)."""
    hits = kb.search(question, k=k)  # findings go to the LLM, not the retriever (they swamp short questions)
    user = (f"Scan findings (from the AI model, single axial slice):\n{findings_text}\n\n"
            f"Knowledge-base passages:\n{format_context(hits)}\n\nQuestion: {question}")
    msgs = [{"role": "system", "content": SYSTEM}]
    for turn in (history or [])[-4:]:
        content = turn.get("content")
        if isinstance(content, list):  # Gradio 6 structured content
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if turn.get("role") in ("user", "assistant") and isinstance(content, str) and content:
            content = re.sub(r"<span class='nb-offline'>.*?</span>", "", content).strip()
            msgs.append({"role": turn["role"], "content": content})
    msgs.append({"role": "user", "content": user})
    reply = llm.chat(msgs)
    if reply:
        return reply, hits, True
    return _fallback_answer(question, findings_text, hits), hits, False
