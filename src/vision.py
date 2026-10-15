"""Image -> markdown/LaTeX transcription via the local vision model.

Problems arrive as photos, screenshots, or scanned PDFs. muse-glimmer is
multimodal, so it transcribes them into markdown with LaTeX math, which then
flows through the ordinary text pipeline.

Transcription is CACHED to `questions/.transcribed/`. That is not just a speed
optimization: the vision model is 31 GB, and with the orchestrator and prover
already resident (36 GB against a 40 GB cap) loading it EVICTS one of them. So
this is designed as a one-off pass whose output is reused, never as an inline
step inside the research loop.
"""

import base64
import hashlib
import os

import requests

from . import config

CACHE_DIR = os.path.join(config.ROOT, "questions", ".transcribed")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")

PROMPT = (
    "Transcribe this image into clean markdown. Reproduce ALL mathematical "
    "notation faithfully as LaTeX ($...$ inline, $$...$$ display). Preserve "
    "problem numbering exactly as shown (e.g. 'Problem 1.', 'Q2)'). Transcribe "
    "only what is present -- do not solve anything, do not add commentary, do "
    "not omit any part of the text."
)


def _cache_path(path: str) -> str:
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, f"{base}-{h}.md")


#: set once this process has actually loaded the vision model
USED = [False]


def unload() -> None:
    """Hand the vision model's memory back before the orchestrator loads."""
    try:
        requests.post(f"{config.OLLAMA_NATIVE}/api/generate",
                      json={"model": config.VISION_MODEL, "keep_alive": 0}, timeout=60)
    except Exception:
        pass
    USED[0] = False


def transcribe_image(path: str, force: bool = False) -> str:
    """Transcribe one image. Cached by content hash."""
    cached = _cache_path(path)
    if not force and os.path.exists(cached):
        return open(cached, encoding="utf-8").read()
    USED[0] = True

    try:
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        r = requests.post(
            f"{config.OLLAMA_NATIVE}/api/chat",
            json={"model": config.VISION_MODEL,
                  "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
                  "stream": False,
                  "options": {"num_predict": config.VISION_MAX_TOKENS, "temperature": 0.1}},
            timeout=1800)
        r.raise_for_status()
        msg = r.json().get("message", {})
        # Thinking models put prose in `reasoning`; take content first.
        text = ((msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip())
    except Exception as exc:
        return f"<!-- vision transcription failed: {exc} -->"

    if text:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as fh:
            fh.write(f"<!-- transcribed from {os.path.basename(path)} by "
                     f"{config.VISION_MODEL} -->\n\n{text}\n")
    return text


def pdf_pages_to_images(pdf_path: str, dpi: int = 180, max_pages: int = 20) -> list:
    """Rasterize PDF pages. Used only when a PDF has no extractable text (scanned)."""
    try:
        import fitz
    except ImportError:
        return []
    out = []
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, page in enumerate(doc[:max_pages]):
            img = os.path.join(CACHE_DIR, f"{base}-p{i + 1}.png")
            if not os.path.exists(img):
                page.get_pixmap(dpi=dpi).save(img)
            out.append(img)
    except Exception:
        return []
    return out


def transcribe_pdf(pdf_path: str) -> str:
    """Transcribe a scanned PDF page by page."""
    pages = pdf_pages_to_images(pdf_path)
    if not pages:
        return ""
    return "\n\n".join(transcribe_image(p) for p in pages)


def available() -> bool:
    """Whether the vision model is actually pulled."""
    try:
        r = requests.get(f"{config.OLLAMA_NATIVE}/api/tags", timeout=20)
        names = {m.get("name", "") for m in r.json().get("models", [])}
        return config.VISION_MODEL in names
    except Exception:
        return False
