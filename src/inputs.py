"""Question input: read problems from files dropped in `questions/`.

Supports .md, .txt, .html/.htm, .pdf and images. A *target* may be a file, a
directory (walked recursively), a glob, or a manifest listing paths -- and any
number of targets may be given at once, mixing formats freely:

    load(["questions/"])                      # everything, recursively
    load(["a.pdf", "shots/*.png", "b.md"])    # explicit mixed batch
    load(read_manifest("set.list"))           # a curated ordered list

A file may hold one problem or several -- numbered/headed blocks are split so a
single PDF of an exam becomes one problem per question. Identical statements
arriving from two sources (a photo and its transcript, say) are deduplicated.
"""

import glob as globmod
import hashlib
import os
import re

from . import config, vision

QUESTIONS_DIR = os.path.join(config.ROOT, "questions")
SUPPORTED = (".md", ".txt", ".html", ".htm", ".pdf") + vision.IMAGE_EXT

# Never walked. `.transcribed` holds this pipeline's own vision output -- reading
# it back would double every image-sourced problem. `archive/` is the convention
# for parking a question file you want kept but not run; name a file explicitly
# to load it from there anyway.
SKIP_DIRS = {".transcribed", "archive", "__pycache__", ".git", ".ipynb_checkpoints"}

# Explicit markers -- "Problem 3.", "Q2)", "## Question 4" -- are authoritative.
_EXPLICIT = re.compile(
    r"^[ \t]*(?:#{1,4}[ \t]*)?(?:Problem|Question|Exercise|Q)[ \t]*\.?[ \t]*(\d+)[ \t]*[.):]?[ \t]*",
    re.MULTILINE | re.IGNORECASE)
# A bare "3." is used ONLY when no explicit marker appears anywhere in the
# document: enumerated steps inside a single problem look exactly the same, and
# splitting on those shreds one problem into nonsense fragments.
_BARE = re.compile(r"^[ \t]*(?:#{1,4}[ \t]*)?(\d{1,2})[ \t]*[.)][ \t]+", re.MULTILINE)

_VISION_FAILED = "<!-- vision transcription failed"


def _natural_key(path: str) -> list:
    """Sort P2 before P10, and page 2 before page 10."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def iter_files(targets) -> tuple:
    """Expand targets (files, dirs, globs) into a de-duplicated file list.

    Directories are walked recursively, skipping SKIP_DIRS. Returns
    (files, empty_targets) -- a target yielding nothing is returned rather than
    dropped, so a typo in a batch is reported instead of silently costing a
    problem.
    """
    if isinstance(targets, str):
        targets = [targets]

    seen, files, empty = set(), [], []

    def add(path: str) -> bool:
        """Returns whether the path is a supported problem file (new or repeat)."""
        if not path.lower().endswith(SUPPORTED):
            return False
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            files.append(path)
        return True

    for target in targets or []:
        target = os.path.expanduser(target)
        matches = ([target] if os.path.exists(target)
                   else sorted(globmod.glob(target, recursive=True)))
        supported = False
        for match in matches:
            if os.path.isfile(match):
                supported |= add(match)
            elif os.path.isdir(match):
                found = []
                for root, dirs, names in os.walk(match):
                    dirs[:] = sorted(d for d in dirs
                                     if d not in SKIP_DIRS and not d.startswith("."))
                    found += [os.path.join(root, n) for n in names]
                for path in sorted(found, key=_natural_key):
                    supported |= add(path)
        if not supported:
            empty.append(target)
    return files, empty


def read_manifest(path: str) -> list:
    """Read a list of targets, one per line. `#` comments and blanks ignored.

    Relative paths resolve against the manifest's own directory, so a manifest
    travels with the question set it names.
    """
    base = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line if os.path.isabs(line) else os.path.join(base, line))
    return out


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return ""


def _read_html(path: str) -> str:
    raw = open(path, encoding="utf-8", errors="ignore").read()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        return soup.get_text("\n")
    except Exception:
        return re.sub(r"<[^>]+>", " ", raw)


def read_file(path: str) -> str:
    """Read a problem file. Images -- and scanned PDFs -- go through the vision model."""
    ext = os.path.splitext(path)[1].lower()
    if ext in vision.IMAGE_EXT:
        text = vision.transcribe_image(path)
    elif ext == ".pdf":
        text = _read_pdf(path)
        # A PDF with no extractable text is a scan: rasterize and transcribe.
        if len(text.strip()) < 40:
            text = vision.transcribe_pdf(path) or text
    elif ext in (".html", ".htm"):
        text = _read_html(path)
    else:
        text = open(path, encoding="utf-8", errors="ignore").read()
    return re.sub(r"[ \t]+", " ", text).strip()


def needs_vision(files: list) -> list:
    """Which of these files can only be read by the vision model."""
    out = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in vision.IMAGE_EXT:
            out.append(f)
        elif ext == ".pdf" and len(_read_pdf(f).strip()) < 40:
            out.append(f)
    return out


def split_problems(text: str, source: str = "") -> list:
    """Split a document into individual problems. Falls back to one whole-file problem."""
    stem = os.path.splitext(os.path.basename(source))[0]
    marks = list(_EXPLICIT.finditer(text)) or list(_BARE.finditer(text))
    if len(marks) < 2:
        body = text.strip()
        return [{"title": stem or "problem", "text": body}] if body else []

    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.start():end].strip()
        if len(body) < 20:            # heading with no content
            continue
        out.append({"title": f"{stem} Q{m.group(1)}" if stem else f"Q{m.group(1)}",
                    "text": body})
    return out


def _display(path: str) -> str:
    """Repo-relative where possible, absolute otherwise -- never `../../../tmp/...`."""
    full = os.path.abspath(path)
    root = config.ROOT + os.sep
    return os.path.relpath(full, config.ROOT) if full.startswith(root) else full


def _fingerprint(text: str) -> str:
    return hashlib.sha256(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()


def load(targets=None) -> list:
    """Load problems from any number of files, directories, or globs.

    Defaults to `questions/`. Unreadable inputs and failed transcriptions are
    reported and skipped rather than passed downstream as problem text.
    """
    if targets is None or targets == [] or targets == "":
        targets = [QUESTIONS_DIR]
    files, empty = iter_files(targets)
    for t in empty:
        print(f"WARNING: no supported problem file at: {t}")
    if not files:
        return []

    missing = needs_vision(files)
    if missing and not vision.available():
        print(f"WARNING: {len(missing)} file(s) need the vision model "
              f"({config.VISION_MODEL}), which is not pulled. They will be skipped.")
    elif missing:
        print(f"reading {len(missing)} image/scan file(s) with {config.VISION_MODEL} "
              "first, before any other model loads", flush=True)

    problems, seen = [], {}
    for f in files:
        rel = _display(f)
        text = read_file(f)
        if not text:
            print(f"  skipped (empty or unreadable): {rel}")
            continue
        if text.startswith(_VISION_FAILED):
            print(f"  skipped (transcription failed): {rel}")
            continue
        for p in split_problems(text, source=f):
            fp = _fingerprint(p["text"])
            if fp in seen:
                print(f"  skipped (duplicate of {seen[fp]}): {p['title']} [{rel}]")
                continue
            seen[fp] = p["title"]
            p["source"] = rel
            problems.append(p)
    if vision.USED[0]:
        # One-off pass: the 31 GB vision model would otherwise sit in memory
        # and evict the orchestrator or the prover the moment they load.
        vision.unload()
        print(f"vision model unloaded; {len(problems)} question(s) read", flush=True)
    return problems
