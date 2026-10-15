"""Dense retrieval over Mathlib. Replaces keyword matching for premise selection.

WHY THIS EXISTS
---------------
`mathlib_search.suggest_for_unknown` ranks declarations by keyword overlap
against a hand-built synonym table. That table had to be patched three times in
one session because English prose kept outranking mathematics:

    "positive"  -> pos      beat  `gcd`/`lcm`   on a gcd-lcm claim
    "squared"               beat  `coprime`     on a coprimality claim
    "two"/"twice"           beat  `det`/`Matrix` on a determinant claim

Each patch fixed one case and left the mechanism intact, because the mechanism
is the problem: lexical overlap does not know that "valuation" means
`factorization`, and no synonym list ever finishes. The agent asked for a
"p-adic valuation" lemma and got nothing, then invented `val` -- the failure
that motivated injecting premises up front in the first place.

Dense retrieval is what the field uses instead (LeanDojo/ReProver, REAL-Prover,
LeanSearch). A claim and the declaration that closes it land near each other in
embedding space even with no shared tokens.

DESIGN CONSTRAINTS, all forced by this machine
----------------------------------------------
* No torch. Base conda pins torch 1.12.1 and `transformers` needs >= 2.5, so
  sentence-transformers cannot import. Embeddings therefore come from ollama's
  /api/embed, which is already a hard dependency of this project.
* No faiss. 225,703 declarations x 768 dims is a 330 MB fp16 matrix; one
  brute-force matmul scores the whole corpus in ~50 ms. An ANN index would add
  a dependency and a build step to save milliseconds inside an operation whose
  neighbours cost minutes.
* fp16 on disk, fp32 for the matmul. Halves the file; numpy upcasts anyway.
"""

import json
import os
import re
import subprocess
import time

import numpy as np
import requests

from . import config

KB_DIR = os.path.join(config.ROOT, "kb")
CORPUS = os.path.join(KB_DIR, "corpus.jsonl")
VECTORS = os.path.join(KB_DIR, "vectors.npy")          # legacy single-model path
META = os.path.join(KB_DIR, "meta.json")

EMBED_MODEL = os.environ.get("KB_EMBED_MODEL", "nomic-embed-text")
# Retrieval is an ENSEMBLE over several embedders, measured rather than assumed.
# On a 3,008-declaration mini-index scoring by rank, the three disagree in
# useful ways:
#     det_mul            mxbai #1   nomic #11  bge  --
#     exists_isMaxOn     bge   #5   nomic --   mxbai --
#     dist_comm          bge   #5   nomic --   mxbai --
#     factorization_gcd  nomic #4   mxbai #13  bge  #18
# Individually 6/8, 7/8, 6/8; the UNION of their top-20s covers 8/8. That is
# the precondition RRF actually needs -- comparable strength, different failure
# directions -- and it is exactly what dense+BM25 did NOT have (3/8 vs 1/8),
# where fusion dropped `irrational_sqrt_two` from rank 1 to rank 13.
ENSEMBLE = [m for m in os.environ.get(
    "KB_EMBED_MODELS", "nomic-embed-text,bge-m3,mxbai-embed-large").split(",") if m.strip()]
# nomic-embed-text is trained with asymmetric task prefixes. Omitting them is a
# silent quality loss, not an error: a QUERY embedded as a DOCUMENT lands in the
# wrong neighbourhood and retrieval degrades with nothing in the logs to say so.
Q_PREFIX = "search_query: "
D_PREFIX = "search_document: "
# Only nomic is trained with asymmetric task prefixes; feeding them to a model
# that never saw them in training adds noise to every vector.
_PREFIXED = {"nomic-embed-text"}


def _prefixes(model: str) -> tuple:
    return (Q_PREFIX, D_PREFIX) if model.split(":")[0] in _PREFIXED else ("", "")


def _vec_path(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return os.path.join(KB_DIR, f"vectors__{safe}.npy")


def built_models() -> list:
    """Models with a usable index, in ENSEMBLE order."""
    out = [m for m in ENSEMBLE if os.path.exists(_vec_path(m))]
    if not out and os.path.exists(VECTORS):
        out = [EMBED_MODEL]                       # legacy single-index layout
    return out

MATHLIB_SRC = os.path.join(config.LEAN_PROJECT, ".lake", "packages", "mathlib", "Mathlib")
_PKGS = os.path.join(config.LEAN_PROJECT, ".lake", "packages")


def _toolchain_src() -> str:
    """Lean core source, from the toolchain this project pins."""
    try:
        with open(os.path.join(config.LEAN_PROJECT, "lean-toolchain"), encoding="utf-8") as fh:
            ver = fh.read().strip().replace(":", "---").replace("/", "--")
        return os.path.expanduser(f"~/.elan/toolchains/{ver}/src/lean")
    except OSError:
        return ""


def sources() -> list:
    """(root, module prefix) for every tree worth indexing.

    Mathlib alone is NOT enough, and the gap is not academic: `Nat.gcd_mul_lcm`
    -- the one-liner that closes a goal this system previously spent 1,190s
    failing -- lives in Lean CORE (Init/Data/Nat/Lcm.lean), not Mathlib. An
    index missing core is missing exactly the most fundamental lemmas, which
    are the ones most likely to close a goal outright.
    """
    core = _toolchain_src()
    out = []
    for root, prefix in (
            (MATHLIB_SRC, "Mathlib"),
            (os.path.join(core, "Init") if core else "", "Init"),
            (os.path.join(core, "Std") if core else "", "Std"),
            (os.path.join(_PKGS, "batteries", "Batteries"), "Batteries")):
        if root and os.path.isdir(root):
            out.append((root, prefix))
    return out

# theorem/lemma plus the type. Definitions matter too -- half of "what is this
# called in Mathlib" is a def, not a theorem.
_DECL = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|nonrec\s+|noncomputable\s+)?"
    r"(theorem|lemma|def|abbrev|structure|class) ([A-Za-z_][A-Za-z0-9_'.]*)")

_cache: dict = {}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

_NS_OPEN = re.compile(r"^namespace\s+([A-Za-z_][A-Za-z0-9_'.]*)")
_NS_END = re.compile(r"^end\b\s*([A-Za-z_][A-Za-z0-9_'.]*)?")
_SECTION = re.compile(r"^section\b")


def extract(limit: int = 0) -> list:
    """Every declaration as {name, kind, text, module}, FULLY QUALIFIED.

    Source text, not Lean's elaborated environment. The elaborated type is more
    precise, but retrieval matches against how a HUMAN would describe the
    lemma, and the source line is what a human wrote.

    The namespace stack is not a nicety. Mathlib writes `theorem gcd_mul_lcm`
    inside `namespace Nat`, and the real name is `Nat.gcd_mul_lcm`. A first
    build recorded the bare source names, which made retrieval look broken
    (recall 1/8) when it was in fact returning the right lemma at rank 1 with
    the wrong label -- and, far worse, would have fed the prover
    `gcd_mul_lcm`, an unknown identifier. Suggesting a name that does not
    resolve is the exact failure this module exists to prevent.
    """
    roots = sources()
    if not roots:
        return []
    out, seen = [], set()
    for src_root, prefix in roots:
      for root, _, files in os.walk(src_root):
        for fn in sorted(files):
            if not fn.endswith(".lean"):
                continue
            path = os.path.join(root, fn)
            module = prefix + "." + os.path.relpath(path, src_root).replace(os.sep, ".")[:-5]
            module = module.rstrip(".")
            stack = []                       # entries: name or None (section)
            in_comment = False
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    for raw in fh:
                        line = raw.rstrip("\n")
                        s = line.strip()
                        # Prose is not code. Doc comments discuss namespaces in
                        # English ("use a namespace instead."), and parsing
                        # those produced junk prefixes like
                        # `instead..interior_sInter` -- a name that cannot
                        # resolve, which is precisely what this module must
                        # never hand the prover.
                        if in_comment:
                            if "-/" in s:
                                in_comment = False
                            continue
                        if s.startswith("/-"):
                            if "-/" not in s:
                                in_comment = True
                            continue
                        if s.startswith("--") or not s:
                            continue
                        ns = _NS_OPEN.match(s)
                        if ns:
                            # `namespace A.B` pushes one frame spelling "A.B".
                            stack.append(ns.group(1).strip("."))
                            continue
                        if _SECTION.match(s):
                            stack.append(None)
                            continue
                        if _NS_END.match(s):
                            if stack:
                                stack.pop()
                            continue
                        m = _DECL.match(s)
                        if not m:
                            continue
                        bare = m.group(2)
                        prefix = ".".join(p for p in stack if p)
                        # A declaration written `theorem Nat.foo` is already
                        # qualified; do not double-prefix it.
                        name = bare if (not prefix or bare.startswith(prefix + ".")) \
                            else f"{prefix}.{bare}"
                        if name in seen:
                            continue
                        seen.add(name)
                        out.append({"name": name, "kind": m.group(1),
                                    "text": s[:500], "module": module})
                        if limit and len(out) >= limit:
                            return out
            except OSError:
                continue
    return out


def _embed_batch(texts: list, timeout: int = 300, model: str = "") -> list:
    r = requests.post(f"{config.OLLAMA_NATIVE}/api/embed",
                      json={"model": model or EMBED_MODEL, "input": texts},
                      timeout=timeout)
    r.raise_for_status()
    return r.json().get("embeddings", [])


def build(batch: int = 128, limit: int = 0, progress_every: int = 25,
          model: str = "") -> dict:
    """Extract, embed, and write the index for ONE model. Idempotent."""
    model = model or EMBED_MODEL
    qp, dp = _prefixes(model)
    os.makedirs(KB_DIR, exist_ok=True)
    t0 = time.time()
    decls = extract(limit=limit)
    if not decls:
        return {"error": f"no Mathlib source at {MATHLIB_SRC}"}
    print(f"[kb] {model}: extracted {len(decls):,} declarations "
          f"in {time.time() - t0:.1f}s", flush=True)

    # The embedded string is name + type together. The name carries Mathlib's
    # own vocabulary ("factorization"), the type carries the mathematics
    # ("gcd", "min"). Either alone loses half the signal.
    docs = [f"{dp}{d['name']} : {d['text']}" for d in decls]

    vecs, t1 = [], time.time()
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        try:
            vecs.extend(_embed_batch(chunk, model=model))
        except Exception as exc:
            print(f"[kb] batch at {i} failed ({exc}); retrying once", flush=True)
            try:
                vecs.extend(_embed_batch(chunk, model=model))
            except Exception:
                # Zero vectors never win a cosine ranking, so a lost batch
                # degrades recall rather than corrupting the index alignment --
                # which a silent skip WOULD do, shifting every later row.
                dims = len(vecs[0]) if vecs else 768
                vecs.extend([[0.0] * dims for _ in chunk])
        done = i + len(chunk)
        if (i // batch) % progress_every == 0 and i:
            rate = done / max(time.time() - t1, 1e-9)
            eta = (len(docs) - done) / max(rate, 1e-9)
            print(f"[kb] embedded {done:,}/{len(docs):,}  {rate:.0f}/s  eta {eta/60:.1f} min",
                  flush=True)

    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-9)          # pre-normalise: cosine -> dot
    path = _vec_path(model)
    np.save(path, arr.astype(np.float16))
    with open(CORPUS, "w", encoding="utf-8") as fh:
        for d in decls:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    info = {"count": len(decls), "dims": int(arr.shape[1]), "model": model,
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - t0, 1),
            "mb": round((os.path.getsize(path) + os.path.getsize(CORPUS)) / 1e6, 1)}
    meta = {}
    if os.path.exists(META):
        try:
            meta = json.load(open(META))
        except Exception:
            meta = {}
    if "models" not in meta:
        meta = {"models": {}}
    meta["models"][model] = info
    json.dump(meta, open(META, "w"), indent=1)
    print(f"[kb] {model}: built {info['count']:,} vectors, {info['mb']} MB, "
          f"{info['seconds']}s", flush=True)
    return info


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------

def available() -> bool:
    return bool(built_models()) and os.path.exists(CORPUS)


def _decls() -> list:
    if "decls" not in _cache:
        if not os.path.exists(CORPUS):
            return None
        with open(CORPUS, encoding="utf-8") as fh:
            _cache["decls"] = [json.loads(l) for l in fh]
    return _cache["decls"]


def _load(model: str = "") -> tuple:
    """(vectors, declarations) for one model. (None, None) when unbuilt."""
    model = model or (built_models() or [EMBED_MODEL])[0]
    key = f"vecs::{model}"
    if key not in _cache:
        path = _vec_path(model)
        if not os.path.exists(path):
            path = VECTORS if os.path.exists(VECTORS) else ""
        if not path or not os.path.exists(CORPUS):
            return None, None
        _cache[key] = np.load(path).astype(np.float32)
    decls = _decls()
    if decls is None:
        return None, None
    vecs = _cache[key]
    # A vector file built against a DIFFERENT corpus would mis-label every hit
    # -- row i would name a declaration that is no longer row i. Refuse rather
    # than return confidently wrong premises.
    if vecs.shape[0] != len(decls):
        return None, None
    return vecs, decls


def search(query: str, k: int = 12, kinds: tuple = (), model: str = "") -> list:
    """Top-k declarations by cosine similarity for ONE model."""
    model = model or (built_models() or [EMBED_MODEL])[0]
    vecs, decls = _load(model)
    if vecs is None or not query.strip():
        return []
    qp, _ = _prefixes(model)
    try:
        q = _embed_batch([f"{qp}{query.strip()}"], timeout=60, model=model)
    except Exception:
        return []
    if not q:
        return []
    qv = np.asarray(q[0], dtype=np.float32)
    if qv.shape[0] != vecs.shape[1]:
        return []
    qv /= max(float(np.linalg.norm(qv)), 1e-9)
    scores = vecs @ qv                                  # corpus is pre-normalised
    # argpartition over argsort: we need the top k of 225k, not a total order.
    take = min(max(k * 4, k), scores.shape[0])
    idx = np.argpartition(-scores, take - 1)[:take]
    idx = idx[np.argsort(-scores[idx])]
    out = []
    for i in idx:
        d = decls[int(i)]
        if kinds and d["kind"] not in kinds:
            continue
        out.append({**d, "score": round(float(scores[int(i)]), 4)})
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------
# lexical half of hybrid retrieval
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _tokens(text: str) -> list:
    """Split identifiers as a mathematician reads them.

    `Nat.factorization_gcd` must yield {nat, factorization, gcd} so a query
    saying "gcd" can reach it. camelCase splits too: `isCompact` -> {is,
    compact}.
    """
    out = []
    for raw in _TOKEN.findall(text or ""):
        parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", raw)
        out.extend(p.lower() for p in (parts or [raw]))
    return out


def _bm25_index() -> dict:
    """Postings + lengths for BM25. Built once per process from the corpus.

    Not persisted: building it is ~seconds over 266k short documents, and a
    stale postings file that disagrees with the vectors would be a silent
    correctness bug rather than a slow start.
    """
    if "bm25" in _cache:
        return _cache["bm25"]
    _, decls = _load()
    if decls is None:
        return {}
    postings: dict = {}
    lengths = np.zeros(len(decls), dtype=np.float32)
    for i, d in enumerate(decls):
        toks = _tokens(d["name"]) + _tokens(d["text"])
        lengths[i] = len(toks) or 1
        seen: dict = {}
        for t in toks:
            seen[t] = seen.get(t, 0) + 1
        for t, tf in seen.items():
            postings.setdefault(t, []).append((i, tf))
    _cache["bm25"] = {"postings": postings, "lengths": lengths,
                      "avg": float(lengths.mean()) if len(lengths) else 1.0,
                      "n": len(decls)}
    return _cache["bm25"]


# Concept -> module-path fragments. Retrieval degrades with corpus size, not
# with vocabulary: nomic scores 6/8 on 3k declarations and 3/8 on 265,774, and
# adding embedders did not help because at full scale they miss the SAME cases.
# Narrowing the candidate pool first puts ranking back in the regime where it
# demonstrably works.
#
# Measured on the lemma that blocked IMO26 P1 -- `Nat.factorization_gcd`, which
# Mathlib has and the agent ruled out twice because retrieval never surfaced it:
#     unfiltered  265,774 candidates  -> not in the top 8
#     filtered        861 candidates  -> RANK 1
# TOPIC hints narrow; TYPE hints do not. A first version mapped `nat`/`int` to
# the `Nat.`/`Int.` module prefixes, and since "positive integers" trips the
# `int` tag, a claim about p-adic valuations pulled in all 15,903 Int/Nat
# declarations -- a pool so large that ranking drowned exactly as it does on the
# full corpus, and the target stayed invisible. Type hints are therefore a
# FALLBACK, used only when no topic is recognised.
_MODULE_HINTS = {
    "divisibility": ("GCD", "Factorization", "PadicVal", "Prime", "Squarefree",
                     "Divisib", "Nat.Factorial"),
    "prime": ("Prime", "Factorization", "PadicVal", "Squarefree"),
    "matrix": ("Matrix", "LinearAlgebra", "Determinant"),
    "poly": ("Polynomial",),
    "finset": ("Finset", "Fintype", "Combinatorics", "BigOperators"),
    "measure": ("MeasureTheory", "Integral"),
    "complex": ("Complex", "SpecialFunctions"),
    "function": ("Calculus", "ContinuousOn", "Topology.Order", "Monotone", "Deriv"),
    "set": ("Data.Set", "Order.Bounds", "Filter"),
}
# Only consulted when nothing above matched -- coarse, but better than the full
# corpus when the claim genuinely is just "about the reals".
_TYPE_HINTS = {
    "real": ("Analysis", "Real", "Topology.MetricSpace", "Order.Field"),
    "rat": ("Rat",),
    "nat": ("Nat.",),
    "int": ("Int.",),
}


def _module_filter(query: str) -> list:
    """Module fragments implied by the claim's own concepts. [] = no narrowing."""
    from . import archive
    low = (query or "").lower()
    tags = {k for k, pat in archive._PROSE_OBJECTS.items() if re.search(pat, low)}
    frags: list = []
    for t in tags:
        frags.extend(_MODULE_HINTS.get(t, ()))
    if not frags:                       # no topic recognised -- fall back to type
        for t in tags:
            frags.extend(_TYPE_HINTS.get(t, ()))
    # Capitalised identifiers in the claim are strong module hints on their own
    # ("padicValNat" -> PadicVal) and cost nothing to add.
    for word in re.findall(r"\b[A-Z][A-Za-z]{3,}\b", query or ""):
        frags.append(word[:12])
    return sorted(set(frags))


def search_scoped(query: str, k: int = 12, model: str = "",
                  min_pool: int = 40, max_pool: int = 20000) -> list:
    """Rank within the modules the claim implies, falling back to the full corpus.

    The pool is only used when it is big enough to plausibly contain the answer
    and small enough for ranking to work. Outside that band this is exactly
    `search`, so the narrowing can never do worse than not narrowing -- it can
    only decline to help.
    """
    model = model or (built_models() or [EMBED_MODEL])[0]
    vecs, decls = _load(model)
    if vecs is None or not query.strip():
        return []
    frags = _module_filter(query)
    if not frags:
        return search(query, k=k, model=model)

    qp, _ = _prefixes(model)
    try:
        q = _embed_batch([f"{qp}{query.strip()}"], timeout=60, model=model)
    except Exception:
        return []
    if not q:
        return []
    qv = np.asarray(q[0], dtype=np.float32)
    if qv.shape[0] != vecs.shape[1]:
        return []
    qv /= max(float(np.linalg.norm(qv)), 1e-9)

    # ONE POOL PER FRAGMENT, fused by rank -- not a single merged pool.
    #
    # Merging is what failed: `Factorization`+`PadicVal` alone is 861
    # declarations and ranks `Nat.factorization_gcd` FIRST, but unioning it with
    # GCD/Prime/Divisib gives 3,029 and buries it again. Ranking quality is a
    # function of pool size, so each narrow pool is ranked on its own terms and
    # RRF combines them. This is the case fusion is actually for: several
    # retrievers of comparable strength failing in different directions --
    # unlike dense+BM25, where one was simply weaker and only added noise.
    runs, pools = [], []
    for frag in frags:
        idx = [i for i, d in enumerate(decls) if frag in d["module"]]
        if not (min_pool <= len(idx) <= max_pool):
            continue
        rows = np.asarray(idx, dtype=np.int64)
        scores = vecs[rows] @ qv
        take = min(max(k, 10), len(rows))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        runs.append([{**decls[int(rows[int(j)])],
                      "score": round(float(scores[int(j)]), 4),
                      "pool": len(rows), "via": frag} for j in top])
        pools.append(len(rows))
    # ALWAYS fuse with the unfiltered run. Scoping must be ADDITIVE.
    #
    # Replacing the full-corpus ranking measured WORSE: 2/8 against 3/8. It
    # gained `Nat.factorization_gcd` and lost `Nat.gcd_mul_lcm` and
    # `Nat.Coprime.pow`, because a filter that excludes the target makes it
    # unreachable no matter how good the ranking inside the pool is --
    # `Nat.gcd_mul_lcm` lives in `Init.Data.Nat.Lcm`, which no "GCD" fragment
    # matches. Fusing keeps every hit the broad search already had and adds the
    # ones only a narrow pool can surface.
    full = search(query, k=max(k, 20), model=model)
    if full:
        runs.append(full)
    if not runs:
        return search(query, k=k, model=model)
    if len(runs) == 1:
        return runs[0][:k]
    return _rrf(runs, k)


def search_lexical(query: str, k: int = 12) -> list:
    """BM25 over declaration names and types.

    Dense retrieval crosses vocabulary gaps but is fuzzy about EXACT names: a
    query naming `Nat.factorization_gcd` outright should return it at rank 1,
    and an embedding cosine will happily rank a dozen near-synonyms above it.
    The two methods fail in different directions, which is what makes fusing
    them worth more than tuning either.
    """
    idx = _bm25_index()
    if not idx:
        return []
    qt = _tokens(query)
    if not qt:
        return []
    k1, b, n = 1.5, 0.75, idx["n"]
    scores = np.zeros(n, dtype=np.float32)
    lengths, avg = idx["lengths"], idx["avg"]
    for t in set(qt):
        posts = idx["postings"].get(t)
        if not posts:
            continue
        df = len(posts)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        ids = np.fromiter((p[0] for p in posts), dtype=np.int64, count=df)
        tfs = np.fromiter((p[1] for p in posts), dtype=np.float32, count=df)
        denom = tfs + k1 * (1.0 - b + b * lengths[ids] / max(avg, 1e-9))
        scores[ids] += idf * (tfs * (k1 + 1.0)) / denom
    if not scores.any():
        return []
    take = min(k, n)
    top = np.argpartition(-scores, take - 1)[:take]
    top = top[np.argsort(-scores[top])]
    _, decls = _load()
    return [{**decls[int(i)], "score": round(float(scores[int(i)]), 4)}
            for i in top if scores[int(i)] > 0]


def _rrf(runs: list, k: int, weights: list = None) -> list:
    """Reciprocal-rank fusion over several ranked runs.

    RRF rather than score normalisation: cosine (bounded, model-specific scale)
    and BM25 (unbounded) cannot be compared numerically without a tuning
    constant that will be wrong on the next corpus. Rank is the one thing every
    retriever agrees on.
    """
    C = 60.0                      # standard damping; rank 1 is not 60x rank 60
    weights = weights or [1.0] * len(runs)
    fused: dict = {}
    for run, w in zip(runs, weights):
        for rank, hit in enumerate(run, 1):
            slot = fused.setdefault(hit["name"], {"hit": hit, "rrf": 0.0})
            slot["rrf"] += w / (C + rank)
    out = sorted(fused.values(), key=lambda s: -s["rrf"])[:k]
    return [{**s["hit"], "rrf": round(s["rrf"], 5)} for s in out]


def search_ensemble(query: str, k: int = 12, depth: int = 40) -> list:
    """Fuse every built embedding model. The measured best retrieval path.

    Three embedders individually scored 6/8, 7/8, 6/8 on the failure cases and
    missed DIFFERENT ones; the union of their top-20s covered 8/8. Fusing
    retrievers that fail in the same direction (dense + weak BM25) achieved
    nothing and cost rank-1 hits -- fusing ones that fail differently is the
    case RRF is actually for.
    """
    models = built_models()
    if not models:
        return []
    runs = [r for r in (search(query, k=depth, model=m) for m in models) if r]
    if not runs:
        return []
    return _rrf(runs, k) if len(runs) > 1 else runs[0][:k]


def search_hybrid(query: str, k: int = 12, depth: int = 40,
                  lexical_weight: float = 0.5) -> list:
    """Ensemble of embedders, plus BM25 at reduced weight.

    Lexical is down-weighted rather than dropped: at full weight it pulled
    `irrational_sqrt_two` from rank 1 to rank 13, because it could not find the
    lemma at all and its votes only diluted the dense consensus. It still earns
    a seat -- an exact-name query is the one case dense retrieval is fuzzy on.
    """
    models = built_models()
    runs = [r for r in (search(query, k=depth, model=m) for m in models) if r]
    lex = search_lexical(query, k=depth)
    if not runs:
        return lex[:k]
    weights = [1.0] * len(runs)
    if lex:
        runs.append(lex)
        weights.append(lexical_weight)
    return _rrf(runs, k, weights)


def suggest(statement: str, bad_names: list = (), limit: int = 8) -> str:
    """Formatted premise block for the prompt. Mirrors mathlib_search.suggest_for_unknown."""
    # MEASURED: single-model dense beats every fusion tried, at full scale.
    #
    #   nomic alone   3/8   gcd_mul_lcm #3  irrational #1  pow #4
    #   bge-m3 alone  2/8   gcd_mul_lcm #5  irrational #1
    #   dense+BM25    3/8   irrational fell #1 -> #13
    #   two-model     3/8   irrational #2, pow #4 -> #19
    #
    # The mini-index promise did not survive scale. On 3,008 declarations the
    # three embedders scored 6/8, 7/8, 6/8 and missed DIFFERENT cases, so their
    # union covered 8/8; at 265,774 they miss the SAME cases -- bge's hits are
    # a strict subset of nomic's. Fusion needs retrievers that fail in
    # different directions, and at this scale they no longer do: distractors,
    # not vocabulary, are what beat them.
    #
    # Ranks matter here because this block shows only the top few, so a fusion
    # that keeps recall but pushes the right lemma from #1 to #19 is strictly
    # worse. search_ensemble/search_hybrid stay available for when a candidate
    # FILTER (by module or domain) makes the corpus small enough for
    # complementarity to reappear -- that is the real fix for the scale
    # problem, not more embedders.
    # MEASURED, and the scoped variant did NOT win. Full tally at recall@20
    # over the eight real failure cases:
    #
    #   dense (plain)        3/8   irrational #1, gcd_mul_lcm #3, pow #4
    #   dense + BM25         3/8   irrational fell #1 -> #13
    #   two embedders        3/8   pow fell #4 -> #19
    #   scoped (replace)     2/8   gained factorization_gcd, LOST two
    #   scoped + full        2/8   fusing 8 pools drowns the broad run's hits
    #
    # Scoping does fix the case it was built for -- `Nat.factorization_gcd`,
    # the lemma that blocked IMO26 P1, goes from unfindable to rank 1 in an
    # 861-declaration pool. But a module filter can EXCLUDE the target
    # (`Nat.gcd_mul_lcm` lives in `Init.Data.Nat.Lcm`, which no GCD fragment
    # matches), and fusing many narrow pools crowds out the broad run: seven
    # pool leaders each outrank a broad-run hit at position 3.
    #
    # So plain dense stays. search_scoped remains for targeted use, and the
    # real fix is a filter that cannot drop the answer -- module hints learned
    # from proofs that actually worked, not hand-written by me.
    hits = search(statement, k=limit)
    if not hits:
        return ""
    lines = ["", "Relevant Mathlib declarations (dense retrieval, use the REAL names):"]
    for h in hits:
        lines.append(f"  {h['name']} : {h['text'][:150]}")
    for bad in list(bad_names)[:2]:
        alt = search(bad.replace(".", " ").replace("_", " "), k=3)
        if alt:
            lines.append(f"  instead of `{bad}` (does not exist): "
                         + ", ".join(a["name"] for a in alt))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mathlib dense-retrieval knowledge base")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--models", nargs="*", help="embedders to build (default: ENSEMBLE)")
    ap.add_argument("--limit", type=int, default=0, help="cap declarations (smoke test)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--query", help="search the built index")
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    if a.build:
        for m in (a.models or ENSEMBLE):
            print(json.dumps(build(batch=a.batch, limit=a.limit, model=m), indent=1))
    elif a.query:
        if not available():
            print("no index; run --build first")
        for h in search_ensemble(a.query, k=a.k):
            print(f"  {h['score']:.4f}  {h['name']:52s} {h['text'][:90]}")
    else:
        print(json.dumps({"built_models": built_models(),
                          "meta": json.load(open(META)) if os.path.exists(META) else {}},
                         indent=1))
