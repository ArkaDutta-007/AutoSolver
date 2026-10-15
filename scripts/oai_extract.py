#!/usr/bin/env python3
"""Mine OpenAI's Lean library for prover training pairs. Runs on the A6000 box.

WHERE THE DATA COMES FROM
-------------------------
github.com/openai/math (Apache-2.0) ships `lean/OAI`: 122k files, 2.2M
theorems, no `sorry`, 91% importing nothing but Mathlib and each other. Most
lemmas lean on definitions local to their paper, but a large minority state
something in plain Mathlib vocabulary -- exactly the kind of goal the prover is
handed. Those are the pairs worth having.

STANDALONE OR NOTHING
---------------------
A pair is kept only if, cut out of its file, it compiles in OUR environment:
PREAMBLE (byte-identical to src/prover.py) + the file's `open`s + the theorem,
against the Mathlib the Autosolver pins (not OpenAI's newer one). A lemma that
needs a local definition, a section `variable`, a renamed Mathlib lemma or a
local instance fails here and is dropped -- nobody has to judge that by eye.
`#print axioms` after each one rejects `sorryAx` and `native_decide`, and
catches a candidate a broken neighbour swallowed (no output at all).

    python3 oai_extract.py scan   --src ~/oaimath/math/lean --out cands.jsonl
    python3 oai_extract.py verify --env ~/oaimath/env --cands cands.jsonl --out pairs.jsonl
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
import tempfile

PREAMBLE = ("import Mathlib\n"
            "open scoped Matrix BigOperators Topology NNReal ENNReal Pointwise\n"
            "set_option maxHeartbeats 400000\n\n")
CORE = ("Mathlib", "Lean", "Std", "Batteries", "Aesop", "Qq", "Plausible", "Init")
AXIOMS_OK = {"propext", "Classical.choice", "Quot.sound"}

IMPORT = re.compile(r"^\s*(?:public\s+)?import\s+(?:all\s+)?([\w.]+)", re.M)
DECLARED = re.compile(r"\b(?:theorem|lemma|def|abbrev|structure|class|inductive|instance|"
                      r"opaque|axiom)\s+([^\s(:{\[⦃]+)")
HEAD = re.compile(r"^((?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|nonrec)\s+)*)"
                  r"(theorem|lemma)\s+([^\s(:{\[⦃]+)", re.S)
TOKEN = re.compile(r"[A-Za-z_Ͱ-Ͽἀ-῿][\w.'Ͱ-Ͽ₀-ₜ!?]*")
CMD = re.compile(r"(?:@\[|/--|/-!|namespace\b|section\b|end\b|open\b|variable\b|universe\b|"
                 r"set_option\b|attribute\b|theorem\b|lemma\b|def\b|abbrev\b|instance\b|"
                 r"structure\b|class\b|inductive\b|example\b|noncomputable\b|private\b|"
                 r"protected\b|notation\b|local\b|scoped\b|macro\b|syntax\b|elab\b|include\b|"
                 r"omit\b|#|irreducible_def\b|alias\b|deriving\b|mutual\b|initialize\b|opaque\b)")
OPENERS, CLOSERS = "([{⟨⦃⟦", ")]}⟩⦄⟧"


def strip_comments(t):
    """Blank out comments (nested `/- -/` and `--`), keeping docstrings' text out of the
    way of the parser. Offsets are preserved so line structure survives."""
    out, i, n, depth = list(t), 0, len(t), 0
    while i < n:
        if t.startswith("/-", i):
            depth += 1
            out[i] = out[i + 1] = " "
            i += 2
            continue
        if depth and t.startswith("-/", i):
            depth -= 1
            out[i] = out[i + 1] = " "
            i += 2
            continue
        if depth:
            if t[i] != "\n":
                out[i] = " "
            i += 1
            continue
        if t.startswith("--", i):
            while i < n and t[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if t[i] == '"':                      # a string may hold "--" or "/-"
            i += 1
            while i < n and t[i] != '"':
                i += 2 if t[i] == "\\" else 1
        i += 1
    return "".join(out)


def split_assign(text):
    """(signature, proof) at the first `:=` outside brackets, or None."""
    depth = 0
    for i, ch in enumerate(text):
        if ch in OPENERS:
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
        elif ch == ":" and depth == 0 and text.startswith(":=", i):
            return text[:i].rstrip(), text[i + 2:].strip()
    return None


def commands(src):
    """Top-level commands as (code, docstring). Mathlib style puts every command at
    column 0 and indents continuations; a file that does not is mostly lost, and
    the compiler still has the last word on whatever is kept."""
    bare = strip_comments(src)
    lines, bare_lines = src.split("\n"), bare.split("\n")
    starts = [i for i, b in enumerate(bare_lines) if b[:1].strip() and CMD.match(b)]
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(lines)
        code = "\n".join(bare_lines[s:e]).rstrip()
        before = "\n".join(lines[starts[k - 1] if k else 0:s])
        m = re.search(r"/--((?:(?!-/).)*)-/\s*$", before, re.S)
        if code.strip():
            out.append((code, " ".join(m.group(1).split()) if m else ""))
    return out


def scan_file(path, mod, local):
    """Candidates from one file. `local` = names declared here and in its OAI imports."""
    src = open(path, encoding="utf-8", errors="replace").read()
    scopes = [{"kind": "file", "opens": [], "vars": False, "ns": ""}]
    pending_open = []
    found = []
    for code, doc in commands(src):
        head = code.lstrip()
        first = head.split(None, 1)[0] if head else ""
        rest = head[len(first):].strip()
        if first == "namespace":
            ns = ".".join(filter(None, [scopes[-1]["ns"], rest.split()[0] if rest else ""]))
            opens = [] if ns.startswith("OAI") else [f"open {ns}"]
            scopes.append({"kind": "namespace", "opens": opens, "vars": False, "ns": ns})
            continue
        if first == "section" or head.startswith("noncomputable section"):
            scopes.append({"kind": "section", "opens": [], "vars": False, "ns": scopes[-1]["ns"]})
            continue
        if first == "end":
            if len(scopes) > 1:
                scopes.pop()
            continue
        if first == "open":
            m_in = re.match(r"(open\s.*?)\s+in\b\s*", head, re.S)
            if not m_in:
                scopes[-1]["opens"].append(" ".join(head.split()))
                continue
            pending_open.append(" ".join(m_in.group(1).split()))
            head = head[m_in.end():]
            if not head:
                continue
            first = head.split(None, 1)[0]
        if first == "variable" or first in ("include", "omit"):
            scopes[-1]["vars"] = True
            pending_open = []
            continue
        m = HEAD.match(head)
        if not m:
            pending_open = []
            continue
        opens = [o for s in scopes for o in s["opens"]] + pending_open
        pending_open = []
        if any(s["vars"] for s in scopes):
            continue
        name = m.group(3)
        if name.startswith("OAI."):
            name = name.split(".")[-1]
        parts = split_assign(head[m.end():])
        if not parts:
            continue
        sig, proof = parts
        if not (80 <= len(proof) <= 3000 and len(sig) <= 1500):
            continue
        if re.search(r"\b(sorry|admit|native_decide)\b", proof):
            continue
        toks = set(TOKEN.findall(sig + " " + proof))
        if (toks | {p for t in toks for p in t.split(".")}) & local:
            continue
        found.append({"module": mod, "name": name, "opens": opens,
                      "sig": sig.strip(),
                      "proof": proof, "doc": doc})
    return found


def scan(a):
    root = os.path.expanduser(a.src)
    files = {}
    for d, _, fs in os.walk(os.path.join(root, "OAI")):
        for f in fs:
            if f.endswith(".lean"):
                p = os.path.join(d, f)
                files[os.path.relpath(p, root)[:-5].replace("/", ".")] = p
    imps, names = {}, {}
    for m, p in files.items():
        t = open(p, encoding="utf-8", errors="replace").read()
        imps[m] = IMPORT.findall(t)
        names[m] = {n.split(".")[-1] for n in DECLARED.findall(t)} | set(DECLARED.findall(t))
    closed = {}

    def ok(m, seen=()):
        if m not in closed:
            closed[m] = m not in seen and all(
                i.startswith(CORE) or (i in files and ok(i, seen + (m,))) for i in imps[m])
        return closed[m]

    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for m in sorted(files):
            if not ok(m):
                continue
            local = set(names[m]).union(*(names.get(i, set()) for i in imps[m]))
            for c in scan_file(files[m], m, local):
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} candidates from {sum(closed.values())} Mathlib-only files")


def minif2f(a):
    """The regression set, in candidate form: statement only, nothing to train on."""
    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for line in open(a.src, encoding="utf-8"):
            r = json.loads(line)
            if r.get("split") != "test":
                continue
            m = re.match(r"\s*theorem\s+(\S+)\s+(.*?)\s*:=\s*by\s*(sorry)?\s*$",
                         r["formal_statement"], re.S)
            opens = re.findall(r"^open .*$", r.get("header", ""), re.M)
            fh.write(json.dumps({"module": "miniF2F.test", "name": m.group(1), "opens": opens,
                                 "sig": m.group(2), "proof": "by sorry", "doc": ""},
                                ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} miniF2F test statements")


def program(c, tag, sorry=False):
    """(source, declared name): the candidate as the Autosolver would compile it,
    renamed so a batch can hold two lemmas that share a short name. Helper lemmas
    (`prefix`, competition proofs only) ride along inside their own namespace."""
    proof = "by sorry" if sorry else c["proof"]
    if c.get("prefix") and not sorry:
        opens = "".join(f"{o}\n" for o in c["opens"])
        return (f"namespace C{tag}\n{opens}{c['prefix']}\n\n"
                f"theorem {c['name']}_{tag} {c['sig']} := {proof}\nend C{tag}\n",
                f"C{tag}.{c['name']}_{tag}")
    head = "".join(f"{o} in\n" for o in c["opens"])
    return f"{head}theorem {c['name']}_{tag} {c['sig']} := {proof}\n", f"{c['name']}_{tag}"


DEFINES = re.compile(r"^\s*(?:noncomputable\s+)?(?:def|abbrev|structure|class|instance|inductive)\b", re.M)


def parse_file(text, name=None):
    """The main theorem of a whole competition file: its opens, helper lemmas,
    signature and proof. None when the statement could lean on a local
    definition (the prompt carries the statement alone) or the file is scoped."""
    bare = strip_comments(text)
    if re.search(r"^\s*(namespace|section|variable|mutual)\b", bare, re.M):
        return None
    lines = [ln for ln in bare.split("\n") if not re.match(r"\s*(import|set_option)\b", ln)]
    opens = [" ".join(ln.split()) for ln in lines if ln.startswith("open ") and " in" not in ln]
    body = "\n".join(ln for ln in lines if not ln.startswith("open "))
    heads = list(re.finditer(r"^(?:theorem|lemma)\s+([^\s(:{\[⦃]+)", body, re.M))
    if not heads:
        return None
    m = next((h for h in heads if h.group(1) == name), None) if name else heads[-1]
    parts = split_assign(body[m.end():]) if m else None
    prefix = body[:m.start()].strip() if m else ""
    if not parts or DEFINES.search(prefix):
        return None
    sig, proof = parts
    if re.search(r"\b(sorry|admit|native_decide)\b", proof):
        return None
    return {"name": m.group(1), "opens": opens, "sig": sig.strip(),
            "proof": proof.strip(), "prefix": prefix, "doc": ""}


def _norm(s):
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def comp(a):
    """Competition problems: Lean Workbook (Goedel's proofs) and NuminaMath-LEAN.
    Proved ones go to --out for full verification, unproved ones to --out-stmts
    (statement check only; usable for plain sampling, never for hints). Anything
    matching a miniF2F test statement or problem text is dropped: miniF2F is the
    competition benchmark and must stay unseen."""
    import pyarrow.parquet as pq
    rng = random.Random(0)
    m2f_sig, m2f_txt = set(), set()
    for line in open(a.minif2f_src, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") == "test":
            m2f_sig.add(_norm(r["formal_statement"].split(" ", 2)[-1].split(":=")[0]))
            m2f_txt.add(_norm(r.get("informal_prefix", ""))[:160])
    clean = lambda c, txt="": (_norm(c["sig"]) not in m2f_sig
                               and (not txt or _norm(txt)[:160] not in m2f_txt))
    proved, stmts = [], []
    wb = pq.read_table(a.workbook).to_pylist()
    rng.shuffle(wb)
    for r in wb:
        c = parse_file(r["full_proof"])
        if c and clean(c) and len(proved) < a.limit_each:
            c["module"] = f"comp.workbook.{r['problem_id']}"
            proved.append(c)
    nm = pq.read_table(a.numina, columns=["uuid", "problem", "formal_statement", "formal_ground_truth",
                                          "ground_truth_type", "formal_proof", "rl_data"]).to_pylist()
    rng.shuffle(nm)
    n_proved = 0
    for r in nm:
        head = re.search(r"theorem\s+([^\s(:{\[⦃]+)", r["formal_statement"] or "")
        if not head:
            continue
        ref = (r["formal_ground_truth"] if r["ground_truth_type"] == "complete" else "") or r["formal_proof"]
        if ref and n_proved < a.limit_each:
            c = parse_file(ref, head.group(1))
            if c and clean(c, r["problem"]):
                c["module"] = f"comp.numina.{r['uuid']}"
                proved.append(c)
                n_proved += 1
        elif not ref and len(stmts) < a.limit_each // 2 and 0 < (r["rl_data"] or {}).get("win_rate", 0) < 0.6:
            body = re.sub(r"^\s*import .*$", "", strip_comments(r["formal_statement"]), flags=re.M)
            parts = split_assign(body[body.find(head.group(0)) + len(head.group(0)):])
            if parts and clean({"sig": parts[0]}, r["problem"]):
                opens = [" ".join(ln.split()) for ln in body.split("\n") if ln.startswith("open ")]
                stmts.append({"module": f"comp.numina.{r['uuid']}", "name": head.group(1),
                              "opens": opens, "sig": parts[0].strip(), "proof": "by sorry",
                              "prefix": "", "doc": ""})
    for path, rows in ((a.out, proved), (a.out_stmts, stmts)):
        with open(path, "w", encoding="utf-8") as fh:
            for c in rows:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(proved)} proved competition problems ({n_proved} NuminaMath), "
          f"{len(stmts)} statement-only")


SIMPLE_OPEN = re.compile(r"open(\s+scoped)?((?:\s+[^\s()]+)+)")


def open_units(o):
    """`open A B` -> `open A`, `open B`: one bad name must not sink the rest."""
    m = SIMPLE_OPEN.fullmatch(o)
    if not m or re.search(r"\b(hiding|renaming)\b", o):
        return [o]
    kw = "open scoped" if m.group(1) else "open"
    return [f"{kw} {n}" for n in m.group(2).split()]


def usable_opens(cands, env, lean, timeout):
    """Which opens exist in Mathlib. A file inside `namespace OAI` opens its own
    sub-namespaces, which do not exist here and would fail every lemma after them."""
    units = sorted({u for c in cands for o in c["opens"] for u in open_units(o)})
    lines, at = PREAMBLE.rstrip("\n").split("\n"), {}
    for u in units:
        lines.append("section")
        lines.append(u)
        at[len(lines)] = u
        lines.append("end")
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    try:
        r = subprocess.run([lean, "--json", fh.name], capture_output=True, text=True,
                           timeout=timeout, env=env)
    finally:
        os.unlink(fh.name)
    bad = {at.get(json.loads(x)["pos"]["line"]) for x in r.stdout.splitlines()
           if x.startswith("{") and json.loads(x)["severity"] == "error"}
    print(f"opens: {len(units) - len(bad - {None})}/{len(units)} exist in Mathlib", flush=True)
    return set(units) - bad


def merge_opens(units):
    """At most `open A B ..` + `open scoped X ..` + the odd `hiding`: a prompt
    with twelve `open .. in` lines is not one the prover has ever seen."""
    plain = [u[5:] for u in units if u.startswith("open ") and SIMPLE_OPEN.fullmatch(u)
             and not u.startswith("open scoped ") and " " not in u[5:]]
    scoped = [u[12:] for u in units if u.startswith("open scoped ") and " " not in u[12:]]
    other = [u for u in units if u[5:] not in plain and u[12:] not in scoped]
    return ([f"open {' '.join(dict.fromkeys(plain))}"] if plain else []) + \
        ([f"open scoped {' '.join(dict.fromkeys(scoped))}"] if scoped else []) + \
        list(dict.fromkeys(other))


def check_batch(batch, env, lean, timeout, statements_only=False):
    """Compile a batch in one Lean process; return {id: ok}. With statements_only
    the proof is `sorry` and only the statement has to elaborate."""
    lines = PREAMBLE.rstrip("\n").split("\n") + ["set_option autoImplicit false", ""]
    spans = {}
    for c in batch:
        tag = f"c{c['id']}"
        start = len(lines) + 1
        src, declared = program(c, tag, sorry=statements_only)
        lines += src.split("\n")
        lines.append(f"#print axioms {declared}")
        spans[c["id"]] = (start, len(lines))
        lines.append("")
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    try:
        r = subprocess.run([lean, "--json", fh.name], capture_output=True, text=True,
                           timeout=timeout, env=env)
        msgs = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
    except subprocess.TimeoutExpired:
        return {c["id"]: False for c in batch}
    finally:
        os.unlink(fh.name)
    out = {}
    for c in batch:
        lo, hi = spans[c["id"]]
        mine = [m for m in msgs if lo <= m["pos"]["line"] <= hi]
        bad = any(m["severity"] == "error" for m in mine)
        ax = [m["data"] for m in mine if "depend" in m["data"] and f"_c{c['id']}'" in m["data"]]
        if bad or len(ax) != 1:
            out[c["id"]] = False
            continue
        used = set(re.findall(r"[\w.]+", ax[0].partition("axioms:")[2]))
        out[c["id"]] = used <= AXIOMS_OK | ({"sorryAx"} if statements_only else set())
    return out


def verify(a):
    env_dir = os.path.expanduser(a.env)
    path = os.environ["PATH"]
    run = lambda *cmd: subprocess.run(cmd, cwd=env_dir, capture_output=True, text=True,
                                      check=True).stdout.strip()
    env = dict(os.environ, LEAN_PATH=run("lake", "env", "printenv", "LEAN_PATH"), PATH=path)
    lean = run("lake", "env", "which", "lean")
    cands = [json.loads(x) for x in open(a.cands, encoding="utf-8")]
    random.Random(0).shuffle(cands)
    cands = cands[: a.limit] if a.limit else cands
    ok = usable_opens(cands, env, lean, a.timeout)
    for i, c in enumerate(cands):
        c["id"] = i
        c["opens"] = merge_opens([u for o in c["opens"] for u in open_units(o) if u in ok])
    batches = [cands[i:i + a.batch] for i in range(0, len(cands), a.batch)]
    kept = done = 0
    with cf.ThreadPoolExecutor(a.workers) as pool, open(a.out, "w", encoding="utf-8") as fh:
        for res, batch in zip(pool.map(lambda b: check_batch(b, env, lean, a.timeout,
                                                             a.statements_only), batches),
                              batches):
            for c in batch:
                done += 1
                if res.get(c["id"]):
                    kept += 1
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{done}/{len(cands)} checked, {kept} compile standalone", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--env", required=True)
    v.add_argument("--cands", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--batch", type=int, default=40)
    v.add_argument("--workers", type=int, default=20)
    v.add_argument("--timeout", type=int, default=1200)
    v.add_argument("--statements-only", action="store_true")
    k = sub.add_parser("comp")
    k.add_argument("--workbook", required=True)
    k.add_argument("--numina", required=True)
    k.add_argument("--minif2f-src", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--out-stmts", required=True)
    k.add_argument("--limit-each", type=int, default=6000)
    f = sub.add_parser("minif2f")
    f.add_argument("--src", required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    {"scan": scan, "verify": verify, "minif2f": minif2f, "comp": comp}[a.cmd](a)


if __name__ == "__main__":
    main()
