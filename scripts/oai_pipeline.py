#!/usr/bin/env python3
"""Mined problems -> a gated, merged Goedel-Prover-V2-32B, in one command.

    python oai_pipeline.py run    --profile lab   [--gpus 1,2] [--train-gpus 0,1,2] [--scale 0.5]
    python oai_pipeline.py status --profile lab
    python oai_pipeline.py stop   --profile lab
    python oai_pipeline.py run    --profile selftest      # CPU only: mock server, 2-layer model
    python oai_pipeline.py iterate --profile colab --rounds 3 --budget 110

Stages, each recorded in OUT/<stage>.done and skipped once done. A killed run
resumes where it stopped: samplers skip finished problems, training resumes
from its last saved adapter, and a time-boxed step keeps the clock it started.

  preflight   data, Lean, weights, disk; refuses if the GPUs are in use
  sample      Autosolver best-of-n prompt on research AND competition train problems
  hint        problems it never solved, with the reference proof shown (STaR)
  repair      the Autosolver's repair turn on its own compile failures
  eval_base   held-out research, held-out competition, miniF2F, and the repair
              turn on their failures -- bf16, the production server flags
  train       LoRA SFT (shortest verified answers) + preference pairs (against
              truncated, looping and repeated-error answers), merged into bf16
  eval_tuned  the same problems and repair contexts, same flags, merged weights
  gate        verdict.json + report.md: pass rates, not-concluding rates, speed

`iterate` is expert iteration on top of that run (round 1): round k samples
problems no round has tried with round k-1's tuned model, then retrains from the
base on every round's proofs (OUT/r2, OUT/r3, ...), and is scored on round 1's
held-out problems. A round starts only if the one before beat its predecessor
and it fits the hour budget with the final evaluation (OUT/final): the best
round against the base at 16 samples a problem, pass@1 to pass@10. One merged
copy of the weights is on disk at a time; any other is rebuilt from its adapter.

Data generation may run on an fp8 server (twice the KV room, so far more
samples per hour); every evaluation runs under the eval quantization (bf16 on
the lab, matching production). Nothing here deploys: the report says how.
"""

import argparse
import filecmp
import glob
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oai_sample as S  # noqa: E402

BASE = "Goedel-LM/Goedel-Prover-V2-32B"
HOURS = {"sample": 7, "hint": 2, "repair": 2, "eval": 4, "eval_repair": 1, "train": 9}
PROFILES = {
    "lab": dict(root="~/oaimath", gpus="1,2", train_gpus="", port=8011,
                python="~/venvs/train/bin/python", vllm="~/venvs/vllm-goedel/bin/vllm",
                env="~/oaimath/env", merged="", mock=False, gen_quant="fp8", eval_quant="",
                util=0.90, model_len=16384, n=4, inflight=16, lean_workers=24, limit=0,
                oai_per_family=6, eval_per_family=3, max_examples=1000, max_pairs=300,
                max_len=8192, train_args=[], bench_tokens=512, hours=HOURS,
                server_env={"VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                            "VLLM_USE_FLASHINFER_SAMPLER": "0"},
                gen_kv="fp8", oai_part=0.5, per_problem=2,
                later=dict(hours={"sample": 20, "hint": 3, "repair": 3, "eval": 4, "eval_repair": 1,
                                  "train": 9},
                           oai_per_family=12, oai_part=2 / 3, per_problem=4, max_examples=5000,
                           max_pairs=1200, inflight=24),
                final_eval=dict(n=16, hours={"eval": 16, "eval_repair": 1})),
    # one A100 80GB / 96GB card: bf16 32B leaves room for ~2 sequences of KV, so
    # evaluation runs fp8 too (base and tuned alike); merged weights stay off Drive
    "colab": dict(root="/content/oaimath", gpus="0", train_gpus="", port=8011,
                  python=sys.executable, vllm="vllm", env="/content/oaimath/env",
                  merged="/content/merged", mock=False, gen_quant="fp8", eval_quant="fp8",
                  util=0.92, model_len=16384, n=4, inflight=12, limit=0,
                  lean_workers=max(4, min(24, (os.cpu_count() or 8) - 4)),
                  oai_per_family=4, eval_per_family=2, max_examples=700, max_pairs=200,
                  max_len=0, train_args=["--save-every", "25"], bench_tokens=512, server_env={},
                  hours={"sample": 4, "hint": 1, "repair": 1, "eval": 2, "eval_repair": 0.5,
                         "train": 4},
                  # rounds 2+: fp8 KV doubles the generator's room for sequences (falls back
                  # without it); OAI, where the model is weakest, gets 2/3 of it; up to 4
                  # distinct proofs a problem keep the variety pass@k feeds on
                  gen_kv="fp8", oai_part=0.5, per_problem=2,
                  later=dict(hours={"sample": 20, "hint": 3, "repair": 3, "eval": 3,
                                    "eval_repair": 0.75, "train": 7},
                             oai_per_family=12, oai_part=2 / 3, per_problem=4, max_examples=4000,
                             max_pairs=1000, inflight=16),
                  # the final evaluation tops round 1's 4 base samples a problem up to 16
                  final_eval=dict(n=16, hours={"eval": 13, "eval_repair": 1})),
    "selftest": dict(root="~/oaimath", gpus="", train_gpus="", port=8019,
                     python="~/venvs/train/bin/python", vllm="", env="~/oaimath/env",
                     merged="", mock=True, gen_quant="", eval_quant="", util=0, model_len=16384,
                     n=2, inflight=4, lean_workers=8, limit=3, oai_per_family=1,
                     eval_per_family=1, max_examples=40, max_pairs=10, max_len=2048,
                     train_args=["--accum", "4", "--save-every", "2"], bench_tokens=200,
                     server_env={}, hours={k: 0.25 for k in HOURS},
                     gen_kv="", oai_part=0.5, per_problem=2,
                     later=dict(oai_per_family=2, per_problem=4),
                     final_eval=dict(n=4, hours={"eval": 0.25, "eval_repair": 0.25})),
}
CHILDREN = []


class Cfg(argparse.Namespace):
    home = None             # round 1's directory: rounds 2+ and the final one live inside it
    round_no = 1
    is_final = False
    stages = None           # default STAGES
    gen_from = None         # train dir whose adapter generates (None: the base)
    tuned_from = None       # train dir whose adapter is "tuned" (None: OUT/train)
    data_dirs = None        # every round's directory so far (None: [OUT])

    def p(self, *parts):
        return os.path.join(self.out, *parts)

    def h(self, *parts):
        """Shared by every round: the server record, progress, the hour ledger."""
        return os.path.join(self.home or self.out, *parts)


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def jload(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def jdump(obj, path):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


# -- processes: only what this run recorded is ever signalled ------------------

def _started(pid):
    """Kernel start time of pid, so a recycled PID is never mistaken for ours."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _boot():
    try:
        with open("/proc/sys/kernel/random/boot_id") as fh:
            return fh.read().strip()
    except OSError:
        return None


def record(pid, pgid, **extra):
    return dict(extra, pid=pid, pgid=pgid, start=_started(pid), boot=_boot())


def group_alive(rec):
    if not rec or rec.get("boot", _boot()) != _boot():
        return False                                  # another machine or boot: a new Colab VM
    try:
        os.waitpid(rec["pid"], os.WNOHANG)            # reap it if it is our own exited child
    except ChildProcessError:
        pass
    now = _started(rec["pid"])
    if now is not None and now != rec["start"]:
        return False                                  # that PID now belongs to someone else
    try:
        os.killpg(rec["pgid"], 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_group(rec, grace=90):
    if not group_alive(rec):
        return
    os.killpg(rec["pgid"], signal.SIGTERM)
    for _ in range(grace):
        time.sleep(1)
        if not group_alive(rec):
            return
    os.killpg(rec["pgid"], signal.SIGKILL)


def refuse_busy(c, gpus):
    """Shared box: never start on a GPU someone else is using."""
    if not gpus or c.mock:
        return
    for attempt in range(3):                          # our own server may still be letting go
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        used = {int(i): int(m) for i, m in (ln.split(",") for ln in out.strip().splitlines())}
        busy = {g: used.get(g) for g in map(int, gpus.split(",")) if used.get(g, 0) > 2000}
        if not busy:
            return
        time.sleep(15)
    raise SystemExit(f"GPUs in use (MiB): {busy} -- nothing started. Free them or pass --gpus.")


#: A stage can run for hours with its workers writing only to logs/, and a
#: Colab cell shows nothing but what this process prints -- three silent hours
#: read as a hang. So while it waits it says, this often, where each one is.
BEAT_S = 300          # the cell's heartbeat: a progress block every 5 minutes
STEP_RE = re.compile(r"step (\d+)/(\d+) loss ([\d.]+).*?(\d+) tok/s")
PROB_RE = re.compile(r"(\d+)/(\d+) problems, (\d+)/(\d+) samples verified")
STAGE = {"name": "", "t0": 0.0}    # the stage run() is in, and since when


def tail(path, n):
    """The last n lines of a log (from its last 16 KB); None when there is no log."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16384))
            lines = fh.read().decode("utf-8", "replace").strip().splitlines()
        return [ln.rstrip()[:300] for ln in lines[-n:]]
    except OSError:
        return None


def last_line(path):
    t = tail(path, 1)
    return "(no log yet)" if t is None else t[-1][:200] if t else "(nothing logged yet)"


def bar(done, total, width=20):
    f = min(1.0, done / total) if total else 0.0
    n = round(f * width)
    return f"[{'█' * n}{'░' * (width - n)}] {f:4.0%}"


def hours(h):
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def progress(path, pace):
    """One worker, read from its log: a bar and its numbers when it prints them
    (the sampler, the trainer), else its last line. `pace` keeps the first count
    seen and when, so time left comes from this run's own speed."""
    lines = tail(path, 40)
    if lines is None:
        return "(no log yet)"
    for ln in reversed(lines):
        if ln.startswith("saved"):       # the merge that follows prints no counter
            return "training finished; merging the adapter into the 32B weights (~15 min, no counter)"
        m = STEP_RE.search(ln) or PROB_RE.search(ln)
        if not m:
            continue
        i, n = int(m[1]), int(m[2])
        i0, t0 = pace.setdefault(path, (i, time.time()))
        left = (f" · ~{hours((time.time() - t0) / (i - i0) * (n - i) / 3600)} left at this pace"
                if i > i0 else "")
        if m.re is STEP_RE:
            return f"{bar(i, n)} step {i}/{n} · loss {m[3]} · {m[4]} tok/s{left}"
        k, t = int(m[3]), int(m[4])
        return f"{bar(i, n)} {i}/{n} problems · {k}/{t} proofs verified ({k / max(t, 1):.0%}){left}"
    return (lines[-1] if lines else "")[:200] or "(nothing logged yet)"


def overview(c):
    """The whole run: each stage done (✓), running (▶) or to come (·), and the budget left."""
    names = [n for n, _ in c.stages or STAGES]
    spent = (time.time() - STAGE["t0"]) / 3600
    marks = " ".join(("✓" if os.path.exists(c.p(f"{n}.done")) else "▶" if n == STAGE["name"] else "·") + n
                     for n in names)
    where = "final evaluation, " if c.is_final else f"round {c.round_no}, " if c.home else ""
    return [f" · {where}stage {names.index(STAGE['name']) + 1}/{len(names)} {STAGE['name']} ({hours(spent)} in)"
            f" · ~{hours(max(0.0, eta(c) - spent))} of budget left in the run", "  " + marks]


def gpu_line():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
        return "  " + " · ".join(f"GPU {i}: {u}% busy, {int(m) / 1024:.0f}/{int(t) / 1024:.0f} GB"
                                 for i, u, m, t in (map(str.strip, ln.split(","))
                                                    for ln in out.strip().splitlines()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""


def watch(c, named, beat=None):
    """Wait for every (name, Popen), logging a progress block every `beat`
    seconds (also written to progress.txt, to read from Drive on any device);
    returns the names that failed."""
    t0, pace = time.time(), {}
    while any(p.poll() is None for _, p in named):
        live = next(p for _, p in named if p.poll() is None)
        try:
            live.wait(timeout=beat or BEAT_S)
        except subprocess.TimeoutExpired:
            head, *rest = overview(c) if STAGE["name"] else [""]
            block = [f"── still working ({(time.time() - t0) / 60:.0f} min in){head}", *rest]
            block += [f"  {name}: " + ("finished" if p.poll() is not None
                                       else progress(c.p("logs", f"{name}.log"), pace))
                      for name, p in named]
            if STAGE["name"] and not getattr(c, "mock", False) and shutil.which("nvidia-smi"):
                block.append(gpu_line())
            for ln in filter(None, block):
                log(ln)
            try:
                with open(c.h("progress.txt"), "w", encoding="utf-8") as fh:
                    fh.write(time.strftime("%m-%d %H:%M:%S (VM clock, UTC)\n") + "\n".join(block) + "\n")
            except OSError:
                pass
    for name, p in named:            # the cell is all they see: say why, not just where
        if p.returncode:
            log(f"{name} failed (exit {p.returncode}); the end of its log:")
            for ln in tail(c.p("logs", f"{name}.log"), 40) or ["(no log)"]:
                log("  | " + ln)
    return [name for name, p in named if p.returncode]


def run_logged(c, name, cmd, env=None):
    with open(c.p("logs", f"{name}.log"), "a") as logf:
        p = subprocess.Popen([str(x) for x in cmd], stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=dict(os.environ, **(env or {})))
        CHILDREN.append(p)
        if watch(c, [(name, p)]):
            raise RuntimeError(f"{name} failed (exit {p.returncode}); see {c.p('logs', name + '.log')}")


# -- the server ----------------------------------------------------------------

def url(c):
    return f"http://127.0.0.1:{c.port}/v1"


def responding(c):
    try:
        with urllib.request.urlopen(url(c) + "/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def stop_server(c):
    rec = jload(c.h("server.json"))
    if rec:
        kill_group(rec)
        if os.path.exists(c.h("server.json")):
            os.remove(c.h("server.json"))
        log(f"server {rec['spec']} stopped")


def use_server(c, spec):
    """gen: the generator (the base in round 1, the last round's tuned model
    after) under the generation quantization and KV cache. base / tuned: base
    or merged weights under the eval quantization -- serve_goedel.sh's flags."""
    src = {"gen": c.gen_from, "tuned": c.tuned_from or c.p("train")}.get(spec)
    if src:
        ensure_merged(c, src)
    model = c.merged if src else c.model_dir
    quant = c.gen_quant if spec == "gen" else c.eval_quant
    kv = getattr(c, "gen_kv", "") if spec == "gen" else ""
    rec = jload(c.h("server.json"))
    if rec and (rec["model"], rec.get("asked"), rec.get("spec")) == (model, quant, spec) \
            and group_alive(rec) and responding(c):
        return rec                                    # same weights, same flags: keep it
    stop_server(c)
    if responding(c):
        raise SystemExit(f"port {c.port} answers, but not a server this run started; use --port")
    refuse_busy(c, c.gpus)
    for q, k in dict.fromkeys([(quant, kv), (quant, ""), ("", "")]):   # what will not start falls back
        if c.mock:
            cmd = [sys.executable, os.path.join(HERE, "oai_mock_vllm.py"), "--pairs", c.pairs_oai,
                   c.pairs_comp, c.pairs_m2f, "--port", str(c.port)] + (["--better"] * (spec == "tuned"))
        else:
            cmd = [c.vllm, "serve", model, "--served-model-name", BASE,
                   "--tensor-parallel-size", str(len(c.gpus.split(","))),
                   "--max-model-len", str(c.model_len), "--gpu-memory-utilization", str(c.util),
                   "--enable-prefix-caching", "--host", "127.0.0.1", "--port", str(c.port)] \
                + (["--quantization", q] if q else []) + (["--kv-cache-dtype", k] if k else [])
        logpath = c.p("logs", f"server-{spec}.log")
        with open(logpath, "a") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=c.gpus,
                                             CUDA_DEVICE_ORDER="PCI_BUS_ID", **c.server_env))
        rec = record(proc.pid, proc.pid, spec=spec, quant=q, kv=k, asked=quant, model=model)
        jdump(rec, c.h("server.json"))
        log(f"server {spec}: {model} {q or 'bf16'}{' kv ' + k if k else ''} on GPUs [{c.gpus}], "
            f"port {c.port}")
        t0 = time.time()
        while time.time() - t0 < 2400 and proc.poll() is None:
            if responding(c):
                log(f"server {spec} up after {time.time() - t0:.0f}s")
                return rec
            time.sleep(5)
        stop_server(c)
        with open(logpath, encoding="utf-8", errors="replace") as fh:
            log(f"server {spec} ({q or 'bf16'}) did not come up:\n" + "".join(fh.readlines()[-15:]))
    raise RuntimeError(f"no {spec} server; see {logpath}")


def bench(c, tag):
    """Decode tokens/s, one stream (how the Autosolver's repair loop calls it)
    and eight at once (best-of-n). Same prompt, fixed length, both models."""
    path = c.p(f"bench_{tag}.json")
    if jload(path):
        return jload(path)
    msgs = [{"role": "user", "content": S.prompt(
        "theorem bench (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 0 < a * b := by")}]

    def go(n):
        body = json.dumps({"model": BASE, "messages": msgs, "max_tokens": n, "temperature": 0,
                           "ignore_eos": True}).encode()
        t = time.time()
        req = urllib.request.Request(url(c) + "/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)["usage"]["completion_tokens"], time.time() - t

    go(32)
    tok, dt = go(c.bench_tokens)
    got = []
    threads = [threading.Thread(target=lambda: got.append(go(c.bench_tokens))) for _ in range(8)]
    t = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    out = {"single_tok_s": round(tok / dt, 1), "batch8_tok_s": round(sum(g[0] for g in got) / (time.time() - t), 1),
           "quant": (jload(c.h("server.json")) or {}).get("quant") or "bf16"}
    jdump(out, path)
    log(f"bench {tag}: {out}")
    return out


# -- sampling ------------------------------------------------------------------

def remaining(c, key, hours):
    """Hours left on a time-boxed step. The clock starts at its first run and
    survives restarts, so a resumed step does not get its time twice. Rounds 2+
    run on iterate's hour ledger: hours the VM was gone are not spent."""
    path = c.p("clock", key)
    if c.home:
        if not jload(path):
            jdump({"ledger": spent(c)}, path)
        return hours - (spent(c) - jload(path)["ledger"])
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(str(time.time()))
    with open(path) as fh:
        return hours - (time.time() - float(fh.read())) / 3600


def sampler(c, name, pairs, *args, part=1.0):
    """One oai_sample.py worker with `part` of the server's in-flight problems."""
    cmd = [c.python, os.path.join(HERE, "oai_sample.py"), "--pairs", pairs, "--out", c.p(f"{name}.jsonl"),
           "--base", url(c), "--model", BASE, "--model-len", c.model_len, "--env", c.env,
           "--inflight", max(1, round(c.inflight * part)),
           "--lean-workers", max(1, round(c.lean_workers * part))]
    cmd += (["--limit", c.limit] if c.limit else []) + list(args)
    logf = open(c.p("logs", f"{name}.log"), "a")
    p = subprocess.Popen([str(x) for x in cmd], stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
    CHILDREN.append(p)
    log(f"{name} started")
    return name, p


def wait(c, procs):
    failed = watch(c, procs)
    if failed:
        raise RuntimeError(f"failed: {failed}; see logs/")


def brief(path):
    if not os.path.exists(path):
        return {}
    s = S.summarize(path)
    s["solved"] = len(s["solved"])
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}


def pools(c):
    return [("oai", c.pairs_oai), ("comp", c.pairs_comp)]


def eval_sets(c):
    return [("oai", c.pairs_oai, ["--split", "test", "--per-family", c.eval_per_family]),
            ("comp", c.pairs_comp, ["--split", "test", "--per-family", 1]),
            ("minif2f", c.pairs_m2f, ["--split", "all"])]     # one module: never family-capped


# -- stages --------------------------------------------------------------------

def tiny_model(c):
    """Goedel's own config and tokenizer at 2 layers x width 64: the selftest's
    32B, small enough to train and merge on a CPU."""
    code = (
        "import torch\n"
        "from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer\n"
        f"cfg = AutoConfig.from_pretrained({BASE!r})\n"
        "cfg.update(dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128,\n"
        "                num_attention_heads=4, num_key_value_heads=2, head_dim=16))\n"
        "if getattr(cfg, 'layer_types', None): cfg.layer_types = cfg.layer_types[:2]\n"
        "torch.manual_seed(0)\n"
        f"AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained({c.model_dir!r})\n"
        f"AutoTokenizer.from_pretrained({BASE!r}).save_pretrained({c.model_dir!r})\n")
    run_logged(c, "tiny_model", [c.python, "-c", code], {"CUDA_VISIBLE_DEVICES": ""})


def preflight(c):
    refuse_busy(c, c.gpus)
    refuse_busy(c, c.train_gpus)
    for f in (c.pairs_oai, c.pairs_comp, c.pairs_m2f):
        if not os.path.exists(f) or not os.path.getsize(f):
            raise SystemExit(f"missing data: {f}")
    ok, why, err = S.Lean(c.env, 1).check(S.PREAMBLE + "theorem t : (2 : ℕ) + 2 = 4 := by norm_num",
                                          {"name": "t", "sig": ": (2 : ℕ) + 2 = 4", "opens": []})
    if not ok:
        raise SystemExit(f"Lean env {c.env} cannot check a known proof: {why} {err[:300]}")
    run_logged(c, "deps", [c.python, "-c", "import peft, transformers, torch, requests"])
    if not c.mock:
        run_logged(c, "deps", [c.vllm, "--version"])
    where = c.merged
    while not os.path.exists(where):
        where = os.path.dirname(where)
    held = sum(os.path.getsize(f) for f in glob.glob(os.path.join(c.merged, "*"))) / 2**30
    free = shutil.disk_usage(where).free / 2**30 + held          # a merged copy is replaced
    if free < (1 if c.mock else 75):
        raise SystemExit(f"{free:.0f} GiB free where merged weights go ({c.merged}); need 75")
    counts = {}
    for name, path in pools(c):
        ps = S.load(path)
        counts[name] = {"train": sum(S.split(p) == "train" for p in ps),
                        "test": sum(S.split(p) == "test" for p in ps)}
    counts["minif2f"] = sum(1 for _ in open(c.pairs_m2f, encoding="utf-8"))
    return {"data": counts, "model": c.model_dir, "eta_hours": eta(c)}


def stage_sample(c):
    use_server(c, "gen")
    h = remaining(c, "sample", c.hours["sample"] * c.scale)
    if h > 0:                                   # comp: 1 problem per family = shuffled
        wait(c, [sampler(c, f"plain_{pool}", pairs, "--split", "train", "--n", c.n, "--per-family",
                         c.oai_per_family if pool == "oai" else 1, "--hours", h,
                         *earlier(c, f"plain_{pool}.jsonl"),
                         part=c.oai_part if pool == "oai" else 1 - c.oai_part)
                 for pool, pairs in pools(c)])
    return {k: brief(c.p(f"{k}.jsonl")) for k in ("plain_oai", "plain_comp")}


def earlier(c, name):
    """--skip with that file from every earlier round: each round samples new problems."""
    files = [os.path.join(d, name) for d in (c.data_dirs or [c.out])[:-1]]
    files = [f for f in files if os.path.exists(f)]
    return ["--skip", *files] if files else []


def stage_hint(c):
    use_server(c, "gen")
    h = remaining(c, "hint", c.hours["hint"] * c.scale)
    procs = []
    for pool, pairs in pools(c):
        src = c.p(f"plain_{pool}.jsonl")
        if h <= 0 or not os.path.exists(src):
            continue
        tried = {r["pid"] for r in S.records(src)}
        zero = sorted(tried - set(S.summarize(src)["solved"]))
        with open(c.p(f"zero_{pool}.jsonl"), "w") as fh:
            fh.writelines(json.dumps({"pid": p}) + "\n" for p in zero)
        if zero:
            procs.append(sampler(c, f"hint_{pool}", pairs, "--split", "train", "--mode", "hint",
                                 "--n", 2, "--only", c.p(f"zero_{pool}.jsonl"), "--hours", h, part=1 / 2))
    wait(c, procs)
    return {f"hint_{p}": brief(c.p(f"hint_{p}.jsonl")) for p, _ in pools(c)}


def stage_repair(c):
    use_server(c, "gen")
    h = remaining(c, "repair", c.hours["repair"] * c.scale)
    wait(c, [sampler(c, f"repair_{pool}", pairs, "--split", "train", "--mode", "repair",
                  "--repair-from", c.p(f"plain_{pool}.jsonl"), "--n", 2, "--hours", h, part=1 / 2)
          for pool, pairs in pools(c) if h > 0 and os.path.exists(c.p(f"plain_{pool}.jsonl"))])
    return {f"repair_{p}": brief(c.p(f"repair_{p}.jsonl")) for p, _ in pools(c)}


def evaluate(c, tag):
    """Base: time-boxed, which fixes the problem set. Tuned: exactly those
    problems and those repair contexts, however long they take (3x cap). The
    final evaluation's base starts from round 1's samples and tops each problem
    up to n; there, both sides have budgets sized to finish."""
    use_server(c, tag)
    bench(c, tag)
    fixed = tag == "tuned" or c.is_final
    cap = 3 if tag == "tuned" and not c.is_final else 1
    for kind in ("", "_repair"):
        h = max(0.01, remaining(c, f"eval_{tag}{kind}", c.hours["eval" + kind] * c.scale * cap))
        procs = []
        for name, pairs, args in eval_sets(c):
            mode = (["--mode", "repair", "--repair-from", c.p(f"eval_base_{name}.jsonl"), "--n", 2]
                    if kind else ["--n", c.n])
            only = c.p(f"eval_base_{name}{kind}.jsonl")
            if fixed and not os.path.exists(only) or kind and not os.path.exists(mode[3]):
                continue
            procs.append(sampler(c, f"eval_{tag}_{name}{kind}", pairs, *args, *mode, "--hours", h,
                                 *(["--only", only] if fixed else []), part=1 / 3))
        wait(c, procs)
    return {f"{name}{kind}": brief(c.p(f"eval_{tag}_{name}{kind}.jsonl"))
            for name, _, _ in eval_sets(c) for kind in ("", "_repair")}


def tensors(d):
    """{name: (dtype, shape)} from safetensors headers -- no weights loaded."""
    out = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with open(f, "rb") as fh:
            head = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        out.update({k: (v["dtype"], v["shape"]) for k, v in head.items() if k != "__metadata__"})
    return out


def check_merged(c):
    """Same tensors (names, shapes, dtypes) and byte-identical config,
    tokenizer, chat template and generation defaults: the architecture vLLM
    runs is the base's, so decode speed cannot differ."""
    base, merged = tensors(c.model_dir), tensors(c.merged)
    files = [f for f in os.listdir(c.model_dir) if not f.endswith(".safetensors")
             and not f.startswith("model.safetensors") and os.path.isfile(os.path.join(c.model_dir, f))]
    differ = [f for f in files if not os.path.exists(os.path.join(c.merged, f)) or not filecmp.cmp(
        os.path.join(c.model_dir, f), os.path.join(c.merged, f), shallow=False)]
    out = {"same_tensors": bool(base) and base == merged, "tensors": [len(base), len(merged)],
           "differing_files": differ}
    if not out["same_tensors"] or differ:          # vLLM loading it is the hard check; the gate shows this
        log(f"WARNING merged weights are not the base's architecture byte for byte: {out}")
    return out


def merged_tag(train):
    """What a merged copy is of: the adapter's path and its last saved step."""
    return {"adapter": os.path.join(train, "adapter"),
            **(jload(os.path.join(train, "adapter", "state.json")) or {})}


def drop_merged(c):
    """Room for the next merge: the disk holds one 62 GB copy at a time."""
    stop_server(c)
    if os.path.exists(c.merged + ".json"):
        os.remove(c.merged + ".json")                 # the label goes first: no stale claim
    shutil.rmtree(c.merged, ignore_errors=True)


def ensure_merged(c, train):
    """c.merged holds train/adapter folded into the base, merging it now if it
    does not (a fresh VM, or another round's copy). c.merged + '.json' labels it."""
    if jload(c.merged + ".json") == merged_tag(train) and os.path.exists(os.path.join(c.merged, "config.json")):
        return
    drop_merged(c)
    refuse_busy(c, c.gpus)
    run_logged(c, "merge", [c.python, os.path.join(HERE, "train_prover.py"), "--merge",
                            "--out", train, "--base", c.model_dir, "--merged", c.merged],
               {"CUDA_VISIBLE_DEVICES": c.gpus, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    jdump(merged_tag(train), c.merged + ".json")
    check_merged(c)


def stage_train(c):
    """From the base, on every round's samples so far."""
    drop_merged(c)
    gpus = c.train_gpus or c.gpus
    refuse_busy(c, gpus)
    data = sorted(f for d in c.data_dirs or [c.out] for pat in ("plain_*", "hint_*", "repair_*")
                  for f in glob.glob(os.path.join(d, pat + ".jsonl")))
    if not data:
        raise SystemExit("no samples to train on")
    max_len = c.max_len or (8192 if gpu_mib(gpus) >= 90000 else 6144)
    run_logged(c, "train", [c.python, os.path.join(HERE, "train_prover.py"), "--data", *data,
                            "--out", c.p("train"), "--base", c.model_dir, "--merged", c.merged,
                            "--merge-after", "--max-examples", c.max_examples,
                            "--max-pairs", c.max_pairs, "--max-len", max_len,
                            "--per-problem", getattr(c, "per_problem", 2), *c.train_args],
               {"CUDA_VISIBLE_DEVICES": gpus, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    jdump(merged_tag(c.p("train")), c.merged + ".json")
    with open(c.p("logs", "train.log"), encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh if ln.startswith(("step ", "OOM"))]
    return dict(check_merged(c), steps=len([ln for ln in lines if ln.startswith("step")]),
                last=lines[-1].strip() if lines else "", max_len=max_len)


def gpu_mib(gpus):
    if not gpus:
        return 0
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    tot = {int(i): int(m) for i, m in (ln.split(",") for ln in out.strip().splitlines())}
    return min(tot.get(int(g), 0) for g in gpus.split(","))


def stage_eval_tuned(c):
    return evaluate(c, "tuned")       # use_server merges first if this VM lacks the copy


def by_pid(path):
    out = {}
    for r in S.records(path):
        out.setdefault(r["pid"], []).append(r)
    return out


KS = (1, 2, 4, 8, 10)


def pass_at(n, ok, k):
    """Chance that k of a problem's n samples, drawn without replacement,
    include one of its `ok` verified ones (the unbiased pass@k estimator)."""
    return 1.0 if n - ok < k else 1 - math.comb(n - ok, k) / math.comb(n, k)


def compare(bpath, tpath, need=1):
    """Paired over the problems both models have at least `need` samples of;
    pass@k for every k in KS that each of them has samples for."""
    b, t = by_pid(bpath), by_pid(tpath)
    common = sorted(p for p in set(b) & set(t) if min(len(b[p]), len(t[p])) >= need)
    if not common:
        return None
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    se = lambda d: statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
    ks = [k for k in KS if k <= min(len(g[p]) for g in (b, t) for p in common)]
    at = lambda g, p, k: pass_at(len(g[p]), sum(r["ok"] for r in g[p]), k)

    def side(g):
        rows = [r for p in common for r in g[p]]
        return {"pass1": mean([at(g, p, 1) for p in common]),
                "passn": mean([any(r["ok"] for r in g[p]) for p in common]),
                "passk": {k: mean([at(g, p, k) for p in common]) for k in ks},
                "truncated": mean([r["why"] == "truncated" for r in rows]),
                "looped": mean([bool(r.get("loop")) for r in rows]),
                "repeat": mean([r["why"] == "repeat" for r in rows]),
                "tokens": mean([r.get("ntok", 0) for r in rows])}

    dk = {k: [at(t, p, k) - at(b, p, k) for p in common] for k in ks}
    return {"problems": len(common), "base": side(b), "tuned": side(t), "delta": mean(dk[1]),
            "se": se(dk[1]), "k": ks[-1], "delta_k": mean(dk[ks[-1]]), "se_k": se(dk[ks[-1]]),
            "_dk": dk}


def score(c, d):
    """Picks the best round: tuned pass@k for k <= 4 (what a 4-sample
    evaluation measures), averaged over k and every held-out problem."""
    num = tot = 0
    for name, _, _ in eval_sets(c):
        v = compare(os.path.join(d, f"eval_base_{name}.jsonl"), os.path.join(d, f"eval_tuned_{name}.jsonl"))
        if v:
            pk = [x for k, x in v["tuned"]["passk"].items() if k <= 4]
            num, tot = num + v["problems"] * sum(pk) / len(pk), tot + v["problems"]
    return num / tot if tot else 0.0


def pooled(sets, k):
    d = [x for v in sets.values() for x in v["_dk"].get(k, [])]
    return sum(d) / max(len(d), 1), len(d)


def stage_gate(c):
    need = c.n if c.is_final else 1      # final: only problems with all 16 samples on both sides
    sets = {f"{n}{k}": compare(c.p(f"eval_base_{n}{k}.jsonl"), c.p(f"eval_tuned_{n}{k}.jsonl"),
                               1 if k else need)
            for n, _, _ in eval_sets(c) for k in ("", "_repair")}
    sets = {k: v for k, v in sets.items() if v}
    plain = {k: v for k, v in sets.items() if not k.endswith("_repair")}
    top = min((v["k"] for v in plain.values()), default=1)
    pool1, count = pooled(plain, 1)
    poolk, _ = pooled(plain, top)
    bb, bt = jload(c.p("bench_base.json")) or {}, jload(c.p("bench_tuned.json")) or {}
    speed = bt.get("single_tok_s", 0) / max(bb.get("single_tok_s", 0), 1e-9)
    checks = {f"{k}: pass@1 not significantly worse": v["delta"] >= -2 * v["se"] for k, v in plain.items()}
    checks["all held-out problems: pass@1 >= base"] = bool(count) and pool1 >= 0
    if top > 1:
        checks[f"all held-out problems: pass@{top} >= base"] = bool(count) and poolk >= 0
    checks["decode speed within 5% of base"] = speed >= 0.95
    tuned = c.tuned_from or c.p("train")
    tr = jload(os.path.join(os.path.dirname(tuned), "train.done")) or {}
    checks["merged: same tensors and files as the base"] = bool(tr.get("same_tensors")) \
        and not tr.get("differing_files")
    verdict = "pass" if all(checks.values()) else "fail"
    out = {"verdict": verdict, "checks": checks, "pooled_delta_pass1": pool1,
           f"pooled_delta_pass{top}": poolk, "speed_ratio": speed, "bench": {"base": bb, "tuned": bt},
           "sets": sets, "merged": c.merged, "adapter": os.path.join(tuned, "adapter"),
           "round": c.round_no, "final": c.is_final}
    if c.gen_from and not c.is_final:    # against the round before, on the same problems
        prev = os.path.dirname(c.gen_from)
        vs = {n: compare(os.path.join(prev, f"eval_tuned_{n}.jsonl"), c.p(f"eval_tuned_{n}.jsonl"))
              for n, _, _ in eval_sets(c)}
        vs = {k: v for k, v in vs.items() if v}
        out["vs_prev"] = {"round": c.round_no - 1, "pooled_delta_pass1": pooled(vs, 1)[0],
                          f"pooled_delta_pass{top}": pooled(vs, top)[0], "k": top,
                          "score": [score(c, prev), score(c, c.out)]}
    for v in sets.values():
        v.pop("_dk")
    jdump(out, c.p("verdict.json"))
    with open(c.p("report.md"), "w", encoding="utf-8") as fh:
        fh.write(report(c, out))
    log(f"VERDICT: {verdict.upper()} -- {c.p('report.md')}")
    return {"verdict": verdict, "pooled_delta_pass1": round(pool1, 4),
            f"pooled_delta_pass{top}": round(poolk, 4), "speed_ratio": round(speed, 3)}


def report(c, v):
    pct = lambda x: f"{100 * x:.1f}%"
    title = f"final evaluation, {c.n} samples a problem" if v["final"] else f"round {v['round']}"
    lines = [f"# Goedel-Prover-V2-32B tuning, {title}: {v['verdict'].upper()}", "",
             "| held-out set | problems | pass@1 base → tuned | Δ (±SE) | pass@k base → tuned "
             "| Δ pass@k (±SE) | truncated | looping | tokens/sample |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, s in v["sets"].items():
        if k.endswith("_repair"):
            continue
        b, t = s["base"], s["tuned"]
        ks = " · ".join(f"@{n} {pct(b['passk'][n])} → {pct(t['passk'][n])}" for n in b["passk"] if n > 1)
        lines.append(f"| {k} | {s['problems']} | {pct(b['pass1'])} → {pct(t['pass1'])} | "
                     f"{100 * s['delta']:+.1f} (±{100 * s['se']:.1f}) | {ks or '-'} | "
                     f"@{s['k']} {100 * s['delta_k']:+.1f} (±{100 * s['se_k']:.1f}) | "
                     f"{pct(b['truncated'])} → {pct(t['truncated'])} | {pct(b['looped'])} → {pct(t['looped'])} | "
                     f"{b['tokens']:.0f} → {t['tokens']:.0f} |")
    lines += ["", "Repair turn (the Autosolver's \"Lean rejected that, fix it\"), same failed contexts for both:", "",
              "| set | contexts | fixed base → tuned | same error again base → tuned |", "|---|---|---|---|"]
    for k, s in v["sets"].items():
        if k.endswith("_repair"):
            b, t = s["base"], s["tuned"]
            lines.append(f"| {k[:-7]} | {s['problems']} | {pct(b['pass1'])} → {pct(t['pass1'])} | "
                         f"{pct(b['repeat'])} → {pct(t['repeat'])} |")
    if v.get("vs_prev"):
        p, k = v["vs_prev"], v["vs_prev"]["k"]
        lines += ["", f"Against round {p['round']}'s tuned model, same problems: pooled pass@1 "
                  f"{100 * p['pooled_delta_pass1']:+.1f}, pass@{k} {100 * p[f'pooled_delta_pass{k}']:+.1f} "
                  f"points (pass@≤4 score {p['score'][0]:.3f} → {p['score'][1]:.3f})."]
    bb, bt = v["bench"]["base"], v["bench"]["tuned"]
    lines += ["", f"Decode speed ({bb.get('quant')}): one stream {bb.get('single_tok_s')} → "
              f"{bt.get('single_tok_s')} tok/s, eight at once {bb.get('batch8_tok_s')} → "
              f"{bt.get('batch8_tok_s')} tok/s.", "", "Checks:", ""]
    lines += [f"- [{'x' if ok else ' '}] {k}" for k, ok in v["checks"].items()]
    lines += ["", f"The adapter: {v['adapter']} (train_prover.py --merge --out <its train dir> folds it "
              "into the base).", "", "Nothing was deployed. To serve the tuned weights in place of the base "
              "(same name, so the Autosolver's settings do not change), restart the prover server with:", "",
              f"    vllm serve {v['merged']} --served-model-name {BASE} <the rest of serve_goedel.sh's flags>",
              "", "Reverting is restarting it with the base."]
    return "\n".join(lines) + "\n"


def stage_seed(c):
    """Rounds 2+ and the final evaluation: preflight, plus round 1's base
    evaluation copied in, so every round is scored on the same problems (and
    the final one tops those samples up to n)."""
    out = preflight(c)
    for f in [os.path.basename(f) for f in glob.glob(c.h("eval_base_*.jsonl"))] + ["bench_base.json"]:
        if not os.path.exists(c.p(f)) and os.path.exists(c.h(f)):
            shutil.copy(c.h(f), c.p(f + ".tmp"))
            os.replace(c.p(f + ".tmp"), c.p(f))
    return out


STAGES = [("preflight", preflight), ("sample", stage_sample), ("hint", stage_hint),
          ("repair", stage_repair), ("eval_base", lambda c: evaluate(c, "base")),
          ("train", stage_train), ("eval_tuned", stage_eval_tuned), ("gate", stage_gate)]
ROUND_STAGES = [("preflight", stage_seed), ("sample", stage_sample), ("hint", stage_hint),
                ("repair", stage_repair), ("train", stage_train), ("eval_tuned", stage_eval_tuned),
                ("gate", stage_gate)]
FINAL_STAGES = [("preflight", stage_seed), ("eval_base", lambda c: evaluate(c, "base")),
                ("eval_tuned", stage_eval_tuned), ("gate", stage_gate)]
BUDGET = {"sample": ["sample"], "hint": ["hint"], "repair": ["repair"], "train": ["train"],
          "eval_base": ["eval", "eval_repair"], "eval_tuned": ["eval", "eval_repair"]}


def eta(c):
    """Hours until the gate, from the stage budgets (training: an estimate)."""
    left = 0.0
    for name, _ in c.stages or STAGES:
        if not os.path.exists(c.p(f"{name}.done")):
            left += sum(c.hours[k] for k in BUDGET.get(name, [])) * c.scale + (0.3 if name in BUDGET else 0)
    return round(left, 1)


# -- commands ------------------------------------------------------------------

def recorded_model(c):
    """The weights dir model.json names, if this machine still has it. A fresh
    Colab VM has none of the last one's files: its cache path reached
    transformers as a repo id and failed, so a missing one is fetched again."""
    d = (jload(c.h("model.json")) or {}).get("dir") or ""
    return d if os.path.exists(os.path.join(d, "config.json")) else ""


def claim(c):
    rec = jload(c.p("pipeline.json"))
    if rec and rec["pid"] != os.getpid() and group_alive(rec) and _started(rec["pid"]):
        raise SystemExit(f"already running (pid {rec['pid']}); `status` or `stop` it")
    jdump(record(os.getpid(), os.getpgid(0), leader=os.getpgid(0) == os.getpid()), c.p("pipeline.json"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(129))


def run(c):
    for d in ("logs", "clock"):
        os.makedirs(c.p(d), exist_ok=True)
    claim(c)
    stages = c.stages or STAGES
    if not c.model_dir:
        if c.mock:
            c.model_dir = c.h("tiny")
        else:
            from huggingface_hub import snapshot_download   # cached on the lab, downloaded on Colab
            try:
                c.model_dir = snapshot_download(BASE, local_files_only=True)
            except Exception:
                where = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
                while not os.path.exists(where):
                    where = os.path.dirname(where)
                free = shutil.disk_usage(where).free / 2**30
                if free < 140:
                    raise SystemExit(f"{free:.0f} GiB free: the weights (62 GB) plus their merged "
                                     "copy (62 GB) need ~140. Pick a runtime with a bigger disk.")
                log("downloading the 32B weights (62 GB)")
                c.model_dir = snapshot_download(BASE)
        jdump({"dir": c.model_dir}, c.h("model.json"))
    if c.mock and not os.path.exists(os.path.join(c.model_dir, "config.json")):
        tiny_model(c)
    # Every VM is a new environment and a resumed run skips preflight, so this
    # runs on each start that still has training or merging ahead. Importing is
    # not training: peft vets torchao only when it builds a LoRA layer.
    if not all(os.path.exists(c.p(f"{n}.done")) for n, _ in stages if n in ("train", "eval_tuned")):
        log("checking this machine can train and merge (a tiny model, on the CPU)")
        run_logged(c, "train_smoke", [c.python, os.path.join(HERE, "train_prover.py"), "--smoke",
                                      "--base", c.model_dir], {"CUDA_VISIBLE_DEVICES": ""})
    log(f"run {c.out}: profile {c.profile}, GPUs [{c.gpus}] train [{c.train_gpus or c.gpus}], "
        f"~{eta(c)} h to the gate")
    try:
        for name, fn in stages:
            if os.path.exists(c.p(f"{name}.done")):
                continue
            with open(c.p("current"), "w") as fh:
                fh.write(name)
            log(f"== {name}")
            t0 = time.time()
            STAGE.update(name=name, t0=t0)
            out = fn(c) or {}
            out["hours"] = round((time.time() - t0) / 3600, 2)
            jdump(out, c.p(f"{name}.done"))
            log(f"{name} done: {json.dumps(out)[:600]}")
        os.remove(c.p("current"))
    finally:
        for p in CHILDREN:
            if p.poll() is None:
                p.terminate()
        stop_server(c)


def later(c, k):
    """Round k's settings: round 1's, overridden by the profile's `later`."""
    dirs = [c.out] + [os.path.join(c.out, f"r{i}") for i in range(2, k + 1)]
    r = Cfg(**dict(vars(c), **c.later))
    r.out, r.home, r.round_no, r.stages, r.data_dirs = dirs[-1], c.out, k, ROUND_STAGES, dirs
    r.gen_from, r.tuned_from = os.path.join(dirs[-2], "train"), os.path.join(dirs[-1], "train")
    return r


def final(c, best):
    r = Cfg(**dict(vars(c), **c.final_eval))
    r.out, r.home, r.is_final, r.stages = os.path.join(c.out, "final"), c.out, True, FINAL_STAGES
    r.tuned_from = os.path.join(best, "train")
    return r


def spent(c):
    return (jload(c.h("hours.json")) or {}).get("hours", 0.0)


def ledger(c):
    """Hours `iterate` has run, across VMs: wall clock, saved every minute (a VM
    that dies loses at most a minute of the count; setup before it is not in it)."""
    def tick():
        last = time.time()
        while True:
            time.sleep(60)
            now = time.time()
            jdump({"hours": round(spent(c) + (now - last) / 3600, 4)}, c.h("hours.json"))
            last = now
    threading.Thread(target=tick, daemon=True).start()


def iterate(c):
    """Round 1 if not done, rounds 2..c.rounds while each beats the one before
    and fits the budget with the final evaluation, then the final evaluation of
    the best round. Rerunning the same command resumes any of it."""
    claim(c)
    ledger(c)
    if not os.path.exists(c.p("gate.done")):
        run(c)
    pin = c.p("final", "best.json")
    done = [c.out]
    for k in range(2, c.rounds + 1):
        r = later(c, k)
        if not os.path.exists(r.p("preflight.done")):          # not begun: may it begin?
            need = eta(r) + eta(final(c, r.out))
            why = ("the final evaluation has begun" if os.path.exists(pin) else
                   f"round {k - 1} did not beat round {k - 2}" if k > 2
                   and score(c, done[-1]) <= score(c, done[-2]) else
                   f"{spent(c):.1f} h spent + ~{need:.1f} h for it and the final evaluation is over "
                   f"the {c.budget} h budget" if spent(c) + need > c.budget else "")
            if why:
                log(f"round {k} not started: {why}")
                break
        log(f"== round {k} ({spent(c):.1f} of {c.budget} h spent)")
        run(r)
        done.append(r.out)
    if not os.path.exists(pin):                       # pinned: a resume evaluates the same round
        os.makedirs(os.path.dirname(pin), exist_ok=True)
        best = max(done, key=lambda d: score(c, d))
        jdump({"dir": best, "scores": {d: score(c, d) for d in done}}, pin)
    best = jload(pin)["dir"]
    log(f"== final evaluation of {best} ({spent(c):.1f} of {c.budget} h spent)")
    run(final(c, best))
    log(f"all done in {spent(c):.1f} h: {c.p('final', 'report.md')}")


def status(c):
    rec = jload(c.p("pipeline.json"))
    up = group_alive(rec) and _started(rec["pid"]) is not None
    rounds = sorted(glob.glob(c.p("r[0-9]*")), key=lambda d: int(os.path.basename(d)[1:]))
    cfgs = [c] + [later(c, int(os.path.basename(d)[1:])) for d in rounds]
    if os.path.exists(c.p("final", "best.json")):
        cfgs.append(final(c, jload(c.p("final", "best.json"))["dir"]))
    print(f"{c.out}: {'RUNNING pid %s' % rec['pid'] if up else 'not running'}"
          + (f", {spent(c):.1f} h spent by iterate" if os.path.exists(c.p("hours.json")) else ""))
    for x in cfgs:
        cur = open(x.p("current")).read() if os.path.exists(x.p("current")) else ""
        print(f"{os.path.relpath(x.out, c.out) if x is not c else 'round 1'}: ~{eta(x)} h of budget left")
        for name, _ in x.stages or STAGES:
            d = jload(x.p(f"{name}.done"))
            state = (f"done ({d.pop('hours')} h)" if d else
                     "RUNNING" if up and name == cur else "stopped here" if name == cur else "pending")
            print(f"  {name:10} {state}" + (f"  {json.dumps(d)[:300]}" if d else ""))
    srv = jload(c.h("server.json"))
    if srv:
        print(f"  server: {srv['spec']} {srv['quant'] or 'bf16'} -- "
              f"{'up' if responding(c) else 'starting/down'} (pid {srv['pid']})")
    recent = sorted(glob.glob(cfgs[-1].p("logs", "*.log")), key=os.path.getmtime)[-3:]
    for path in recent:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tail = (fh.readlines() or [""])[-1].strip()
        print(f"  {os.path.basename(path)}: {tail[:160]}")
    done = [x for x in cfgs if os.path.exists(x.p("report.md"))]
    if done:
        print(open(done[-1].p("report.md"), encoding="utf-8").read())
    if not c.mock and shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout)


def stop(c):
    rec = jload(c.p("pipeline.json"))
    if rec and group_alive(rec) and _started(rec["pid"]):
        if rec.get("leader"):
            kill_group(rec)
        else:
            os.kill(rec["pid"], signal.SIGTERM)
        log(f"pipeline {rec['pid']} stopped")
    stop_server(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "iterate", "status", "stop"))
    ap.add_argument("--profile", default="lab", choices=sorted(PROFILES))
    ap.add_argument("--out", help="run directory (default ROOT/run, ROOT/selftest for the selftest)")
    ap.add_argument("--root")
    ap.add_argument("--gpus")
    ap.add_argument("--train-gpus")
    ap.add_argument("--port", type=int)
    ap.add_argument("--eval-quant", choices=("", "fp8"),
                    help="lab default '' (bf16, as production); fp8 doubles eval throughput")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every time budget")
    ap.add_argument("--rounds", type=int, default=3, help="iterate: rounds in all, round 1 included")
    ap.add_argument("--budget", type=float, default=110,
                    help="iterate: GPU hours for what it runs; a round that would not leave room "
                         "for the final evaluation is not started")
    a = ap.parse_args()
    c = Cfg(**PROFILES[a.profile])
    for k in ("root", "gpus", "train_gpus", "port", "eval_quant"):
        if getattr(a, k) is not None:
            setattr(c, k, getattr(a, k))
    x = lambda p: os.path.expanduser(p) if p else p
    c.rounds, c.budget = a.rounds, a.budget
    c.profile, c.scale, c.root, c.env, c.python, c.vllm = a.profile, a.scale, x(c.root), x(c.env), x(c.python), x(c.vllm)
    c.out = x(a.out) or os.path.join(c.root, "selftest" if c.mock else "run")
    c.pairs_oai, c.pairs_comp, c.pairs_m2f = (os.path.join(c.root, "data", f) for f in
                                              ("pairs.jsonl", "comp_pairs.jsonl", "minif2f.jsonl"))
    c.merged = x(c.merged) or c.p("merged")         # one copy, whichever round it is of
    for d in ("logs", "clock"):
        os.makedirs(c.p(d), exist_ok=True)
    c.model_dir = recorded_model(c)
    if os.path.isdir(os.path.expanduser("~/.elan/bin")):
        os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + os.pathsep + os.environ["PATH"]
    {"run": run, "iterate": iterate, "status": status, "stop": stop}[a.cmd](c)


if __name__ == "__main__":
    main()
