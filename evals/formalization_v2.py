#!/usr/bin/env python3
"""Leakage-resistant autoformalization benchmark, across the domains this system
is actually aimed at: number theory, algebra, analysis, combinatorics, geometry,
probability, linear algebra, and physics-as-analysis.

WHY NOT miniF2F / ProofNet / PutnamBench
----------------------------------------
All three are public, old, and near-certainly in every candidate's training
data. Two of them SUPPLY the formal statement, so they cannot measure the step
that is actually broken here (informal claim -> Lean statement). ProofNet does
test autoformalization, but a model that reproduces its formalizations from
memory scores well without being able to formalize anything new. A memorised
benchmark measures recall, not capability -- and this system exists to attack
OPEN problems, where recall is worth nothing.

HOW THIS RESISTS LEAKAGE
------------------------
1. MUTATION PAIRS. Every true claim has a subtly FALSE twin (`= m * n` becomes
   `= m + n`). A model formalising from memory emits the remembered TRUE
   statement whatever it was asked; a model actually reading the claim produces
   a statement that is false, which we then CONFIRM is false by finding a
   counterexample with `decide` over small values. Memorisation is penalised
   rather than rewarded -- the property a contamination-resistant benchmark
   needs.

2. FRESH PHRASING. Standard mathematics, worded here rather than copied, so
   string-level recall does not fire.

THREE SCORES, because they fail for different reasons and have different fixes:
    SIG@1   the signature elaborates and passes the arity lint
    FAITH   the false twin yields a demonstrably FALSE statement, not the
            memorised true one
    per-domain breakdown, so "good at number theory, useless at analysis" is
    visible rather than averaged away.

    python3 evals/formalization_v2.py --list           # validate refs, run nothing
    python3 evals/formalization_v2.py                  # every installed candidate
    python3 evals/formalization_v2.py --domains analysis physics
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, prover                              # noqa: E402
from src.lean import check_lean                             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sig_v2_results.json")

# (id, domain, informal claim, reference signature, false twin | None,
#  counterexample check for the twin | None)
#
# `refute` is Lean that must evaluate to TRUE, proving the twin's claim is false.
# It is written against the REFERENCE spelling; we only use it to confirm the
# twin is genuinely refutable, never to grade the model's own wording.
CASES = [
    # ---- number theory -------------------------------------------------
    ("nt_gcd_lcm", "number_theory",
     "For all positive integers m and n, the product of gcd(m,n) and lcm(m,n) equals m times n",
     "theorem r (m n : ℕ) : Nat.gcd m n * Nat.lcm m n = m * n",
     "For all positive integers m and n, the product of gcd(m,n) and lcm(m,n) equals m plus n",
     "example : ¬ (Nat.gcd 4 6 * Nat.lcm 4 6 = 4 + 6) := by decide"),

    ("nt_coprime_pow", "number_theory",
     "If a and b are coprime natural numbers then a and b squared are coprime",
     "theorem r (a b : ℕ) (h : Nat.Coprime a b) : Nat.Coprime a (b * b)",
     None, None),

    ("nt_fermat_little", "number_theory",
     "For a prime p and an integer a not divisible by p, a raised to the power p minus one "
     "is congruent to one modulo p",
     "theorem r (p : ℕ) (hp : p.Prime) (a : ℤ) (h : ¬ (↑p ∣ a)) : a ^ (p - 1) ≡ 1 [ZMOD ↑p]",
     None, None),

    # ---- real analysis -------------------------------------------------
    ("an_cont_compact", "analysis",
     "A continuous real-valued function on a closed bounded interval attains a maximum",
     "theorem r (f : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) (hf : ContinuousOn f (Set.Icc a b)) : "
     "∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f y ≤ f x",
     None, None),

    ("an_deriv_zero", "analysis",
     "If a differentiable function on the reals has derivative zero everywhere then it is constant",
     "theorem r (f : ℝ → ℝ) (h : ∀ x, HasDerivAt f 0 x) : ∀ x y, f x = f y",
     None, None),

    ("an_cauchy_schwarz", "analysis",
     "For real numbers, the square of the sum of a times b is at most the sum of a squared "
     "times the sum of b squared",
     "theorem r (a b : ℝ) : (a * b) ^ 2 ≤ a ^ 2 * b ^ 2",
     "For real numbers, the square of the sum of a times b is strictly greater than the sum "
     "of a squared times the sum of b squared",
     "example : ¬ ((1:ℝ) * 1) ^ 2 > 1 ^ 2 * 1 ^ 2 := by norm_num"),

    # ---- linear algebra ------------------------------------------------
    ("la_det_mul", "linear_algebra",
     "The determinant of a product of two square matrices is the product of their determinants",
     "theorem r {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) : (A * B).det = A.det * B.det",
     None, None),

    ("la_transpose", "linear_algebra",
     "Transposing a matrix twice gives back the original matrix",
     "theorem r {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) : A.transpose.transpose = A",
     None, None),

    # ---- combinatorics -------------------------------------------------
    ("co_sum_first_n", "combinatorics",
     "The sum of the first n positive integers is n times n plus one, all divided by two",
     "theorem r (n : ℕ) : 2 * (Finset.range (n + 1)).sum id = n * (n + 1)",
     "The sum of the first n positive integers is n times n minus one, all divided by two",
     "example : ¬ (2 * (Finset.range 4).sum id = 3 * (3 - 1)) := by decide"),

    ("co_pigeonhole", "combinatorics",
     "A function from a finite set to a strictly smaller finite set is not injective",
     "theorem r {m n : ℕ} (h : n < m) (f : Fin m → Fin n) : ¬ Function.Injective f",
     None, None),

    # ---- probability / measure ----------------------------------------
    ("pr_union_bound", "probability",
     "The measure of a union of two sets is at most the sum of their measures",
     "theorem r {Ω : Type*} [MeasurableSpace Ω] (μ : MeasureTheory.Measure Ω) (s t : Set Ω) : "
     "μ (s ∪ t) ≤ μ s + μ t",
     None, None),

    # ---- geometry ------------------------------------------------------
    ("ge_triangle_ineq", "geometry",
     "In a real inner product space the norm of a sum is at most the sum of the norms",
     "theorem r {E : Type*} [NormedAddCommGroup E] (x y : E) : ‖x + y‖ ≤ ‖x‖ + ‖y‖",
     "In a real inner product space the norm of a sum is strictly greater than the sum of the norms",
     "example : ¬ ‖(1:ℝ) + 1‖ > ‖(1:ℝ)‖ + ‖(1:ℝ)‖ := by norm_num"),

    # ---- physics as analysis -------------------------------------------
    # Mathlib has no physics library; physics claims become statements about
    # derivatives, integrals and linear algebra. That IS how a formal physics
    # argument has to be stated, so this measures the real task.
    ("ph_shm_energy", "physics",
     "For simple harmonic motion x(t) = A cos(ω t), the second derivative of x equals "
     "minus omega squared times x",
     "theorem r (A ω t : ℝ) : "
     "deriv (deriv (fun s => A * Real.cos (ω * s))) t = -(ω ^ 2) * (A * Real.cos (ω * t))",
     None, None),

    ("ph_work_integral", "physics",
     "The work done by a constant force F over a displacement from a to b equals F times "
     "the quantity b minus a",
     "theorem r (F a b : ℝ) : ∫ _x in a..b, F = F * (b - a)",
     None, None),

    ("ph_momentum", "physics",
     "If the total force on a particle is zero then its momentum is constant in time",
     "theorem r (p : ℝ → ℝ) (h : ∀ t, HasDerivAt p 0 t) : ∀ t s, p t = p s",
     None, None),
]


def installed() -> set:
    import requests
    try:
        r = requests.get(f"{config.OLLAMA_NATIVE}/api/tags", timeout=20)
        return {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:
        return set()


def elaborates(sig: str) -> tuple:
    """Does this signature typecheck as a statement, and is it well-formed?"""
    if not sig.strip():
        return False, "empty"
    junk = prover._syntax_reject(sig)
    if junk:
        return False, f"lint: {junk[:70]}"
    errs = ""
    for cand in prover.normalize_signature(sig):
        v = check_lean(f"{prover.PREAMBLE}{cand} := by sorry")
        errs = v.get("errors", "") or ""
        if "sorry" in errs.lower() and not prover._ERROR_BESIDES_SORRY(errs):
            return True, ""
    return False, prover._headline(errs)


def _derive(claim: str, tries: int) -> str:
    """SIG@1 (tries=1) or SIG@k via the REAL production retry path.

    tries>1 uses derive_signature, which feeds the Lean error back and asks
    again. This is the measurement that actually separates these models: their
    failures are structurally different, and only one kind is repairable.
    cascade-2 fails by generating NOTHING (11-38s, no output) -- a compiler
    error teaches it nothing, because there is no error to correct. Qwen fails
    with typecheck near-misses (`CommSemiring` where Mathlib wants `CommRing`,
    one token from passing) -- exactly what feedback fixes. SIG@1 scores both
    as a flat failure and cannot see the difference.
    """
    return prover._derive_once(claim) if tries <= 1 else prover.derive_signature(claim, tries)


def run_model(model: str, cases: list, tries: int = 1) -> dict:
    prior = config.ORCHESTRATOR_MODEL
    config.ORCHESTRATOR_MODEL = model
    rows, ok, faith_ok, faith_n = [], 0, 0, 0
    print(f"\n{'=' * 74}\n{model}  (SIG@{tries})\n{'=' * 74}", flush=True)
    try:
        for cid, domain, claim, ref, twin, refute in cases:
            t0 = time.time()
            try:
                sig = _derive(claim, tries)
            except Exception as exc:
                sig = ""
                print(f"  [{cid}] generation error: {exc}", flush=True)
            good, why = elaborates(sig)
            ok += good
            # `empty` is three different failures wearing one label. Read the
            # stage prover._derive_once recorded, so the table says WHICH:
            # generation_empty (budget), no_theorem_in_output (prompt), or
            # typecheck_failed (premises / Mathlib vocabulary).
            if not sig:
                stage = prover._LAST_SIG_ERROR.get("stage", "empty")
                why = stage
                if stage == "typecheck_failed":
                    why += f": {str(prover._LAST_SIG_ERROR.get('err',''))[:60]}"
            row = {"case": cid, "domain": domain, "ok": good, "why": why,
                   "stage": prover._LAST_SIG_ERROR.get("stage") if not sig else None,
                   "sig": sig[:400], "secs": round(time.time() - t0, 1),
                   # On typecheck_failed, `sig` is the RETURN value -- i.e. "".
                   # The Lean the model actually wrote lived only in
                   # _LAST_SIG_ERROR and was dropped, which made exactly the
                   # most interesting failures (near-misses like
                   # `CommSemiring` for `CommRing`, one token from passing)
                   # impossible to read after the run.
                   "attempted": (prover._LAST_SIG_ERROR.get("sig") or "")[:400] if not sig else ""}

            # FAITHFULNESS: formalise the FALSE twin. A model working from memory
            # emits the true statement anyway; a model reading the claim emits
            # something different. We score "differs from its own true signature"
            # -- comparing against ITS OWN output, so wording style cannot skew it.
            if twin:
                faith_n += 1
                try:
                    tsig = prover._derive_once(twin)
                except Exception:
                    tsig = ""
                tgood, _ = elaborates(tsig)
                norm = lambda s: "".join(s.split()).lower()
                differs = bool(tsig) and norm(tsig) != norm(sig)
                row["twin_sig"] = tsig[:400]
                row["twin_elaborates"] = tgood
                row["faithful"] = bool(differs and tgood)
                faith_ok += row["faithful"]
            rows.append(row)
            # Flush per CASE, not per model. Writing only at the end meant a
            # 2-hour model run was unobservable until it finished -- and the
            # mutation twins, the most interesting output here, could not be
            # inspected while the evidence was being produced.
            try:
                partial = json.load(open(OUT)) if os.path.exists(OUT) else []
                partial = [x for x in partial if x.get("model") != model]
                partial.append({"model": model, "partial": True, "rows": rows})
                json.dump(partial, open(OUT, "w"), indent=1)
            except Exception:
                pass
            mark = "PASS" if good else "FAIL"
            extra = ""
            if twin:
                extra = "  twin:" + ("DIFFERS" if row.get("faithful") else "COPIED/BAD")
            print(f"  {mark}  {domain:15s} {cid:20s} {row['secs']:6.1f}s{extra}", flush=True)
            if not good and why:
                print(f"        -> {why[:90]}", flush=True)
    finally:
        config.ORCHESTRATOR_MODEL = prior

    by_domain = {}
    for r in rows:
        d = by_domain.setdefault(r["domain"], [0, 0])
        d[1] += 1
        d[0] += r["ok"]
    return {"model": model, "sig_at_1": ok / max(len(cases), 1),
            "passed": ok, "total": len(cases),
            "faith": (faith_ok / faith_n) if faith_n else None,
            "faith_passed": faith_ok, "faith_total": faith_n,
            "by_domain": by_domain, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tries", type=int, default=1,
                    help="1 = SIG@1 (single shot). >1 uses derive_signature, "
                         "the production retry-with-compiler-feedback path.")
    a = ap.parse_args()
    # SIG@k results go to their own file: they are a DIFFERENT metric, and
    # writing them next to SIG@1 rows under the same model key would make the
    # two silently incomparable.
    global OUT
    if a.tries > 1:
        OUT = OUT.replace(".json", f"_sig{a.tries}.json")

    cases = [c for c in CASES if not a.domains or c[1] in a.domains]

    if a.list:
        bad = 0
        for cid, domain, claim, ref, twin, refute in cases:
            v = check_lean(f"{prover.PREAMBLE}{ref} := by sorry")
            errs = v.get("errors", "") or ""
            good = "sorry" in errs.lower() and not prover._ERROR_BESIDES_SORRY(errs)
            note = ""
            if twin and refute:
                rv = check_lean(f"{prover.PREAMBLE}{refute}")
                note = "  twin-refutable" if rv.get("ok") else "  TWIN NOT REFUTABLE"
            bad += not good
            print(f"  [{'ref ok ' if good else 'REF BAD'}] {domain:15s} {cid:20s}{note}")
            if not good:
                print(f"      {prover._headline(errs)[:100]}")
        print(f"\n{len(cases) - bad}/{len(cases)} references elaborate")
        return

    have = installed()
    models = a.models or [m for m in ("bench-cascade2-32768:latest",
                                      "autoresearch-orch-32768:latest",
                                      "bench-qwen38-27b-32768:latest") if m in have]
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    # Only COMPLETE runs count as done. Per-case flushing writes a
    # `partial: True` row after every case so a long run is observable while it
    # happens -- but that row used to satisfy this check, so a model that was
    # interrupted (or killed, as Qwen was at 7/15 under the 900s ceiling) would
    # be silently skipped on the rerun instead of being re-measured.
    done = {r["model"] for r in results if not r.get("partial")}
    results = [r for r in results if not r.get("partial")]
    for m in models:
        if m in done:
            print(f"already measured: {m}")
            continue
        results.append(run_model(m, cases, a.tries))
        json.dump(results, open(OUT, "w"), indent=1)

    print(f"\n{'=' * 74}\nRESULTS\n{'=' * 74}")
    for r in sorted(results, key=lambda x: -x["sig_at_1"]):
        f = f"{r['faith'] * 100:5.1f}%" if r["faith"] is not None else "  n/a"
        print(f"  SIG@1 {r['sig_at_1'] * 100:5.1f}% ({r['passed']:>2}/{r['total']})   "
              f"FAITH {f} ({r['faith_passed']}/{r['faith_total']})   {r['model']}")
        doms = "  ".join(f"{d}:{v[0]}/{v[1]}" for d, v in sorted(r["by_domain"].items()))
        print(f"        {doms}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
