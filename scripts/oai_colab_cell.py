# ── Paste this as ONE Colab cell (GPU runtime: A100 80GB, or the 96GB card) ──
# Tunes Goedel-Prover-V2-32B on its own Lean-verified proofs of research and
# competition problems, evaluates base against tuned on held-out problems, and
# leaves verdict.json, report.md and the LoRA adapter in MyDrive/oaimath/run.
# The VM is disposable; Drive is the state. If the session dies, run this same
# cell again on a new VM and the pipeline resumes where it stopped.
#
# Before running: `bash scripts/oai_run.sh bundle` on the Mac, then upload
# ~/Downloads/oaimath_bundle.tgz to Google Drive as MyDrive/oaimath/oaimath_bundle.tgz.
# The VM needs ~140 GB of disk (weights 62 GB + merged 62 GB); it refuses before
# downloading anything if there is less.
# Evaluation here is fp8 for base and tuned alike (bf16 32B leaves one card no
# room for a batch); the lab's bf16 numbers are the production ones.
#
# After round 1 (run/), each further round lets the last tuned model sample
# problems no round has tried, retrains from the base on every round's proofs
# (run/r2, run/r3), and keeps going only while a round beats the one before.
# Then the best round and the base get 16 samples per held-out problem:
# pass@1 to pass@10 in run/final/report.md. run/progress.txt shows progress
# from any device; run/hours.json shows the GPU hours spent so far.

SCALE = 1.0     # multiplies every time budget. Round 1 ran at 1.5; the rounds after are
                # sized at 1.0 (~37 GPU hours a round, ~23 for the final evaluation).
ROUNDS = 3      # rounds in all, round 1 included
BUDGET = 110    # GPU hours for everything after round 1. A round starts only if it and the
                # final evaluation fit in what is left. Keep these three the same on re-runs.

from google.colab import drive
drive.mount("/content/drive")
D = "/content/drive/MyDrive/oaimath"

!mkdir -p /content/oaimath {D}/run && tar -xzf {D}/oaimath_bundle.tgz -C /content/oaimath
!bash /content/oaimath/oai_colab_setup.sh
!cd /content/oaimath && python oai_pipeline.py iterate --profile colab --scale {SCALE} --rounds {ROUNDS} --budget {BUDGET} --out {D}/run 2>&1 | tee -a {D}/run/pipeline.log
!cat {D}/run/final/report.md 2>/dev/null || cat {D}/run/report.md
