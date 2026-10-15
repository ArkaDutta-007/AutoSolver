# ── Paste this as ONE Colab cell (A100 80GB runtime) ─────────────────────────
# Trains the formalizer on 58k public NL->Lean pairs and pushes the adapter to a
# PRIVATE HuggingFace repo, so the result survives the session dying and no
# tunnel back to the Mac is needed.
#
# Before running: upload evals/formalize_bench.json (89 KB) via the file picker
# and rename it bench.json. Without it the script refuses to start -- Herald is
# Mathlib-derived and so is that benchmark, so training unfiltered would put the
# test set into training and every later number would be a memorisation score.

!pip -q install "transformers>=4.44" peft accelerate pyarrow huggingface_hub

import os
from huggingface_hub import login, HfApi
from google.colab import drive, userdata

# HF_TOKEN goes in Colab Secrets (key icon, left sidebar) with WRITE scope --
# never pasted into the notebook, which would put it in the saved .ipynb.
login(userdata.get("HF_TOKEN"))
drive.mount("/content/drive")

REPO = "YOUR_HF_USERNAME/lean-formalizer-v3"        # <- change this
DRIVE = "/content/drive/MyDrive/formalizer"

# Drive checkpoints every 200 steps; if the session dies, rerun this same cell
# and it resumes rather than restarting.
!wget -q -O train_colab.py https://raw.githubusercontent.com/ArkaDutta-007/AutoSolver/master/scripts/train_colab.py || echo "fetch train_colab.py manually if the repo is private"
!python train_colab.py --drive {DRIVE} --bench bench.json --epochs 1 --bs 16

api = HfApi()
api.create_repo(REPO, private=True, exist_ok=True)
api.upload_folder(folder_path=f"{DRIVE}/run", repo_id=REPO,
                  allow_patterns=["adapter*", "*.json"])   # adapter only, ~150 MB
print(f"pushed adapter -> https://huggingface.co/{REPO}")
