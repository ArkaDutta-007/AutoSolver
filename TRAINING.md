# Fine-tuning Goedel-Prover-V2-32B

AutoSolver's prover is [Goedel-Prover-V2-32B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B).
In agent runs it fails in two recurring ways:

- **It does not conclude.** Its reasoning runs out of tokens, or circles,
  before it writes the final code block.
- **It repeats its mistake.** In the repair turn the agent shows it Lean's
  error and asks for a fix, and the model often resubmits a proof that fails
  with the same error.

It also sees research-level statements far less often than competition
problems. The pipeline in `scripts/oai_*.py` and `scripts/train_prover.py`
fine-tunes the model on its own verified proofs to address all three, under two
constraints: decoding speed must not change, and the tuned model must be no
worse than the base on any held-out set.

## Data

| Set | Source | Train / held out |
|---|---|---|
| Research | Lean lemmas from [openai/math](https://github.com/openai/math) (Apache-2.0) | 12,924 / 614 |
| Competition | Lean Workbook (Goedel's proofs) and NuminaMath-LEAN | 7,708 / 663 |
| miniF2F test | regression set, never trained on | 212 |

The research set comes from `lean/OAI` in openai/math: 122k files and 2.2M
theorems. A theorem is kept only if, cut out of its file, it compiles in the
agent's own environment: the agent's import preamble, the file's `open`
lines, and the Mathlib version pinned in `lean/`. Lemmas that need a local
definition, a section variable or a renamed Mathlib lemma fail this check and
are dropped. `#print axioms` rejects anything that relies on `sorryAx` or
`native_decide`.

Held-out problems are split by paper (by problem for the competition sets), so
near-duplicate lemmas from one paper never land on both sides. Competition
problems matching a miniF2F test statement or problem text are removed.

## Method

1. **Sample on-policy.** The base model attempts the training problems with
   the agent's exact prompt. `tests/test_oai_scripts.py` asserts that the
   prompts in `scripts/oai_sample.py` are byte-identical to `src/prover.py`, so
   training and inference see the same distribution.
2. **Hint.** For problems it never solves, the model sees a verified reference
   proof and writes its own reasoning and proof. Training uses that output
   without the hint (STaR-style rationalization), which is how proofs the model
   cannot yet find get into the weights.
3. **Repair.** The model gets its own failed attempts in the agent's repair
   format. A reply that fixes the error is a training target. A reply that
   fails with the same error is labelled `repeat`.
4. **Verify.** A proof counts only if it compiles, uses no axiom beyond
   `propext`, `Classical.choice` and `Quot.sound`, and proves the statement it
   was given. The last condition is checked by Lean (`type_of%` plus `rfl`
   against the requested type), so a weakened restatement fails.
5. **Train.** LoRA (rank 16, alpha 32, learning rate 5e-5, one epoch) with two
   kinds of example. Supervised targets are the shortest verified,
   non-looping answer for each prompt. Preference pairs put that answer above
   a truncated, looping or repeated-error answer to the same prompt, trained
   with DPO (beta 0.1) plus the chosen answer's NLL. The DPO reference is the
   base model with the adapter switched off, so no second copy of the 32B
   weights is needed. The adapter is merged into the bf16 weights, so the
   served model has the base's architecture, size and dtype.
6. **Gate.** The tuned model replaces the base only if pass@1 on every
   held-out set is within two standard errors of the base or better, pooled
   pass@1 does not drop, decoding speed stays above 0.95x, and the merged
   checkpoint has the same tensors and files as the base.

After round 1, further rounds use the previous round's model to sample
problems no round has tried yet, then retrain from the base on all rounds'
proofs. A new round starts only if the previous one beat its predecessor and
the remaining GPU-hour budget covers it plus the final evaluation. The final
evaluation compares the best round with the base at 16 samples per problem
and reports pass@1 through pass@10.

## Results

**Round 1** (Colab, one A100 80 GB, fp8 evaluation for base and tuned alike,
4 samples per problem) passed the gate:

| Held-out set | Change in pass@1 (points) |
|---|---|
| Competition | +11.9 |
| miniF2F test | +7.5 |
| Research (openai/math) | +2.6 ± 1.9 |
| Pooled | +8.2 |

The repair-turn fix rate fell by 2 to 3 points on all three sets. The gate
does not include it, but it is the clearest thing left to improve.

**Rounds 2 and 3** and the 16-sample final evaluation are running. This
section will be updated with their numbers.

## Running it

The pipeline is one resumable command. Every stage records a `.done` marker,
and a killed run resumes where it stopped: samplers skip finished problems,
training resumes from its last saved adapter, and time-boxed stages keep
their original clock.

### Prepare the data

This needs Lean but no GPU. Verification is CPU-heavy, so use a machine with
many cores.

```bash
mkdir -p ~/oaimath/data ~/oaimath/env && cd ~/oaimath
git clone https://github.com/openai/math
# a Lean environment pinned to the same Mathlib as the agent
cp /path/to/AutoSolver/lean/{lakefile.toml,lake-manifest.json,lean-toolchain} env/
(cd env && lake exe cache get)

S=/path/to/AutoSolver/scripts
python3 $S/oai_extract.py scan   --src math/lean --out cands.jsonl
python3 $S/oai_extract.py verify --env env --cands cands.jsonl --out data/pairs.jsonl
python3 $S/oai_extract.py comp   --workbook WORKBOOK.parquet --numina NUMINA.parquet \
        --minif2f-src MINIF2F.jsonl --out comp_cands.jsonl --out-stmts comp_stmts.jsonl
python3 $S/oai_extract.py verify --env env --cands comp_cands.jsonl --out data/comp_pairs.jsonl
python3 $S/oai_extract.py minif2f --src MINIF2F.jsonl --out m2f_cands.jsonl
python3 $S/oai_extract.py verify --env env --cands m2f_cands.jsonl --out data/minif2f.jsonl --statements-only
```

`WORKBOOK.parquet` is Lean Workbook with Goedel's proofs, `NUMINA.parquet` is
NuminaMath-LEAN, and `MINIF2F.jsonl` is miniF2F with `split` and
`formal_statement` fields. Each subcommand's `--help` lists its options.

### Train on Colab

You need an A100 80 GB runtime and about 140 GB of disk (base weights 62 GB plus
a merged copy). The VM is disposable; Google Drive holds the state.

1. Build the bundle in `~/oaimath`, next to `data/` and `env/`:

   ```bash
   cd ~/oaimath
   cp $S/oai_pipeline.py $S/oai_sample.py $S/oai_extract.py $S/oai_mock_vllm.py \
      $S/train_prover.py $S/oai_colab_setup.sh .
   tar -czf oaimath_bundle.tgz oai_pipeline.py oai_sample.py oai_extract.py \
       oai_mock_vllm.py train_prover.py oai_colab_setup.sh \
       data/pairs.jsonl data/comp_pairs.jsonl data/minif2f.jsonl \
       env/lakefile.toml env/lake-manifest.json env/lean-toolchain
   ```

2. Upload it to Drive as `MyDrive/oaimath/oaimath_bundle.tgz`.
3. Paste `scripts/oai_colab_cell.py` into one Colab cell and run it. `SCALE`,
   `ROUNDS` and `BUDGET` (GPU hours) at the top set the size of the run.

If the session dies, run the same cell on a new VM. `run/progress.txt` on
Drive shows progress, `run/hours.json` the GPU hours spent, and
`run/final/report.md` the result.

### Train on your own GPUs

You need at least two 48 GB GPUs. Install
`torch transformers peft vllm huggingface_hub pyarrow`, then set the `python`
and `vllm` paths of the `lab` profile at the top of `scripts/oai_pipeline.py`
to your environment. The profile also expects the data in `~/oaimath/data`
and the Lean environment in `~/oaimath/env`.

```bash
python3 scripts/oai_pipeline.py run --profile lab --gpus 0,1      # round 1
python3 scripts/oai_pipeline.py iterate --profile lab --gpus 0,1 --rounds 3 --budget 110
python3 scripts/oai_pipeline.py status --profile lab
python3 scripts/oai_pipeline.py stop --profile lab
```

`--gpus` picks the GPUs vLLM samples on and `--train-gpus` the ones training
uses (by default the same). The run refuses to start while those GPUs are in
use. `scripts/oai_run.sh` drives the same commands from another machine over
SSH (`GOEDEL_HOST` names the server).

To test the whole pipeline on a CPU, with a mock vLLM server, a two-layer model
and real Lean checks:

```bash
python3 scripts/oai_pipeline.py run --profile selftest
```

The `selftest` profile has its own `python` path to set.

### Deploy

Nothing deploys automatically. `report.md` prints the vLLM command that serves
the merged weights under the base model's name,
`Goedel-LM/Goedel-Prover-V2-32B`, so the agent's `.env` does not change.

## A negative result: a specialist formalizer

We also tried replacing the orchestrator's formalization step with a small
specialist: Qwen2.5-7B with LoRA on informal-to-Lean statement pairs
(`scripts/train_formalizer.py`, `scripts/train_colab.py`). Before seeing any
result we fixed the decision rule in `scripts/formalizer_ab.sh`: on the
210-item benchmark, the specialist would take over only if it reached 65/210
statements proved equivalent to the gold (T2), 7 points above the 27B
baseline's 58/210. Seven points is the smallest shift the benchmark can
detect.

| Model | Compiles (T1) | Equivalent to gold (T2) |
|---|---|---|
| 27B orchestrator (baseline) | 122/210 | 58/210 |
| 7B specialist, 58k public pairs | 141/210 | 55/210 |

The specialist writes statements that compile more often but mean the right
thing slightly less often, so it stays off. It remains available behind
`FORMALIZER_BASE` for anyone who wants to try a better recipe. Training data
is rebuilt by `evals/formalize_dataset.py` (Mathlib docstrings) and
`evals/fetch_public_pairs.py` (public pairs, filtered against the benchmark).
