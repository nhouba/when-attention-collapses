# Reproducing the synthetic and FUSS results

This document maps every synthetic-benchmark and FUSS number of the paper to the
command that produces it, the JSON field it is read from, the value printed in
the paper, and the approximate compute. Table, figure, section and appendix
numbers refer to the NeurIPS 2026 camera-ready version. The LISA experiments
(Sec. 3.3–3.4, Table 3, Figures 2 and 4, App. A–C) are not part of this release.

* Reference values in machine-readable form: [`results/paper/`](../results/paper/README.md).
* Runnable scripts: [`scripts/reproduce/`](../scripts/reproduce/).
* [`scripts/make_tables.py`](../scripts/make_tables.py) turns run outputs back into the
  paper's tables (`python scripts/make_tables.py results/runs`). Run on
  `results/paper/raw`, it rebuilds the tables from the original per-seed result files.

---

## 1. Conventions

**Environment of the original runs.** PyTorch 2.6, Python 3.13, one NVIDIA GH200
(120 GB) per process, `OMP_NUM_THREADS=MKL_NUM_THREADS=4`, runs executed one after
another. The released code was tested with the versions pinned in
`requirements.txt` (CPU).

**Commands.** All commands run from the repository root. The scripts in
`scripts/reproduce/` read these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python` | interpreter |
| `OUT_ROOT` | `results/runs` | parent of all output directories |
| `FUSS_DIR` | `data/fuss/ssdata` | FUSS root created by `bash experiments/download_fuss.sh`; must contain `train/` and `validation/` |
| `TASKS` | all three | restrict the synthetic scripts to some tasks, e.g. `TASKS=sinusoids` |
| `PARTS` | all | parts of `app_i_stress.sh`: `amp stress arch order latency` |
| `FUSS_METHODS` | six methods | methods run by `fuss.sh` |

**Synthetic protocol (Sec. 3.1).**
- Tasks: `sinusoids` (A), `gaussian_bumps` (B), `multiscale` (C); N = 256 samples.
- K ~ U{1, …, 4} sources; one dominant source with amplitude ~ U(5, 10), the others ~ U(0.2, 1).
- 5 slots (= max K + 1), 16 tokens, d = 64, I = 3 refinement iterations, γ = 3, τ = 0.3.
- Adam, learning rate 3e-4, cosine schedule, batch 128, gradient clipping at 1.0, 200 epochs.
- 5000 training mixtures (generated with seed `s`) and 1000 test mixtures (seed `s + 10000`).
- `--seeds N` runs seeds `0, …, N-1`.

These are the defaults of `experiments/synthetic.py` and `experiments/loss_baselines.py`;
the reproduce scripts pass `--epochs 200 --n-train 5000 --n-test 1000` explicitly
(`SYN_COMMON` in `scripts/reproduce/common.sh`).

**Metrics.** In every `*_results.json` record:
- `collapse_rate` is the **peak overlap** of Definition 1, computed over the first K slots on inputs with K ≥ 2;
- `mean_attn_overlap` is the **cosine overlap** of Definition 1;
- `k_accuracy` is the exact-K accuracy with existence logit > 0.

As in the research code, these metrics are averaged over test batches (batch 128, so the
last of the 1000 test mixtures form a smaller batch), not pooled over all inputs; the
difference is small and the released numbers are bit-identical to the original ones.

**Aggregation.** The paper reports the mean and the **population** standard
deviation (`numpy.std`, ddof = 0) over seeds, rounded to two decimals.
`scripts/make_tables.py` does the same.

**Resume.** Every experiment script appends to `<output-dir>/*_results.json` and
skips each (task, method, seed, sweep[, sweep configuration]) that is already
there, **without checking epochs or training-set size** (it prints a warning).
Use a fresh output directory for every configuration, or pass `--no-resume`. The
reproduce scripts give each experiment and slice its own directory under
`results/runs/`, so slices can run in parallel.

**Determinism.** Runs are seeded, but GPU and CPU kernels are not bitwise
deterministic across hardware and library versions, and single seeds of this
benchmark are sensitive to small numerical differences.
- An independent rerun of the full 8-method synthetic sweep (same seeds) reproduced
  the peak-overlap values of all 120 runs to three decimals.
- Repeated FUSS runs with the released code gave identical per-seed values (the duplicate
  records in `results/paper/raw/log_records.json`).
- Two high-variance cells changed between GPU reruns: stick-breaking on sinusoids
  (seeds 3–4; mean over seeds 0.465 reported, 0.455 in the other copy) and evidence
  depletion at amplitude ratio 1.5 (seeds 3–4; 0.41 reported, 0.45 in the other copy).
- On a CPU or another GPU, expect different per-seed values and differences of about
  ±0.05 in high-variance cells.

**Compute.** GPU-hours below are sums of the per-run `train_time_s` of the original
runs on one GH200, one process, runs in series. The model is tiny and the GPU is
launch-bound: on an Apple-silicon laptop CPU an epoch on sinusoids takes about
1.0 s (vanilla) and 1.7 s (evidence depletion), so a 200-epoch run takes about
3–6 CPU-minutes, versus 1100–1900 s on the GH200. For the synthetic experiments,
many parallel single-process CPU jobs are the cheapest route.

---

## 2. Quick check (about 5–7 minutes on a laptop CPU)

```bash
bash scripts/reproduce/quick_check.sh
```

- **What it runs:** vanilla vs `evidence_linear` on sinusoids, 3 seeds, **40 epochs**,
  5000 / 1000 mixtures. It runs on the CPU unless `QC_DEVICE=gpu`.
- **Pass criterion:** the script exits non-zero unless the mean peak overlap of
  `evidence_linear` is below half that of vanilla.
- **Reference run** (Apple-silicon CPU, versions of `requirements.txt`, 5.4 min wall time):

  | | peak overlap (mean ± pop. std) | per seed 0 / 1 / 2 | cosine overlap |
  |---|---|---|---|
  | vanilla | 0.38 ± 0.13 | 0.53 / 0.38 / 0.22 | 0.97 |
  | evidence_linear | 0.15 ± 0.11 | 0.31 / 0.08 / 0.07 | 0.62 |
  | paper (200 epochs, 5 seeds) | 0.32 ± 0.09 vs 0.07 ± 0.07 | | 0.90 vs 0.39 |

  Single seeds vary strongly at this budget (and between PyTorch versions), so expect
  different per-seed values; the gap between the two methods is the robust signal.
- `QC_EPOCHS` and `QC_SEEDS` change the budget (e.g. `QC_SEEDS=5 QC_EPOCHS=200` is the
  paper's sinusoid setting for these two methods).

---

## 3. Figure 1 — slot collapse across domains (synthetic and FUSS groups)

- **What it shows:** peak overlap, mean ± std over 5 seeds, for vanilla, sequential,
  linear and quadratic depletion on the three synthetic tasks and FUSS. The LISA group
  is not released.
- **Commands:** synthetic bars from `fig1_table1_synthetic.sh` (§4), FUSS bars from
  `fuss.sh` (§13).
- **Plotting:** the bar heights are the Table 1 cells (`collapse_rate`, mean ± `np.std`).
  The script that drew the paper's figure is not included;
  `python experiments/plot_synthetic.py --results <dir>/ablation_results.json` draws
  the per-task collapse bars of a synthetic run.

---

## 4. Table 1 — slot collapse (synthetic columns)

```bash
bash scripts/reproduce/fig1_table1_synthetic.sh      # all tasks, or TASKS=<task> per slice
```

Command per task:

```bash
python experiments/synthetic.py \
  --methods vanilla sequential evidence_binary evidence_linear evidence_quadratic \
            evidence_cubic dataspace_quadratic dataspace_quadratic_bias \
  --tasks <task> --seeds 5 --epochs 200 --n-train 5000 --n-test 1000 \
  --output-dir results/runs/synthetic_main/<task>
```

- **JSON:** records with `sweep == "method"`, field `collapse_rate`. The same run also
  gives Table 8 (synthetic columns), Table 9 and the data-space rows.
- **Compute:** 120 runs, about 52 GPU-h (1100–1900 s per run).
- **Reference values:** `results/paper/synthetic_main.json` (per-seed values from the
  original result files; every cell matches with population std).

| Method | Sinusoids | Gaussians | Multi-scale | FUSS (§13) |
|---|---|---|---|---|
| Vanilla SA | 0.32 ± 0.09 | 0.26 ± 0.07 | 0.22 ± 0.02 | 0.30 ± 0.05 |
| Sequential SA | 0.90 ± 0.09 | 0.81 ± 0.09 | 0.97 ± 0.02 | 0.94 ± 0.02 |
| Evidence linear (1−α) | 0.07 ± 0.07 | 0.03 ± 0.03 | 0.01 ± 0.01 | 0.04 ± 0.01 |
| Evidence quadratic (1−α²) | 0.16 ± 0.23 | 0.08 ± 0.06 | 0.35 ± 0.19 | 0.26 ± 0.04 |
| Data-space quad. (no bias) | 0.41 ± 0.40 | 0.23 ± 0.39 | 0.36 ± 0.20 | — |
| Data-space quad. (with bias) | 0.18 ± 0.30 | 0.08 ± 0.04 | 0.35 ± 0.20 | — |

Numbers in the text of Sec. 3.1 that follow from this run: sequential 0.81–0.97 and
vanilla 0.22–0.32; data-space std 0.39–0.40; 0.18 vs 0.16 and 0.08 vs 0.08 (data space
with bias vs evidence space); cubic beats quadratic on sinusoids, 0.11 vs 0.16 (Table 9).

---

## 5. Table 2 — alternative mechanisms

```bash
bash scripts/reproduce/table2_alternatives.sh     # vanilla / evidence rows come from §4
```

| Row | Command (per task, 5 seeds, `SYN_COMMON`) | Reference values |
|---|---|---|
| Sparsemax, Entmax-1.5, Coverage (mech.), Coverage (loss), Stick-breaking (+ DSPN for App. H) | `--methods sparsemax entmax15 coverage coverage_loss stickbreaking dspn` | per-seed values from the original training logs (3 decimals) |
| MESH | `--methods mesh` | per-seed values from the original result file |
| SA-SH | `--methods sa_sh` | per-seed values from the original training logs; the module is a reconstruction (§15) |
| Vanilla, Evidence depletion | from §4 | per-seed values from the original result file |

- **Compute:** 90 runs for sparsemax … dspn (about 37 GPU-h), 15 runs for MESH (about
  7 GPU-h), 15 runs for SA-SH (about 6 GPU-h).
- **Reference values:** `results/paper/alternatives.json`, per seed.

| Method | Sinusoids | Gaussians | Multi-scale |
|---|---|---|---|
| Vanilla SA | 0.32 ± 0.09 | 0.26 ± 0.07 | 0.22 ± 0.02 |
| Sparsemax | 0.22 ± 0.14 | 0.19 ± 0.03 | 0.21 ± 0.04 |
| Entmax-1.5 | 0.36 ± 0.11 | 0.26 ± 0.08 | 0.18 ± 0.05 |
| SA-SH | 0.07 ± 0.04 | 0.13 ± 0.03 | 0.12 ± 0.03 |
| MESH | 0.06 ± 0.02 | 0.18 ± 0.02 | 0.12 ± 0.03 |
| Coverage (mech.) | 0.78 ± 0.06 | 0.57 ± 0.13 | 0.70 ± 0.12 |
| Coverage (loss) | 0.23 ± 0.06 | 0.30 ± 0.04 | 0.24 ± 0.04 |
| Stick-breaking | 0.47 ± 0.20 | 0.76 ± 0.32 | 0.43 ± 0.19 |
| Evidence depletion (linear) | 0.07 ± 0.07 | 0.03 ± 0.03 | 0.01 ± 0.01 |

**Cosine statements (Sec. 3.1 and App. H).** Evidence depletion 0.27 / 0.14 on
Gaussians / multi-scale; every alternative ≥ 0.53; MESH 0.64 / 0.89 / 0.84; the DSPN
set predictor ≥ 0.53 (0.531 on multi-scale). The MESH values come from the original
result file (field `mean_attn_overlap`); the others are summary values as reported in
the paper (per-seed files not retained). The same runs regenerate all of them in the
field `mean_attn_overlap`.

**Matching pursuit (App. H).**
`python experiments/matching_pursuit.py --seeds 5 --n-test 1000 --output-dir results/runs/alternatives_matching_pursuit`
- Eval-only, takes seconds.
- Appends records with method `matching_pursuit` to `ablation_results.json`; its
  `collapse_rate` is 0.0 by construction (there is no attention), so do not average it
  into collapse tables.
- Original run: K-accuracy 0.61 / 0.56 / 0.60 / 0.60 / 0.57 for seeds 0–4 (not quoted in
  the paper, which describes matching pursuit qualitatively).

---

## 6. Table 10 — loss-based regularization (App. H; Sec. 3.1)

```bash
bash scripts/reproduce/app_h_loss_baselines.sh
# per task: python experiments/loss_baselines.py --tasks <task> --seeds 5 --epochs 200 \
#           --n-train 5000 --n-test 1000 --output-dir results/runs/loss_baselines/<task>
```

- **Variants:** `vanilla_divloss` (w_div = 5), `vanilla_ortho` (w_div = 0.5, w_ortho = 3),
  `vanilla_divloss_ortho` (both), and the same three with prefix `seq_` for sequential SA.
- **JSON:** `baseline_results.json`, field `collapse_rate` (this script does not compute
  the cosine overlap).
- **Compute:** 90 runs, about 38 GPU-h.
- **Reference values:** `results/paper/loss_baselines.json` (per-seed values from the
  original result file; every cell matches with population std).

| Variant | Sinusoids | Gaussians | Multi-scale |
|---|---|---|---|
| Vanilla, w_div = 5 | 0.14 ± 0.04 | 0.19 ± 0.04 | 0.10 ± 0.02 |
| Vanilla, w_ortho = 3 | 0.08 ± 0.02 | 0.10 ± 0.03 | 0.07 ± 0.02 |
| Vanilla, both | 0.06 ± 0.03 | 0.11 ± 0.06 | 0.06 ± 0.01 |
| Sequential, w_div = 5 | 0.74 ± 0.16 | 0.68 ± 0.05 | 0.85 ± 0.07 |
| Sequential, w_ortho = 3 | 0.45 ± 0.06 | 0.50 ± 0.08 | 0.49 ± 0.04 |
| Sequential, both | 0.30 ± 0.04 | 0.36 ± 0.03 | 0.43 ± 0.01 |

Numbers in the text that follow from this run: 0.06 / 0.11 / 0.06 and 0.30–0.43
(Sec. 3.1); 0.32 → 0.06 and 0.90 → 0.30 (App. H).

---

## 7. App. D — Proposition 1 and the empirical readout check

```bash
python experiments/app_d/check_bound.py              # numerical check of Prop. 1 and its tightness
python experiments/app_d/kappa_mag.py                # kappa*, bound 1 - 2 kappa^2, rho (magnitude token space)
python experiments/app_d/measure_superposition.py    # per-task dominance of the token energy
# readout geometry of trained models (needs the checkpoints of §12.4):
python experiments/app_d/readout_check.py --ckpt-dir results/runs/stress_order/checkpoints
```

- `readout_check.py` loads the 200-epoch vanilla and `evidence_linear` checkpoints
  (seeds 0–2) that `experiments/order_studies.py` writes to `<output-dir>/checkpoints/`,
  and evaluates 1024 sinusoid mixtures with K ∈ {2, 3, 4} (data seed 777, evaluation seed
  5000). It writes `results/runs/app_d/readout_check.json`.
- The other three scripts need no training and run in seconds to minutes.
- **Reference values:** `results/paper/app_d_readout_check.json`.

| Quantity (paper) | Value |
|---|---|
| vanilla, least similar slot pair, readout cosine, median per seed | 0.999 / 0.997 / 0.998 (“0.997–0.999”) |
| evidence depletion, readout cosine, median per seed | 0.921 / 0.805 / 0.833 (“0.81–0.92”) |
| vanilla, attention cosine of that pair | 0.55 / 0.34 / 0.24 |
| evidence depletion, mean-pair attention cosine | 0.05 / 0.09 / 0.19 |
| effect of removing the depletion factors e_ℓ | < 0.01 (max 0.006) |
| κ\* median, bound 1 − 2κ\*² median | 0.56, 0.37 |
| ρ median | 0.21 |

---

## 8. Table 7 — sensitivity to γ and τ (App. E)

```bash
bash scripts/reproduce/app_e_sensitivity.sh
# python experiments/synthetic.py --gamma-sweep --tau-sweep --tasks sinusoids --seeds 3 \
#        --epochs 200 --n-train 5000 --n-test 1000 --output-dir results/runs/sensitivity
```

- **Grid:** either sweep flag runs the **joint** 3 × 3 grid with `evidence_quadratic`.
  Without `--methods`, the method comparison is skipped.
- **JSON:** records with `sweep == "sensitivity"`, `method == "evidence_g{gamma}_t{tau}"`,
  fields `gamma`, `tau`.
- **Seeds:** 0–2. The (γ = 3, τ = 0.3) cell uses the same configuration and seeds as the
  quadratic row of Table 1 and gives identical per-seed values (0.020 / 0.129 / 0.016).
- **Compute:** 27 runs, about 13 GPU-h.
- **Reference values:** `results/paper/sensitivity.json` (per-seed values from the
  original result file; every cell matches with population std).

| | τ = 0.1 | τ = 0.3 | τ = 1.0 |
|---|---|---|---|
| γ = 1 | 0.35 ± 0.45 | 0.03 ± 0.01 | 0.76 ± 0.17 |
| γ = 3 | 0.31 ± 0.42 | 0.06 ± 0.05 | 0.69 ± 0.33 |
| γ = 10 | 0.02 ± 0.01 | 0.03 ± 0.01 | 0.26 ± 0.25 |

---

## 9. Table 8 — both collapse metrics (App. F)

- **Synthetic columns:** the run of §4; peak overlap is `collapse_rate`, cosine overlap
  `mean_attn_overlap` (means over 5 seeds).
- **FUSS column:** the run of §13 (`--eval-sisdr` pass), same two fields.
- **LISA column:** not released.

| Method | Sinusoids | Gaussians | Multi-scale | FUSS |
|---|---|---|---|---|
| Vanilla | 0.32 / 0.90 | 0.26 / 0.87 | 0.22 / 0.90 | 0.30 / 0.79 |
| Sequential | 0.90 / 0.95 | 0.81 / 0.91 | 0.97 / 1.00 | 0.94 / 0.97 |
| Evidence (linear) | 0.07 / 0.39 | 0.03 / 0.27 | 0.01 / 0.14 | 0.04 / 0.50 |

**Provenance.** The peak values have per-seed values (synthetic: original result file;
FUSS: original training logs). The cosine values are summary values as reported in the
paper (per-seed files not retained); their unrounded 5-seed means are in
`results/paper/synthetic_main.json` and `fuss.json` (e.g. 0.896 / 0.873 / 0.904 for
vanilla, 0.995 for sequential on multi-scale). As a partial check, the order-study
training runs (§12.4; same configuration, seeds 0–2, identical peak values) give a
sinusoid cosine overlap of 0.88 for vanilla and 0.27 for evidence depletion.

---

## 10. Table 9 — all depletion forms (App. G)

The run of §4, rows `evidence_binary`, `evidence_linear`, `evidence_quadratic` and
`evidence_cubic`. Every cell matches with population std.

| Method | Sinusoids | Gaussians | Multi-scale |
|---|---|---|---|
| Binary (hard mask) | 0.30 ± 0.27 | 0.41 ± 0.19 | 0.71 ± 0.26 |
| Linear (1−α) | 0.07 ± 0.07 | 0.03 ± 0.03 | 0.01 ± 0.01 |
| Quadratic (1−α²) | 0.16 ± 0.23 | 0.08 ± 0.06 | 0.35 ± 0.19 |
| Cubic (1−α³) | 0.11 ± 0.05 | 0.26 ± 0.32 | 0.52 ± 0.13 |

---

## 11. App. H — implementation of the alternatives

Implementation constants (all in `src/evidence_depletion/baselines.py`):
- sparsemax and entmax-1.5 are exact sorting-based maps with closed-form backward passes,
  applied over the slot axis;
- SA-SH: 5 log-domain Sinkhorn iterations and logit noise σ = 0.05 (training and evaluation);
- MESH: 5 normalized-gradient steps on the transport cost, step size 0.1, initial noise 1e-6;
- coverage (mechanism): β = 3, matched to γ;
- coverage (loss): penalty on the token-normalized attention, weight 1.0;
- stick-breaking is deterministic;
- the DSPN-style set predictor uses learned queries and one multi-head cross-attention pass.

Evidence depletion, sequential SA, the data-space variants, coverage (mechanism) and
stick-breaking (both built on the sequential module) use τ = 0.3; sparsemax, entmax,
SA-SH, MESH, coverage (loss) and DSPN use no temperature.

---

## 12. App. I — stress tests, robustness and cost

```bash
bash scripts/reproduce/app_i_stress.sh          # PARTS="amp stress arch order latency"
```

All parts use sinusoids and linear depletion.

### 12.1 Amplitude ratio (Table 11)

```bash
python experiments/synthetic.py --amp-ratio-sweep --track-dynamics --sweep-methods <m> \
       --seeds 5 --epochs 200 --n-train 5000 --n-test 1000 \
       --output-dir results/runs/stress_amp_ratio/<m>      # m in vanilla sequential evidence_linear
```

- **Generator:** ratios {1, 1.5, 2, 3, 5, 10}. With `amp_ratio = r` the dominant amplitude
  is r × U(0.2, 1.0) and the others are U(0.2, 1.0), so the ratio is r in expectation.
  (The default setting, U(5, 10) vs U(0.2, 1), is not one of these ratios.)
- **JSON:** `sweep == "amp_ratio"`, field `amp_ratio`, plus a per-epoch `dynamics` list.
- **Compute:** 90 runs, about 38 GPU-h. The sequential slice (about 14 GPU-h) is not
  reported and can be skipped.
- **Reference values:** `results/paper/stress.json` → `amp_ratio` (per-seed values from
  the original training logs).

| Ratio | 1 | 1.5 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| Vanilla | 0.30 | 0.33 | 0.30 | 0.34 | 0.34 | 0.24 |
| Evidence | 0.37 | 0.41 | 0.26 | 0.21 | 0.12 | 0.19 |

- Evidence std (population): 0.30 / 0.25 / 0.22 / 0.14 / 0.06 / 0.20 (“0.14–0.22 except at 5×”).
- Evidence per seed at ratio 1: 0.066 / 0.736 / 0.179 / 0.150 / 0.736.

### 12.2 Slot over-provisioning and cardinality

```bash
python experiments/synthetic.py --stress-sweep --seeds 3 --epochs 200 --n-train 5000 \
       --n-test 1000 --output-dir results/runs/stress_slots
```

- **Configurations:** `s_sweep` (S ∈ {5, 8, 12, 16}, K ≤ 4; vanilla and evidence_linear),
  `bigk` (K ≤ 8, S = 9) and `weak` (K = 6 at equal amplitude, S = 7; not reported).
- **JSON:** `sweep == "stress"`, fields `stress_config`, `n_slots`, `k_accuracy`,
  `depleted_matched`, `depleted_unmatched`.
- **Compute:** 42 runs, about 26 GPU-h.

| S | 5 | 8 | 12 | 16 |
|---|---|---|---|---|
| Evidence | 0.035 | 0.082 | 0.037 | 0.092 |
| Vanilla | 0.28 | 0.33 | 0.32 | 0.42 |

- K ≤ 8, S = 9: evidence 0.24 vs vanilla 0.54.
- The depleted-evidence fractions (matched slots 0.88 → 0.71, unmatched 0.91 → 0.56) and
  “exact-K accuracy ≈ 0 at S ≥ 8” are summary values as reported in the paper; only the
  S = 16 values (0.712, 0.559, K-accuracy 0.0) have a retained source. The run above
  regenerates all of them.

### 12.3 Token count

```bash
python experiments/synthetic.py --arch-sweep --seeds 3 --epochs 200 --n-train 5000 \
       --n-test 1000 --output-dir results/runs/stress_arch
```

- **Sweep:** L ∈ {8, 16, 32, 64}. The d ∈ {32, 128} and I ∈ {1, 5} axes also run but are
  not reported.
- **JSON:** `sweep == "arch"`, fields `l_tokens`, `d_model`, `n_iterations`.
- **Compute:** 48 runs, about 20 GPU-h.
- **Paper values:** L = 64: 0.484 ± 0.396. “Stable up to L = 32”: evidence 0.100, 0.035
  and 0.016 at L = 8, 16, 32.

### 12.4 Slot ordering and early errors

```bash
python experiments/order_studies.py --seeds 3 --epochs 200 --n-train 5000 --n-test 1000 \
       --n-orders 10 --output-dir results/runs/stress_order
```

- **Steps:** trains vanilla and evidence_linear, saves checkpoints to
  `<output-dir>/checkpoints/` (used by App. D, §7), then runs the eval-only studies.
- **JSON:** `order_studies_results.json`, field `study` ∈ {train, forced_error,
  order_sensitivity, evidence_ordered}.
- **Compute:** about 3.3 GPU-h. Reference values from the original result file.

| Claim | Field | Value |
|---|---|---|
| Collapse, no corruption → uniform | `collapse_rate` | 0.039 → 0.144 |
| Sources recovered: none → uniform / wrong-peak | `recovery_remaining` | 0.189 → 0.195 / 0.208 |
| Greedy evidence order vs canonical | `greedy_collapse_rate`, `canonical_collapse_rate` | 0.070 vs 0.039 (parameter MAE 1.994 vs 1.996) |
| Random inference orders | per-seed mean of 10 random orders | differs from canonical by 0.021 / 0.016 / 0.043 (< 0.05) |

### 12.5 Latency (Table 12)

```bash
python experiments/latency.py --output-dir results/runs/stress_latency
```

- **Defaults:** batch 32, d = 64, I = 3, 10 warm-up and 100 timed iterations; CPU and
  (if available) CUDA; S ∈ {5, 10, 20, 40}, L ∈ {16, 128}.
- **Paper table:** the `device == "cuda"`, `seq_len == 128` block on a GH200. Other hardware
  differs in absolute terms.

| S | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| Vanilla | 3.5 | 3.2 | 3.3 | 3.3 |
| Evidence | 13.2 | 26.1 | 50.5 | 101.1 |

The linear fit gives 2.51 ms per slot; the ratios are 3.8× at S = 5 and 30.8× at S = 40
(“≈4×” and “≈31×”, Sec. 2.3 and App. I).

---

## 13. FUSS — Table 1 column, Table 13, Table 8 column, App. J

**Data.**
- `bash experiments/download_fuss.sh` fetches `FUSS_ssdata.tar.gz` from Zenodo
  (record 3743844, CC BY 4.0) into `data/fuss/` and extracts it.
- Pass `--fuss-dir data/fuss/ssdata`; it must contain `train/` and `validation/` with
  `exampleNNNNN.wav` and `exampleNNNNN_sources/`.
- Training uses the first 5000 `train` mixtures, evaluation the first 1000 `validation`
  mixtures.

**Command.** One method per output directory:

```bash
bash scripts/reproduce/fuss.sh
# python experiments/fuss.py --fuss-dir data/fuss/ssdata --methods <m> --seeds 5 --epochs 200 \
#        --signal-length 64000 --n-tokens 32 --d-model 128 --max-train 5000 --max-test 1000 \
#        --batch-size 32 --eval-sisdr --output-dir results/runs/fuss/<m>
```

- These flags are the defaults of `experiments/fuss.py` (4 s crops, 32 tokens, d = 128).
- `--eval-sisdr` only adds the attention-to-mask readout evaluation; per-seed collapse is
  identical with and without it, so one pass serves all FUSS artifacts.
- **JSON:** `fuss_results.json` with `collapse_rate`, `mean_attn_overlap`, `k_accuracy`,
  `si_sdr`, `si_sdri`, `si_sdr_mixture` (over 404 multi-source mixtures) and `param_mae_*`.
- **Compute:** about 40–60 min per run on a GH200; the 4 core methods (20 runs) take
  about 17 GPU-h, all 6 methods about 25 GPU-h. FUSS was not timed on a CPU.

**Expected result** (the run behind every FUSS number in the paper; per-seed values from the
original training logs; `results/paper/fuss.json` → `fuss_rerun`):

| Method | Peak overlap, seeds 0–4 | mean ± std | Cosine | SI-SDRi [dB] |
|---|---|---|---|---|
| Vanilla | 0.243 0.326 0.366 0.230 0.316 | 0.30 ± 0.05 | 0.79 | +1.08 ± 0.14 |
| Sequential | 0.963 0.923 0.958 0.924 0.940 | 0.94 ± 0.02 | 0.97 | −1.14 ± 1.96 |
| Evidence linear | 0.037 0.035 0.039 0.056 0.024 | 0.04 ± 0.01 | 0.50 | −0.66 ± 0.80 |
| Evidence quadratic | 0.299 0.247 0.259 0.278 0.192 | 0.26 ± 0.04 | — | −0.40 ± 1.04 |
| Sparsemax | 0.206 0.159 0.188 0.204 0.254 | 0.20 ± 0.03 | — | −2.06 ± 0.60 |
| SA-SH | 0.041 0.044 0.008 0.022 0.018 | 0.03 ± 0.01 | — | +0.85 ± 0.12 |

- This run gives the FUSS column of Table 1, Figure 1 (group D), Table 13 (rounded to two
  decimals), the FUSS column of Table 8 and the SI-SDR diagnostics. The cosine values are summary values as reported in the paper (per-seed files not
  retained).
- “SI-SDR improvement −2.1 to +1.1 dB” (Sec. 3.2, App. J) is the range of these per-method
  means. The SA-SH row is not reported in the paper's tables.

**Earlier run (submitted version).** The submitted version of the paper reported an earlier
FUSS run, made with a previous version of the FUSS script under the same configuration
(per-seed values in `results/paper/earlier_run/fuss_v2_fuss_results.json`, `fuss.json` →
`earlier_run`). App. J of the camera-ready mentions it; the released code does not
reproduce it.

| Method (earlier run) | s0 | s1 | s2 | s3 | s4 | mean ± std |
|---|---|---|---|---|---|---|
| Vanilla | 0.22 | 0.38 | 0.34 | 0.31 | 0.22 | 0.29 ± 0.06 |
| Sequential | 0.94 | 0.91 | 0.94 | 0.91 | 0.95 | 0.93 ± 0.02 |
| Evidence linear | 0.09 | 0.04 | 0.04 | 0.04 | 0.04 | 0.05 ± 0.02 |
| Evidence quadratic | 0.42 | 0.27 | 0.25 | 0.36 | 0.23 | 0.31 ± 0.07 |

**Oracle mask, +7.7 dB (Sec. 3.2, App. J).**

```bash
python experiments/fuss_sisdr_oracle.py --fuss-dir data/fuss/ssdata --signal-length 64000 \
       --n-mixtures 300 --n-chunks 32 --seed 42
```

- Runs in about a minute (these are the defaults).
- The paper number is the printed gap `rank1_model_faithful − floor` = +7.66 dB
  (floor −1.716 dB, rank1_model_faithful 5.948 dB).

**Parameter baseline (App. J, qualitative).**
- `python experiments/fuss_param_baseline.py --fuss-dir data/fuss/ssdata` (minutes on a
  CPU) fits a constant predictor on the training sources (12615) and evaluates it on the
  validation sources (2516).
- Train-median MAE: frequency 0.0388, amplitude 0.0041, centroid 0.0621. The model with
  seed 0 reaches 0.057 / 0.0046 / 0.081, so “did not rise meaningfully above simple
  baselines” holds.
- `python experiments/fuss_param_compare.py --results-glob 'results/runs/fuss/*/fuss_results.json'`
  compares all trained runs with the baseline (this glob is the default).

---

## 14. App. K — visualizations

- **Figure 3** (superposition in audio) is a synthetic illustration; its plotting script is
  not included.
- **Figure 4** is LISA and not released.

---

## 15. Known differences

What the released code does and does not reproduce, and where a reported value has no
retained per-seed file.

1. **SA-SH is a reconstruction.** The `sa_sh` module (Table 2 row; Sinkhorn slot attention
   with symmetry-breaking noise, App. H) was rebuilt from its specification: 5 log-domain
   Sinkhorn iterations over the slot × token logits ending with the slot normalization,
   N(0, 0.05²) logit noise at training and evaluation time, parallel GRU updates as in
   vanilla SA. It is the only method not covered by the bit-for-bit equivalence tests
   against the research code. Its reference values (0.071 ± 0.038, 0.126 ± 0.032,
   0.117 ± 0.035) come from the original training logs, where the method appears under
   the name `mesh`; in this release `mesh` is full MESH.
2. **FUSS, earlier run.** The camera-ready FUSS numbers come from the run of §13, which the
   released `experiments/fuss.py` reproduces (0.30 ± 0.05, 0.94 ± 0.02, 0.04 ± 0.01 and
   0.26 ± 0.04 for vanilla, sequential, linear and quadratic depletion; ≈ 8× reduction).
   The submitted version reported an earlier run, made with a previous version of the FUSS
   script under the same configuration: 0.29 ± 0.06, 0.93 ± 0.02, 0.05 ± 0.02 and
   0.31 ± 0.07. It is not reproduced by the released code; its per-seed values are in
   `results/paper/earlier_run/fuss_v2_fuss_results.json`.
3. **Values without retained per-seed files.** These are summary values as reported in
   the paper; the commands above regenerate them, but they cannot be recomputed from the
   files in `results/paper/`:
   - the cosine overlaps of Table 8 (synthetic and FUSS columns) and the FUSS “≈1.6×”
     cosine reduction;
   - the cosine statements of Sec. 3.1 and App. H, except MESH (“every alternative ≥ 0.53”,
     evidence 0.27 / 0.14, set predictor ≥ 0.53);
   - the depleted-evidence fractions at S = 5 and the exact-K accuracy at S = 8 and 12
     (App. I, slot over-provisioning);
   - the slot over-provisioning and token-count collapse values (App. I), which have
     summary values from the original training logs but no per-seed values.
4. **Logged values with three decimals.** The per-seed values from the original training
   logs (Table 2 except MESH and the vanilla / evidence rows, Table 11, the FUSS rerun) are
   rounded to three decimals, so recomputed statistics can differ from the paper in the
   fourth decimal.
5. **GPU nondeterminism.** Two high-variance cells changed between GPU reruns of the same
   seeds: stick-breaking on sinusoids (reported mean 0.465; another copy of seeds 3–4 gives
   0.455) and evidence depletion at amplitude ratio 1.5 (reported 0.41; another copy gives
   0.45). Reproductions may land on either side, and per-seed values on other hardware
   will differ more generally (§1, “Determinism”).
6. **Plotting scripts.** The scripts that drew Figures 1 and 3 are not included; the numbers
   behind Figure 1 are those of Table 1.

---

## 16. Compute summary

One GH200, runs in series, from the per-run training times of the original runs.

| Script | Runs | GPU-h |
|---|---|---|
| `quick_check.sh` | 6 short | — (about 6 CPU-minutes) |
| `fig1_table1_synthetic.sh` | 120 | ~52 |
| `table2_alternatives.sh` | 120 + 5 eval-only | ~50 |
| `app_e_sensitivity.sh` | 27 | ~13 |
| `app_h_loss_baselines.sh` | 90 | ~38 |
| `app_i_stress.sh` (amp 38, stress 26, arch 20, order 3.3, latency < 0.1) | 186 + 6 | ~87 |
| `fuss.sh`: 6 methods, plus the oracle and parameter baseline | 30 | ~25 |
| **Total** | about 580 trainings | **~265 GPU-h** |

- Without the unreported sequential amplitude-ratio slice (about 14 GPU-h) and the
  unreported architecture axes (about 10 GPU-h) the total drops to about 240 GPU-h.
- The synthetic part (about 540 trainings) takes roughly 30–60 CPU-hours at 3–6 CPU-minutes
  per sinusoid run; runs with S = 16 or L = 64 are slower. Running many single-process jobs
  in parallel is fastest.
