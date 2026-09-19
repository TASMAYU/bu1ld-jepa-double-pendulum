# bu1ld-jepa-double-pendulum

Controlled JEPA-style latent-prediction stress test on reproducible
double-pendulum video trajectories.

**Milestone: M1 — evidence gate.** This commit ships the simulator, the frozen
trajectory split, the frozen experiment contract, the deterministic data
manifest and the trivial baselines. **No JEPA model is trained in M1**, and no
test-split metric has been computed or inspected.

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all        # generate all data, then fit and score the baselines
make verify     # re-derive every array and check it against the manifest hashes
```

Requires Python >= 3.10. The only runtime dependency is NumPy.

`make all` writes `data/manifest.jsonl` and `results/baselines_m1.json`. `make
verify` reconstructs all 200 trajectories from the contract and compares 800
SHA-256 hashes against the manifest.

Reference machine: macOS 26.2 (Darwin 25.2.0), Python 3.13.2, NumPy 1.26.4.
`make all` takes about 11 seconds and `make verify` about 11 seconds.

Make targets:

| target | effect |
|---|---|
| `make all` | full pipeline, train/val only |
| `make data` | simulate, render, hash, manifest |
| `make baselines` | fit and score the baseline family |
| `make verify` | determinism check |
| `make clean` | remove generated data and results |

---

## The frozen contract

Everything that could be adjusted to flatter a later result is fixed in
[`experiment_contract.json`](experiment_contract.json), which is frozen by this
commit. It is not edited after any outcome is observed; any change to it
constitutes a new experiment.

| item | value |
|---|---|
| system | undamped planar double pendulum, point masses on rigid massless rods |
| integrator | classical RK4, `dt = 0.005 s`, 800 steps, 4.0 s |
| parameters | `m1 = m2 = 1.0 kg`, `l1 = l2 = 1.0 m`, `g = 9.81`, damping 0 |
| renderer | 64x64 grayscale, 25 fps (stride 8), 100 frames, pure NumPy |
| evaluation grid | 16x16 mean-pooled, `[0, 1]` float32 |
| window | context 8 frames, target 8 frames, stride 4 |
| metric | MSE, plus RMSE, MSE/target-variance, and per-horizon MSE |
| model selection | validation split only |
| tuning budget | 12 configurations, 8 hours (applies in a later milestone) |

### Why the evaluation grid is fixed at 16x16

All baselines and every future model are scored on the same mean-pooled grid.
This keeps the trivial baselines genuinely trivial (the linear baseline's
normal equations stay a 2048x2048 solve) and keeps every model's number
directly comparable. It is a limitation: fine spatial detail is not scored.
The contract records it so that a later model cannot quietly be evaluated on a
different grid.

---

## The trajectory split

The three splits are separated on the **initial first-rod angle** `theta1_0`,
which orders trajectories by mechanical energy. Splitting is at trajectory
level, never by frame, so near-duplicate frames cannot leak across the
boundary.

| split | n | `theta1_0` range | `E_initial` range | character |
|---|---|---|---|---|
| train | 120 | 10° – 60° | -29.0 to -19.5 J | gentle, low-energy swings |
| val | 40 | 60° – 90° | -19.3 to -10.0 J | medium |
| test | 40 | 120° – 170° | +0.6 to +9.6 J | near-inverted, high-energy |

All trajectories are released from rest (`omega1 = omega2 = 0`) with
`theta2_0` in ±10°. Every trajectory draws its own generator seeded as
`split_seed * 1000000 + index`, so trajectory *i* is reproducible independently
of how many others are generated or in what order.

`results/preview.png` shows six evenly spaced frames from the first trajectory
of each split. The regime separation is visible directly: train stays in the
lower region of the frame for the whole clip, while test visits the inverted
position near the top.

---

## Baselines

The baseline *family* is frozen. Adding a stronger baseline after seeing a
result would invalidate the comparison, so these three are fixed up front.

| baseline | definition |
|---|---|
| `persistence` | repeat the final context frame across the horizon |
| `temporal_mean` | repeat the per-pixel mean of the context frames |
| `linear_ar` | ridge regression from the flattened context to the next frame, rolled out autoregressively |

### Results

`mse_over_target_variance` is MSE divided by the variance of that split's
targets. **1.0 means the predictor is exactly as good as always emitting the
split mean**, below 1.0 means it carries real information, above 1.0 means it
is actively worse than doing nothing.

| split | baseline | MSE | RMSE | MSE / target-variance |
|---|---|---|---|---|
| train | `persistence` | 1.044e-02 | 0.1022 | 0.479 |
| train | `temporal_mean` | 1.489e-02 | 0.1220 | 0.683 |
| train | `linear_ar` | **2.693e-03** | **0.0519** | **0.124** |
| val | `persistence` | **1.792e-02** | **0.1339** | **0.803** |
| val | `temporal_mean` | 1.974e-02 | 0.1405 | 0.884 |
| val | `linear_ar` | 2.960e-02 | 0.1721 | 1.326 |

Per-horizon MSE is reported in full in `results/baselines_m1.json`. Error grows
monotonically with horizon for every baseline, as expected for a chaotic
system.

---

## Findings

**1. The held-out split does real work, and the first result is negative.**
`linear_ar` is by a wide margin the best baseline on train (0.124) and the
worst on validation (1.326) — worse than the no-skill baseline of predicting
the split mean. A linear map fitted on gentle swings actively degrades when
applied to medium swings. This is the behaviour the split was constructed to
expose, and it is the main quantitative result of M1. Any later JEPA model
must beat 0.803 (persistence on validation), not the train numbers.

**2. The failure is structural, not a tuning artefact.** Of the 2048 flattened
context features, **1486 (72.6%) have exactly zero variance on the training
split** — those pixels never light up, because a pendulum released from 10°–60°
never swings into the corners of the frame. The unregularised normal equations
are therefore exactly singular, and the ridge term is doing all the work of
making the solve well posed. A model of this family cannot represent motion
into image regions it never observed, which is precisely the held-out-dynamics
question the project is meant to probe. This count is recorded automatically in
the `fit_diagnostics` block of the results file.

**3. Persistence is a strong baseline here.** It beats `temporal_mean` on both
splits and is the best validation baseline. At an 8-frame horizon on 25 fps
video the pendulum has not moved far, so copying the last frame is hard to
beat. This is worth stating plainly up front rather than discovering later,
because it sets a genuinely non-trivial bar for the JEPA model.

**4. No JEPA comparison has been run and none is implied.** M1 trains no
learned representation model. There is no latent-space number in this
repository, and no claim is made about whether JEPA-style prediction will
outperform these baselines.

---

## Integrator validation

The simulator is undamped, so total mechanical energy must be conserved. Drift
is normalised by `E_scale = (m1+m2)*g*l1 + m2*g*l2 = 29.43 J`, the energy
difference between fully inverted and hanging at rest. A fixed scale is used
rather than each trajectory's own `|E_initial|`, because test trajectories near
the separatrix have `E_initial` passing through zero, where a self-normalised
drift would diverge.

| split | worst relative drift | gate |
|---|---|---|
| train | 2.52e-08 | 1e-05 |
| val | 2.58e-07 | 1e-05 |
| test | 1.50e-06 | 1e-05 |

The pipeline raises rather than warns if the gate is exceeded.

`dt = 0.005` was checked against a `dt = 0.0005` reference on the frame grid.
Maximum state error is 3.2e-05 rad and scales at the expected fourth order
(halving `dt` cuts error roughly 16x: 1.07e-03, 3.18e-05, 8.4e-07, 2.1e-08).
One rendered pixel subtends about 0.073 rad, so integration error is more than
three orders of magnitude below the rendering resolution and does not
contribute measurably to any reported metric.

---

## Determinism

`make verify` re-derives every trajectory from the contract and compares
SHA-256 hashes against the manifest. All 800 hashes match.

Verified on this machine: `make all`, `make baselines` and a full
`make clean && make all` all produce a byte-identical `data/manifest.jsonl`
and `results/baselines_m1.json`. No timestamp is written into either file, so
the byte-identity is meaningful.

Trajectories are stored as `.npy` rather than `.npz` on purpose: `np.savez`
writes a zip container that embeds file modification times, which would make
the bytes differ between runs and the hashes useless.

`data/raw/` is not tracked in git. It is fully regenerable from
`experiment_contract.json` via `make all`, and the manifest records the hash of
every file so regeneration can be checked rather than trusted.

---

## Test-split boundary

- **No test metric has been computed or inspected.** `results/baselines_m1.json`
  records `"splits_evaluated": ["train", "val"]` and `"test_evaluated": false`.
- Test trajectories are *generated* by the pipeline, because a complete
  artifact requires them and generation is fully determined by the contract.
  Generation is not evaluation.
- Evaluating test requires passing `--allow-test` explicitly, which prints a
  warning to stderr. That flag was not used for any number in this repository.
- The split, the metric definitions, the seeds and the baseline family will not
  be changed to rescue a result after an outcome has been observed.

---

## Layout

```
experiment_contract.json   frozen protocol (simulator, split, windows, metrics, policies)
requirements.txt           numpy only
Makefile                   make all | data | baselines | verify | clean
src/
  sim/double_pendulum.py   RK4 integrator, energy and forward kinematics
  sim/render.py            dependency-free NumPy renderer, mean pooling
  data/ic_sampler.py       seeded per-trajectory initial-condition sampling
  data/windows.py          context/target window assembly
  data/manifest.py         simulate + render + hash + manifest
  baselines/trivial.py     persistence, temporal mean, linear AR
  eval/metrics.py          MSE and per-horizon MSE
  run_pipeline.py          single entry point for every stage
scripts/make_preview.py    writes results/preview.png
data/manifest.jsonl        200 trajectories with ICs, seeds, energy drift, SHA-256
results/baselines_m1.json  baseline scores on train and val
results/preview.png        sample frames from one trajectory per split
```

---

## What is not done

- No JEPA model, no encoder, no latent-space metric. That is the next milestone.
- No hyperparameter search of any kind. The tuning budget in the contract is
  unspent.
- The renderer scores 16x16 mean-pooled frames. Fine spatial detail is not
  evaluated.
- The linear baseline is fitted on overlapping sliding windows, so its
  effective sample size is smaller than the window count suggests. This affects
  the train number most and is a known limitation of that baseline.
- No cross-machine reproducibility check has been run. Floating-point results
  are expected to match on the same architecture and may differ in the last
  bits on a different one.
