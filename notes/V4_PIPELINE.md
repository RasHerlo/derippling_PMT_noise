# v4 pipeline — agreed design (coded)

Pickup with `notes/HANDOFF.md`. Schematic: `v4_pipeline_schematic.pdf` at the
repo root (`python -m batch_defringe.v4_schematic`).

**v2.2 `defringe_v22` remains production. v3 `defringe_v3/` is a probe, not
this design.** Do not overwrite those TIFFs. v4 writes `defringe_v4/` only.

---

## Purpose (one sentence)

For each frame, build **one** FFT mask that attenuates the full symmetric
fringe and leaves biology; grow it from the most certain evidence, softly,
and stop when `removed` looks like cells (except shutter: no biology, push).

---

## What v3 got wrong (so we do not repeat it)

| v3 | v4 |
|---|---|
| Union of separate pack_D families | One mask; lines added into it |
| Recursion = next unused `(axis, q)` | Recursion = same mask, next line or more α |
| Stack q-tracker walks ±10, apply ±2 | Per-frame evidence. Prior is a hint, not the applied q |
| Predicted IFFT is PDF-only | Predicted is an internal check every step |
| Image-test yes/no on a whole family | Score the **increment**; undo last step only |
| pack_D leftover residual pass missing | Strength rungs are explicit; shutter goes further |

Steal from pack_D: thin conjugate ridges (not whole rows/columns), attenuate
excess vs local spectral background, image-domain traits on `removed`.

---

## Features → approach

See the table in `HANDOFF.md` (symmetry, dynamics, PMT vs experiment, chirp,
sporadic, fy/fx, harmonics, shutter 2-D ridge, biology on top).

Chirp: center holds main P; edges are nearby bins / `P(x)`, **not** a second
family. Overlapping x-windows may **propose** bins. No independent tile cleans.

---

## Per-frame algorithm (what to implement)

### 0. Hints (not applied q)

Shutter detect (FOV std cliff). **Catalog / shutter-learn / last-frame geometry
prime the FFT-family list as informed guesses.** They are not a locked q and
not skipped. Empty mask is still allowed if nothing gates.

### 1. Collect candidate **lines** (support in FFT)

- Linescan: H, V, both diagonals. Congruence gives a **center** `(qy, qx)` or
  **none**. The same traces also give **several bands**: median/center q first,
  then edge / segment `q(x)` (chirp), not a second family.
- FFT: fy ridges + fx columns, **seeded by priors** then leftover detect.
- Leftover peaks after the current mask.

### 2. Rank (certainty / add order)

Collect already orders the guesses. Do **not** re-sort by FFT agreement and
do **not** drop priors.

1. **Catalog** — `lookup_prior` (computer + channel + fingerprint). Guess.
2. **Shutter-learn** — `learn_shutter_families` on this stack’s quiet window.
3. **Linescan center** — congruence winner ≠ none: `seed_peak_mask` at
   `(qy, qx)`. Thin conjugate blobs, α-scaled. Not pack_D columns.
4. **Linescan edge / segment qs** — same H/V rloess traces (`_pack` segs /
   L–C–R `P(x)`). Same axis as the winner. Center first, edges after.
5. **Leftover FFT** — `detect_families` + fx peak on leftover after kept steps.

If congruence is **none** (typical shutter 2-D ridge): skip 3–4; still run
1, 2, 5.

ChanA and ChanB are separate problems. Seed-10 reused ChanA frame indices on
ChanB only for convenience. Full-stack inspect frames are picked per channel.

### 3. Soft recursion (support + strength)

Two knobs: **which bins** and **how hard** (α). Start low α on core only.

| Step | Change | α |
|---|---|---|
| 0 | Core only | low (~⅓ of pack_D full) |
| 1… | Add next agreed line into the **same** mask | modest on new bins; core may tick up |
| later | Add dubious leftover bins | only if `removed` still looks like fringe |
| last | Raise α on accepted support | live: stop before cells. shutter: push |

Each step: one FFT notch → `predicted`, `removed`, leftover.

- **Predicted** = IFFT of attenuated FFT energy. Should look more like a
  clean symmetric fringe as the mask grows.
- **Removed** = raw − cleaned. Should stay close to predicted.
- Live: if `removed` looks like cells **or** `removed` diverges from
  predicted → **undo this step only**, keep the previous mask.
- Also score the increment (`removed_k − removed_{k−1}`), not only the total.
- Shutter: skip the biology brake; still require predicted to look like fringe.
- Stop when leftover is not fringe, or live brake fired, or rungs exhausted.

α rungs: few (about 3 live, 4 shutter). Spend budget on **adding lines in
certainty order**.

### Congruence (angles)

H, V, and both diagonals always run. Hypotheses: **fx** (vertical stripes →
`qx` from the **horizontal** cut, `qy=0`; the vertical cut may be quiet),
**fy** (horizontal bands → `qy` from the vertical cut), **tilted** (both).
Score = mean relative error of predicted vs measured diagonal periods.
Winner score **> 0.25** → **none** (typical shutter 2-D ridge). Skip
linescan center/edges; still catalog, shutter-learn, leftover FFT.

### α

Cap on FFT attenuation. Rungs **0.28 → 0.55 → 0.85** (shutter + **1.00**).
**0.28** is ~⅓ of pack_D full 0.85 — a chosen first rung, not measured.
Lines are added at 0.28 into **one** mask; then α is raised **collectively**
on the accepted set. v2.2 had per-family `gate` / `eff_max_alpha`. v4 ridge
lines still use pack_D **gate** (skip if 0); peak pieces do not. Shared α
only. Per-line α is an open probe (below).

### Predicted vs removed (RMSE / L2)

`predicted` = IFFT of energy the mask attenuates. `removed` = raw − cleaned.
All over **every pixel of that frame**:

- `RMS(pred)` = √ mean(pred²) — one scalar for the whole image
- `RMSE` = √ mean((removed − pred)²) — L2 (sum-of-squares) distance
- `agree` = 1 − RMSE / (RMS(pred) + RMS(removed)); live keep if **≥ 0.40**

Removed RMS vs 0 is heaviness, not this test.

### (fx, fy) across frames

No stack q-tracker. Intra-frame ridge snap only: live **±10** bins, shutter
**±2**. `Q_CLUSTER_TOL=3` merges duplicates on the **same** frame.

### 4. Write the frame

Cleaned, removed, predicted, applied heatmap, rung log (which lines, α,
whether undone).

Priors for the **next** frame are hints from this mask, not a tracker lock.

---

## Overview PDF (every run)

Write `<channel>/defringe_v4/overview.pdf` (name may follow the output dir).
The PDF is how we judge a run. Pages:

1. **Cover** — channel, shutter span, n frames, how many empty / core-only /
   full masks, median removed RMS, n live steps undone (biology brake).
2. **Shutter detect** — FOV std cliff (existing page).
3. **Stack traces** — per frame: n lines in mask, max α, removed RMS,
   predicted↔removed agreement, brake fired yes/no.
4. **Means** — mean raw / cleaned / removed / predicted.
5. **Rung story (eval frames)** — for shutter mid, 160, 700, 1061, and a
   few live examples (strong fringe, empty, brake-fired, chirp-heavy):
   - ranked line list (core / extra / dubious, linescan vs FFT)
   - mask after each kept step (not just the final)
   - predicted vs removed vs leftover vs cleaned at that step
   - any **undone** step (the increment that looked like cells)
6. **Discarded / undone gallery** — so we can see when we pushed too hard.
7. **Linescan vs FFT** — four traces + congruent mask vs FFT core, on the
   same frames.

Inspect rule: predicted should get *more complete and still stripe-like*;
removed should *track* predicted; if removed grows cells that predicted
does not have, that step is a fail even if RMS went up.

---

## Dest batch (`--out`) · USB drives · progress

```
python -m batch_defringe.process_v4 --root "E:\Rasmus-Guillermo\ECF1" --out "F:\CollectedData"
```

Source under `E:\Rasmus-Guillermo` is **write-guarded**. Only folder structure,
`Experiment.xml`, cleaned TIFF, slim PDF, `per_frame.csv`, `families.json`,
`mask_recipe.json`, and `mask_patterns.npz` land on `--out`. Complete dest
folders are skipped on resume.

**USB drop handling (in-script):** a background keepalive listdir’s the source
(read-only) and writes `F:\CollectedData\.defringe_v4_runs\keepalive.txt` every
20 s so disks are less likely to sleep. A stack is retried 3× on drive I/O
(`Permission denied`, `0 written`, vanished root), waiting up to 3 min for the
drive. After 3 consecutive drive failures the batch **stops** (does not walk
the rest of the job list). Windows USB-selective-suspend off is still the
stronger OS-level fix; the script does not change power settings.

**Progress:** console lines include stack i/N, ok/skip/err, elapsed, ETA this
stack and batch. Watch (refresh) either:

- repo `.defringe_progress.txt` (and `.json`)
- `{out}/.defringe_v4_runs/<utc>/progress.txt`

Permission-denied / incomplete dests are listed for later retry in
`notes/V4_DEST_CATCHUP.md` and `{out}/.defringe_v4_runs/catchup.md`. Same
`--out` command with skip-existing redoes any dest that is not complete.

---

## Implementation

Engine: `batch_defringe/process_v4.py`. Report: `batch_defringe/v4_report.py`.
Tests: `tests/test_process_v4.py`.

```
python -m batch_defringe.process_v4 --root "F:\\bPACNewData2026\\Haj Grant Example"
python -m batch_defringe.process_v4 --root "F:\\bPACNewData2026\\Haj Grant Example" --seed10
python tests/test_process_v4.py
```

Full stack → `<channel>/defringe_v4/` (`*_defringed_v4.tif`, `*_removed_v4.tif`,
`per_frame.csv`, `overview.pdf`). Seed-10 → `defringe_v4/seed10/`.

Seed-10 Haj Grant (2026-09-02): ChanA 160 linescan peak fx 15.2 + edges 12.1 /
18.4, RMS 17.9. ChanB catalog q=81.

Full Haj Grant (same night): ChanA 99.2% active, median RMS 8.94; ChanB 95.5%,
6.43. Removed RMS Pearson r = −0.004 (live −0.035). Overlay
`DATA/defringe_v4_rms_ChanA_ChanB.pdf`. Not a promote.

---

## Open: per-line α (not in v4)

v4 raises **one** `max_alpha` on the whole accepted set. v2.2 / pack_D did
**per-family** `gate` and `eff_max_alpha` (see `per_frame.csv` `family{i}_*`).
A later probe could keep one mask but let each line have its own α, so a
dubious leftover is not pushed as hard as the linescan core. Not implemented.
