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
18.4, RMS 17.9. ChanB catalog q=81. Inspect PDFs under `seed10/overview.pdf`.
