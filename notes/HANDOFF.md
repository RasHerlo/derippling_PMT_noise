# Pickup — 2026-09-01 evening

ChanB production `defringe_v22` is still fine. Do **not** overwrite Haj Grant
ChanA production TIFFs. Do **not** touch `DATA\SUPPORT_*`.

Image-check, spatial-seed, congruence, and the 10-frame seed compare are
**probes**. None of them is wired into `process_stack`.

## Where we are

Protocol slot is still `raw → v2.2 pack_D → SUPPORT retrain → suite2p`.

| Channel (Haj Grant Example) | Production `defringe_v22` | Why |
|---|---|---|
| ChanB | **ok**, q≈81 | Branch C live_clean; tracking held |
| ChanA | **needs_review** | No catalog hit with fx support. fy seeder still misses vertical live fringes (160). Frame 700 still gate=0 on every method in the 10-frame compare. |

Frames **756–760** are a same-file shutter quiet window, not an inventory of
the stack.

## Landed — 10-frame seed compare

`python -m batch_defringe.seed_compare` →
`<channel>/defringe_v22/spatial_seed/seed_compare_10.pdf`.

Ten ChanA frames: live anchors **160, 1061, 700**; shutter **756, 760**; five
RNG-picked (`20260901`) **920, 997, 1238, 1245, 1309**.

| Method | PASS | FAIL | off | none |
|---|---:|---:|---:|---:|
| Original fy `detect_families` | 4 | 0 | 1 | 5 |
| Linescan congruence | 4 | 2 | 2 | 2 |
| Combo (union, image-test pick) | **6** | 2 | 2 | 0 |

Combo is strictly better than either list alone:

- **fx original cannot propose:** 160 (qx 15→20, rms 1.28) and 1245 (qx 6→13,
  rms 0.47). Image-test PASS. This is the missing ChanA live path.
- **Shutter is still fy:** congruence returns **none** on 756 and 760 (diags
  P≈87 / 23 are not one plane wave). Original fy is required. On 756 the
  loudest row-z family was q=49 (rms 1.6); combo ranked all detect q’s by
  image-test RMS and kept **q=10** (rms 26.8). That win is ranking, not
  linescan.
- **Both families present:** 1061 / 1309 — combo keeps the loud fy (rms ~23–25)
  over a weak fx PASS. A leftover round could still take fx.
- **Still dark:** 700 and 997 off (gate=0). 920 and 1238: linescan fx gates
  but **FAIL ridges** — the image-test veto holds.

Tests: `python tests/test_seed_compare.py`. Do **not** blend fy/fx into one
gate. Do **not** IFFT whole FFT rows/columns.

## Landed — congruent linescan seed

Whenever looking for a seed, always run **H, V, both diagonals**. They must
describe the same `(qy, qx)` (`batch_defringe.congruence`). Winner or **none**.
Mask is thin conjugate blobs, not whole rows/columns.

- Frame 160: **fx, qx ≈ 15.2**, score 0.003. H and both diags P ≈ 33.7.
- Frame 756: **none**. Honest: shutter fringe is a 2-D ridge.

`python -m batch_defringe.congruence --frame 160`. Tests:
`python tests/test_congruence.py`.

Rolling lowest-4 + rloess + local FFT is the 1-D curve
(`baseline_smooth.py`, `SEED_K=4`). Global sine and k=5 ACF are discarded.

## Landed — recursive image-domain check

`python -m batch_defringe.image_check` →
`<channel>/defringe_v22/image_check/overview.pdf`.

Propose one family → notch → score `removed` (coverage, even, ridges). Compact
spots on a passing ridge field are ridge-edge grain and **do not reject**.
Gate=0 does not vote.

Tests: `python tests/test_image_check.py`.

## Learned — fy vs fx (unchanged, now quantified)

The seeder only scored **fy-rows**. Frame 160’s vertical fringes live on
**fx-columns**, including fy=0 at `fx ≠ 0`. Treat fy and fx as **separate
candidates**. Do not `max(fy, fx)` then fy-notch.

| Frame | best fy-row | best fx-column (incl. fy≈0) |
|---|---|---|
| 160 | 0.065 at q=61 (below `gate_low=0.10`) | 0.49 at q=40 |
| 756 | **3.19** at q=10 | 3.32 at q=13 |
| 1061 | **0.61** at q=12 | 0.12 |

## Next

1. Review `seed_compare_10.pdf` `removed` panels (especially 160 / 1245 fx and
   756 q=10 vs q=49).
2. If those fx `removed` fields are the vertical fringes: add **fx families as
   separate** proposals in the image-check loop (congruence as the fx
   proposer; original fy detect unchanged). Rank fy families by image-test,
   not peak row-z.
3. Recursive leftover: after accepting fy, leftover fx is a second round —
   combo today picks one family per frame.
4. Later: x-walk as a proposer only — overlapping windows, **no tile cleans**.
5. Do not promote Haj Grant ChanA until live 700 (and similar off frames)
   either gate on a real family or are judged fringe-free.

## Catalog (unchanged)

`fringe_library/catalog.json`. `darkcurrent/` is lab notes, not a seed source.
In-stack shutter is incomplete (set `frames`). Amplitude is never copied.

## Cautions (still)

- Do not flatten 756–760 as the definition of success.
- Do not wire shutter detection into `process_stack` as a hard seed until the
  image-domain check can veto a bad transfer.
- Phase-lock / subtract the shutter *image* is the wrong model.
- Do not start independent tile cleans.
- Do not IFFT full FFT rows/columns (that bandpasses the FOV).

## Probe artifacts (not in git)

```
F:\...\ChanA\defringe_v22\spatial_seed\seed_compare_10.pdf
F:\...\ChanA\defringe_v22\spatial_seed\congruence_frame_160.pdf
F:\...\ChanA\defringe_v22\spatial_seed\congruence_frame_756.pdf
F:\...\ChanA\defringe_v22\image_check\overview.pdf
F:\...\ChanA\defringe_v22\overview.pdf
F:\...\ChanB\defringe_v22\overview.pdf
```

```
python -m batch_defringe.seed_compare
python -m batch_defringe.congruence --frame 160
python -m batch_defringe.image_check
python tests/test_seed_compare.py
python tests/test_congruence.py
python tests/test_image_check.py
python tests/test_spatial_seed.py
```
