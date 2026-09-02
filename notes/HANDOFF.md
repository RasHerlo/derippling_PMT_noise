# Pickup — 2026-09-02 evening

**v4 engine is coded** (`batch_defringe/process_v4.py`). Production remains
v2.2 `defringe_v22`. v3 `defringe_v3/` is a probe. Do not overwrite those
TIFFs. Design: `notes/V4_PIPELINE.md`, schematic `v4_pipeline_schematic.pdf`.

```
python -m batch_defringe.process_v4 --root "F:\\bPACNewData2026\\Haj Grant Example"
```

writes `<channel>/defringe_v4/` (cleaned + removed stacks, `per_frame.csv`,
`overview.pdf`). `--seed10` is the 10-frame probe only.

## Aim (the thing the cleaner is for)

Fringes are **symmetric** plane-wave structure in the FOV. They come and go
across the stack, change strength and sometimes direction/shape, and their
period **shrinks toward the left/right edges** because the resonant scanner
slows down.

For **each frame** there is one hypothetically optimal FFT mask: it
attenuates the full fringe pattern and leaves biology. Linescans (image
domain) and FFT families are two views of that **same** structure. They
should be **integrated into one mask**, then pushed:

- look at what was removed and what is left
- fold leftover fringe back into the mask
- stop when `removed` starts looking like cells (except shutter: no biology,
  so push harder)

**Predicted fringe** = IFFT of the energy the current mask attenuates.
As the mask develops it should look more like a clean, symmetric fringe.
`removed` (raw − cleaned) should stay close to that prediction. If they
diverge, biology (or other non-fringe) sat in those bins — back off the
last increment.

v4 grows **one mask per frame**. Add order: catalog guess → shutter-learn
guess → linescan **center peak** (`seed_peak_mask`) → **edge/segment qs**
from the same traces → leftover FFT. Peak pieces are thin conjugate blobs
(α-scaled). Ridge pieces are pack_D local excess. No stack q-tracker.

**ChanA and ChanB are not coupled.** Fringe families differ (Haj Grant
seed-10: ChanA no catalog, shutter fy 10/30, live often fx ~15; ChanB
catalog q=81, shutter 81/162). A shared shutter *time* window (756–760)
is the same experiment’s mechanical shutter, not a shared fringe. Seed-10
reused ChanA `seed_compare` indices on ChanB for convenience only. The
full-stack overview picks inspect frames **per channel**.

v3 still union-applies separate pack_D families with a stack q-tracker.
Do not promote v4 until the Haj Grant full-stack PDFs are reviewed.

## Known fringe features (how we treat them)

| Feature | Approach |
|---|---|
| **Symmetric across the FOV** | Conjugate peaks in FFT (±q, ±fx). A one-sided blob is not a fringe. Linescan H/V/diags must describe one `(qy, qx)` or **none**. |
| **Dynamic; move a bit in time** | Per-frame mask. A prior (last frame, shutter, catalog) is a hint, not the applied q. |
| **More different between PMTs than between experiments on the same PMT** | Catalog / priors keyed by computer + channel + raster. ChanA and ChanB are separate problems. |
| **Period shortens toward the resonant edges** | One chirped structure, **not** a second family. Center holds the main P; edges are nearby bins / local P(x). Overlapping x-windows as a proposer; **no independent tile cleans**. |
| **Sporadic; power varies by frame** | Empty mask is allowed. Gate / alpha follow this frame’s evidence. |
| **Direction / shape can change** | fy ridges and fx columns (incl. fy=0) are both legal. Integrate into one mask; do not blend into one gate. |
| **Harmonics exist** (e.g. q and 3q) | Extra ridges of the **same** pattern; add them as later, more dubious layers, not as a rival cleaner. |
| **2-D ridge on shutter** | Linescan congruence can honestly be **none**. FFT fy is required. No biology → push the mask. |
| **Biology sits on top of the fringe** | Image-test `removed` **and** `removed` vs predicted IFFT. Live: back off when cells appear or the two images diverge. Shutter: ignore the biology brake; predicted should still look like fringe. |

---

ChanB production `defringe_v22` is still fine. Do **not** overwrite Haj Grant
ChanA production TIFFs unless asked. Do **not** touch `DATA\SUPPORT_*`.

**v3 probe** (`python -m batch_defringe.process_v3`) writes
`<channel>/defringe_v3/` only. Schematic: `v3_pipeline_schematic.pdf` at the
repo root. Catalog is not appended. v2.2 remains the production cleaner.

Shutter quiet windows are auto-detected (FOV std cliff, not mean). Haj Grant
ChanA/B both land on **756–760**. Image-test still vetoes shutter q on live.

## Where we are

Protocol slot is still `raw → v2.2 pack_D → SUPPORT retrain → suite2p`.
v4 is the agreed next cleaner (per-frame one mask). Not a promote.

| Channel (Haj Grant Example) | Production `defringe_v22` | v4 seed-10 (2026-09-02) |
|---|---|---|
| ChanB | **ok**, q≈81 | Catalog branch C q=81 used where it hydrates. 1061 no longer empty (fy peak ~7). |
| ChanA | **needs_review** | 160: linescan peak fx **15.2** + edge bands 12.1/18.4, RMS **17.9** (was 1.98 when pack_D columns stood in for linescan). Leftover FFT still added greedy fx q=40 after that. |

Frames **756–760** on Haj Grant are the auto-detected shutter quiet window
(contrast cliff). Still not an inventory of the stack. Same window on both
PMTs because the shutter is shared in time.

## Landed — v4 per-frame engine

`python -m batch_defringe.process_v4 --root "<experiment>"` →
`<channel>/defringe_v4/` (full stack). `--seed10` → `defringe_v4/seed10/`.

Tests: `python tests/test_process_v4.py`. Schematic:
`python -m batch_defringe.v4_schematic`.

## Landed — in-stack shutter detect

`python -m batch_defringe.shutter_detect` →
`DATA/shutter_detect_overview.pdf`.

Quiet if `std ≤ 0.40 × p75(std)` for ≥3 frames after a cliff
(`Δstd ≥ 0.35 × live_std`). Mean is plotted but not the cut (PMT offset
stays ~850 ADU on ChanA). Tests: `python tests/test_shutter_detect.py`.

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

## Landed — v3 integrated pipeline (probe)

`python -m batch_defringe.process_v3 --root "<experiment>"`

Union apply of image-tested fy **and** fx families. Eval leftover loop
(catalog → shutter-learn → linescan → leftover FFT). Writes cleaned +
removed stacks, `per_frame.csv`, `overview.pdf` with discarded-mask and
inspect pages. Tests: `python tests/test_process_v3.py`.

## Next

1. Review Haj Grant full-stack `defringe_v4/overview.pdf` (ChanA and ChanB).
   Inspect frames are per channel (shutter mid/start/end, strong, weak, empty,
   brake, most-lines) — not the ChanA seed_compare ten.
2. Watch leftover FFT (ChanA 160 still accepted greedy fx q=40 after the
   real 15.2+edges). Do not promote until then.
3. Later: x-walk as a proposer only — overlapping windows, **no tile cleans**.

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
F:\...\ChanA\defringe_v4\overview.pdf
F:\...\ChanB\defringe_v4\overview.pdf
F:\...\ChanA\defringe_v4\seed10\overview.pdf
F:\...\ChanB\defringe_v4\seed10\overview.pdf
F:\...\ChanA\defringe_v3\overview.pdf
F:\...\ChanB\defringe_v3\overview.pdf
F:\...\ChanA\defringe_v22\spatial_seed\seed_compare_10.pdf
F:\...\ChanA\defringe_v22\spatial_seed\congruence_frame_160.pdf
F:\...\ChanA\defringe_v22\spatial_seed\congruence_frame_756.pdf
F:\...\ChanA\defringe_v22\image_check\overview.pdf
F:\...\ChanA\defringe_v22\overview.pdf
F:\...\ChanB\defringe_v22\overview.pdf
```

```
python -m batch_defringe.process_v4 --root "F:\\bPACNewData2026\\Haj Grant Example"
python -m batch_defringe.process_v4 --root "F:\\bPACNewData2026\\Haj Grant Example" --seed10
python tests/test_process_v4.py
python -m batch_defringe.process_v3 --root "F:\\bPACNewData2026\\Haj Grant Example"
python tests/test_process_v3.py
python -m batch_defringe.seed_compare
python -m batch_defringe.congruence --frame 160
python -m batch_defringe.image_check
python tests/test_seed_compare.py
python tests/test_congruence.py
python tests/test_image_check.py
python tests/test_spatial_seed.py
```
