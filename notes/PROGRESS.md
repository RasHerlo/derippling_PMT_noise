# Progress handoff — 2026-09-01 evening

Pickup: **`notes/HANDOFF.md`**.

Landed: congruent 4-cut seed (`batch_defringe.congruence`); 10-frame compare
of linescan vs original fy detect vs combo (`batch_defringe.seed_compare`).
Tests: `tests/test_congruence.py`, `tests/test_seed_compare.py`.

On Haj Grant ChanA (160, 1061, 700, shutter 756/760, random 920/997/1238/1245/1309):
original 4/10 PASS, linescan 4/10, **combo 6/10**. Combo is the only list that
covers both families: fx on 160/1245 (original empty), fy on shutter/1061/1309
(congruence none or weaker). 756: rank fy families by image-test, not peak
row-z (q=10 vs q=49). Frame 700 still gate=0. Not in `process_stack`.

Next: if 160/1245 fx `removed` is the vertical fringe, add fx as a **separate**
image-check candidate. Still no Haj Grant ChanA promote. Later: leftover round
for the second family; x-walk, no tiles.

---

# Progress handoff — 2026-09-01

Pickup: **`notes/HANDOFF.md`**.

Landed: recursive image-domain check (`batch_defringe.image_check`); spatial +
fx seed **probe** (`batch_defringe.spatial_seed`). Tests:
`tests/test_image_check.py`, `tests/test_spatial_seed.py`.

Learned: fy-row seeder misses vertical (x-periodic) fringes on Haj Grant ChanA
160. Do not blend `max(fy,fx)` into a fy notch. Horizontal line-scan period →
qx, then an fx-column family, is the candidate that gated on 160 (image-test
PASS at q 11→20); greedy FFT fx peak at 40 failed ridges. Not in production.

Next: review 160 `spatial_fx` `removed`; if it is the vertical fringe, propose
fy and fx as separate image-check candidates. Still no Haj Grant ChanA promote.
Later: x-walk, no tiles.

---

# Progress handoff — 2026-08-30

Pickup: **`notes/HANDOFF.md`**.

Landed: one seed catalog (`darkcurrent` / `in_stack_shutter` / `live_clean`,
`complete` + shutter `frames`); catalog line on every overview. `darkcurrent/`
stays lab notes.

Next (in order): (1) recursive propose → notch → judge `removed` in image
domain → leftover; (2) later, x-walk as a proposer only — no tile cleans.
Do not promote Haj Grant ChanA until (1) exists. Do not call Haj Grant
756–760 “the fringe”: they are a same-file quiet window showing one
configuration (q≈10 / q≈81).

---

# Progress handoff — 2026-08-28 evening (shutter-seed)

Pickup was `notes/HANDOFF.md` (now superseded by 2026-08-30). Haj Grant ChanA
is still `needs_review`. Frames **756–760** are a shutter quiet window on that
stack (ChanA q≈10, ChanB q≈81), **not** an inventory of the whole experiment.
Probe PDFs under each channel’s `defringe_v22/shutter_seed_test/`.

---

# Progress handoff — 2026-08-28

Overview collector: https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
→ `notes/OPTIMIZATION_STATUS.md` + `notes/optimization_manifest.json`

## Policy / design notes (2026-08-28) — not yet in code

**DarkCurrent definition (agreed):** shutter fully closed, no excitation light,
no sample. The 2026-02-14 folder named `DarkCurrent` was **not** that (Pockels
on; fringe amplitude followed light). True shuttered stacks may therefore be
quiet. They still measure PMT/electronics with no biology.

**Objective / `mag` / `pixelSizeUM` (assumed):** objective choice does not
change the PMT fringe. Do **not** record a mag-swap control. If ThorImage
`pixelSizeUM` changes with the objective, that is a **software** mismatch —
drop `mag` / `pixelSizeUM` from the prior key (raster keys stay). Revisit only
if a later shuttered pair at two objectives contradicts this.

**Pair (seed safety):** a family is *paired* when the ridge at `+q` and its FFT
partner (`hi`, usually `N−q`) both pass the z cut. Default seed requires this.
It is **not** a claim that fringes cannot drift. Haj Grant ChanA died here: no
pair cleared the cut, so tracking never started.

**Dynamics (tracking):** after a seed exists, each 50-frame block may move `q`
by ±6 FFT bins (`track_search`). The notch is meant to follow slow drift.
Wider window = follow larger jumps, with the risk of hopping to a neighbouring
family (e.g. `q≈6` ↔ `q≈14`).

**Averaging vs rates (2026-08-28, user-clarified):** ThorImage `<LSM
frameRate>` is the **scanner / listed** rate and does **not** change when you
only raise averaging. Each saved TIFF frame is the mean of `averageNum`
scanner frames, so **effective stack rate ≈ listed_rate / averageNum**
(`averageMode=0` → treat as 1). Wall-clock: 500 saved frames at avg 1 ≈ 100
saved frames at avg 5. Spatial `q` inside a frame should stay the same family
(usually weaker). Temporal sampling of the stack is what averaging changes.
Listed `frameRate` can still differ across protocols on its own, but in our
XMLs 15.136 vs 29.595 **co-varies with `scanMode`** (≈2×; see
`DARKCURRENT.md` §4.1) — it is not a third independent knob at 512 lines.

**Session DC (policy):** every experimental day and setting should start with
a shuttered DC at **that** raster. That is the best prior (branch A).

**Three prior branches (one catalog):** A = same-day DC or in-stack shutter;
B = older DC or older shutter; C = successful live cleans. All in
`fringe_library/catalog.json` (`source`, `complete`, shutter `frames`).
`darkcurrent/` stays lab notes. `.defringe_cache/` is not that library.

**Booking — prediction extras (beyond one Haj raster DC):** from that
baseline, one shuttered take each: listed ~30 Hz; avg 1 at the **same** listed
rate; `scanMode=0`. Add `fieldSize` or 256/1024 only if those actually vary in
the corpus. Still no mag / gain / light-on.

**Fingerprint (why each key exists):** see `notes/DARKCURRENT.md` §4.1.
Raster keys that can change `q`: `pixelX/Y`, `frameRate`, `fieldSize`,
`scanMode`, `twoWayAlignment`, averaging. Optical keys we will ignore for now:
`mag`, `pixelSizeUM`. Gain: amplitude only.

**One control we actually need for Haj Grant ChanA:** one shuttered recording
on Shinano that copies that trial’s **raster** from `Experiment.xml` (not a
factorial). Exact fields: `notes/DARKCURRENT.md` §6.

**Report gap:** `overview.pdf` is success-only. Failures (`needs_review`) must
get the same one-pager. Pair / Dynamics live in these notes, not on the PDF.

**Proposed code (implemented 2026-08-28, `batch_defringe` v0.4.0):** unsafe
ladder inspect-only; recurrent multi-block seeds + library extra candidates
that must pass on this stack; `track_search=10` with family-identity lock;
failure PDF; single-TIFF CLI = `process_stack`; raster fingerprint drops
`mag`/`pixelSizeUM`; library branches A/B/C in `fringe_library/catalog.json`.

---

# Progress handoff — 2026-08-26

Overview collector: https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
→ `notes/OPTIMIZATION_STATUS.md` + `notes/optimization_manifest.json`

## Leading method

**v2.2 `pack_D`** — `reference/gpt/pmt_fringe_raw_adaptive_v22.py`

```text
raw → defringe v2.2 → SUPPORT (retrain on v2.2) → suite2p
```

vs v2.1 pack_B (kept at `inputs/defringed_v21/` for A/B).

### Why v2.2

Lever sweep (residual min/alpha, high_strength, softer ratio_*):

| ChanA metric | pack_B | **pack_D (v2.2)** |
|---|---:|---:|
| strong25 residual | 9.3% | **6.9%** |
| gate>0.5 residual | 11.7% | **9.1%** |
| gate0 RMS | 0 | **0** |
| injection E_rec (α=1) | 3.54 | **3.50** |

ChanB strong25: 1.77% → **1.42%**. No wider masks.

## Sandbox paths

| Path | Role |
|---|---|
| `inputs/defringed_v22/` | **Preferred** full stacks for SUPPORT retrain + suite2p |
| `inputs/defringed_v21/` | Previous SOTA |
| `defringe_runs/v22_sweep_500fr/accepted/pack_D/` | 500fr winner |
| `defringe_runs/v22_full_seeded500/` | Full-stack run + STATUS |

## Sibling handoffs

- SUPPORT: `notes/HANDOFF_SUPPORT.md` — retrain on **v2.2**, not SUPPORT→defringe recycle  
- suite2p: `notes/HANDOFF_SUITE2P.md` — MC on `defringed_v22`  
- Dark-current controls: `notes/DARKCURRENT.md` — **active thread**, resume at §9

## Open thread — dark-current controls (updated 2026-08-26)

Batch defringe left Haj Grant ChanA as `needs_review` (correctly: no control
exists at 16X / 15.136 Hz / `scanMode=1`). Investigating that produced a new
`darkcurrent/` area and a first measured characterisation of the fringe layer on
Shinano. Headline results: fringe is strongly **edge-weighted in x**, its
amplitude **depends on excitation light**, ChanB `q` drifts slowly and stepwise,
phase does not hold frame to frame, and the long-standing ChanA "`q=6` trap" was
found to rest on an untested inference rather than a measurement.

**Resolved 2026-08-26.** `q≈6` on ChanA is now **independently confirmed** as
real, time-varying fringe structure, re-measured without borrowing background
from DC-adjacent rows and validated against synthetic stacks with known contents
(`python -m darkcurrent confirm`, method and results in §3.1c). The original
estimator turned out to use the DC row itself as background, which is why the
first number could not be trusted. The "trap" language is retired repo-wide.

Still production-neutral: nothing in `batch_defringe` has changed on the basis of
these controls. Full state, settled points and ordered next steps:
`notes/DARKCURRENT.md` §9.

## Key scripts

| Script | Role |
|---|---|
| `pmt_fringe_raw_adaptive_v22.py` | Leading cleaner |
| `sweep_v22_500fr.py` | Lever sweep vs pack_B |
| `run_full_v22_seeded500.py` | Full-stack seeded promote |
| `pmt_fringe_raw_adaptive_v21.py` | Prior pack_B defaults |

## Readout on every clean (2026-08-26)

`batch_defringe` v0.3.0 writes a `defringe_v22/` folder beside each raw stack
holding the cleaned stack, the `raw − cleaned` remainder stack, per-frame numbers,
the FFT mask, mean images and a one-page PDF. Point of it is that later stages can
audit and track what was removed instead of taking the clean on trust. Layout is
in the README.

## Next

1. Confirm `defringed_v22` STATUS complete  
2. SUPPORT retrain on v2.2  
3. suite2p MC bakeoff v22 vs v21 vs raw  
4. Dark-current controls: `notes/DARKCURRENT.md` §9 (resume list). Next two are
   both doable on existing recordings: test whether `q≈6` and `q≈14` co-occur on
   sandbox ChanA (`confirm --q 6,14`), and re-run ChanA `q` tracking with a wider
   window. Going further on the Haj Grant configuration still needs new
   recordings.  
