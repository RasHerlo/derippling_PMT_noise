# Progress handoff — 2026-08-28 evening (shutter-seed)

Pickup: **`notes/HANDOFF.md`**. Haj Grant ChanA is still `needs_review`.
In-stack shutter frames **756–760** are the real ChanA/ChanB fringe (q≈10 /
q≈81). Probe PDFs under each channel’s `defringe_v22/shutter_seed_test/`.
Next: flatten those five ChanA frames fully with a peaked fx comb, then use
that comb + short q-track on live frames; do not re-seed ChanA from the
z-ladder.

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

**Three prior branches (design, not code):** A = same-day exact DC; B =
older DC with a sufficient raster match (qualified guess); C = priors from
successful cleans of similar live experiments. Repo can accumulate this as a
committed fingerprint→families library (`darkcurrent/` + a live-prior
catalog). Per-dataset `.defringe_cache/` is not that library.

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
