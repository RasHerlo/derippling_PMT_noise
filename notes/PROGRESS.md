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
