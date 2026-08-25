# Progress handoff — 2026-08-19

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

## Open thread — dark-current controls (2026-08-25)

Batch defringe left Haj Grant ChanA as `needs_review` (correctly: no control
exists at 16X / 15.136 Hz / `scanMode=1`). Investigating that produced a new
`darkcurrent/` area and a first measured characterisation of the fringe layer on
Shinano. Headline results: fringe is strongly **edge-weighted in x**, its
amplitude **depends on excitation light**, ChanB `q` drifts slowly and stepwise,
phase does not hold frame to frame, and the long-standing ChanA "`q=6` trap" turns
out to rest on an untested inference rather than a measurement.

No production code changed. Full state, settled points and ordered next steps:
`notes/DARKCURRENT.md` §9.

## Key scripts

| Script | Role |
|---|---|
| `pmt_fringe_raw_adaptive_v22.py` | Leading cleaner |
| `sweep_v22_500fr.py` | Lever sweep vs pack_B |
| `run_full_v22_seeded500.py` | Full-stack seeded promote |
| `pmt_fringe_raw_adaptive_v21.py` | Prior pack_B defaults |

## Next

1. Confirm `defringed_v22` STATUS complete  
2. SUPPORT retrain on v2.2  
3. suite2p MC bakeoff v22 vs v21 vs raw  
4. Dark-current controls: `notes/DARKCURRENT.md` §9 (resume list); needs new
   recordings at the Haj Grant configuration to go further  
