# Progress handoff — 2026-08-18

Resume point for PMT fringe derippling (this repo only).  
Overview collector: https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
→ reads `notes/OPTIMIZATION_STATUS.md` + `notes/optimization_manifest.json`

## Goal (unchanged)

```text
raw → defringe (v2.1 pack_B) → SUPPORT (retrain) → suite2p MC / segment
```

Preferences: **raw-only** detection/cleaning, minimal biological artifacts, strong-frame residual fringe ideally **&lt;5–10%**.

## Leading method

**GPT raw-adaptive v2.1 `pack_B`** — `reference/gpt/pmt_fringe_raw_adaptive_v21.py`

| Knob | Value |
|---|---|
| `max_alpha` | 0.85 (partial confidence) |
| `max_alpha_high` | 1.00 |
| `high_gate` / `high_strength` / `strength_span` | 0.95 / 0.18 / 0.12 |
| `residual_strength_min` / `residual_alpha` | 0.05 / 0.95 |

Weak/absent fringe (`gate=0`) unmodified.

## Where we are (2026-08-18)

| Milestone | Status |
|---|---|
| 500fr sweep → accept `pack_B` | done |
| Stack averages (tif + inferno png) | done — under `accepted/pack_B/` |
| Injection stress (α=1) | done — `defringe_runs/v21_packB_injection/` |
| Full 5400-frame stacks | done — **seeded from 500fr families** |
| Promote for suite2p | done — `inputs/defringed_v21/` |

### Full-stack detection caveat (important)

Naive detect on all 5400 frames prefers ChanA **q=6** (weak). Correct approach: seed families from 500fr slices (**ChanA q=14**, **ChanB q=60**), then track+clean the full raw. Script: `run_full_v21_seeded500.py`.

**Do not use** sandbox `inputs/defringed` (legacy v2 / wrong ChanA q).

### Metrics snapshot

500fr residual ridge excess after `pack_B` (median):

| | gate>0.5 | strongest 25% | gate0 RMS |
|---|---:|---:|---:|
| ChanA | 11.7% | 9.3% | 0 |
| ChanB | 3.4% | 1.8% | 0 |

Injection α=1 (`packB_residual`): ChanA E_rec≈3.8 / remain≈9%; ChanB E_rec≈4.9 / remain≈2%.

## Shared sandbox

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy\`

| Path | Role |
|---|---|
| `inputs/raw/` | Full raw (do not rename; suite2p may use) |
| `inputs/defringed/` | **Legacy v2 — do not use for new tests** |
| `inputs/defringed_v21/` | **Current SOTA full stacks for suite2p** |
| `inputs/slices_500fr/raw/` | Fast 500fr loops |
| `defringe_runs/v21_sweep_500fr/accepted/pack_B/` | Winner 500fr + avgs |
| `defringe_runs/v21_packB_injection/` | Injection stress |
| `defringe_runs/v21_full_seeded500/` | Full-stack run + STATUS |
| `mc_runs/` | suite2p only — do not touch |
| `support_runs/` | SUPPORT agent writes here |

## Suite2p handoff (copy-paste)

Use these stacks for comparative segmentation / MC:

```text
...\Level3b copy\inputs\defringed_v21\ChanA\ChanA_stk_defringed_v21.tif
...\Level3b copy\inputs\defringed_v21\ChanB\ChanB_stk_defringed_v21.tif
```

Do not use `inputs/defringed`. Do not touch `mc_runs/`.

## SUPPORT handoff

See `notes/HANDOFF_SUPPORT.md`. Prefer pack_B / `defringed_v21` over SUPPORT-on-raw for compares; **retrain** still needed after full-stack promote.

## Next steps

1. Suite2p comparative tests on `inputs/defringed_v21/` (other agent).
2. SUPPORT inference compare on pack_B vs raw (`support_runs/`); then retrain on full `defringed_v21`.
3. Optional: spot-check residual ridge power on full seeded stacks; harden detection so full-stack median does not need 500fr seed.
4. Repo streamline (deferred).

## Key scripts

| Script | Role |
|---|---|
| `reference/gpt/pmt_fringe_raw_adaptive_v21.py` | Leading cleaner (`pack_B` defaults) |
| `sweep_v21_500fr.py` | High-confidence param sweep |
| `injection_packB_500fr.py` | Injection stress |
| `run_full_v21_seeded500.py` | Full-stack with 500fr-seeded families |
| `write_stack_averages.py` | Mean tif + inferno png |
| `score_sandbox_v21_500fr.py` | Residual ridge score on sandbox 500fr |
