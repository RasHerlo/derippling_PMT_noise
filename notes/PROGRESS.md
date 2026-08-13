# Progress handoff — 2026-08-13

Resume point for continuing PMT fringe derippling work.

## Goal (unchanged)

Remove intermittent, spatially drifting PMT fringe from microscopy TIFF stacks in post-processing:

```text
raw → defringe → SUPPORT
```

Preferences: **raw-only** detection/cleaning (no SUPPORT before defringe), minimal biological artifacts, residual fringe on strong frames ideally **&lt;5–10%**.

## Leading method now

**GPT raw-adaptive v2.1** — `reference/gpt/pmt_fringe_raw_adaptive_v21.py`

Builds on v2 (`pmt_fringe_raw_adaptive.py`):
1. Confidence-scaled `max_alpha`: keep 0.85 for partial confidence; ramp to 0.97 when `gate ≥ 0.95`, paired, and `strength ≥ 0.22` (span 0.15).
2. Residual second pass if paired ridge still above local background (`residual_strength_min=0.08`, `residual_alpha=0.70`).
3. Weak/absent fringe (`gate=0`) still unmodified.

Re-score script: `rescore_v21.py`  
Summary on disk:  
`...\cursor tests\SUMMARY_gpt_raw_adaptive_v21.md`

## Latest 500-frame residual scores

Metric: median fringe-specific excess power(cleaned) / excess(raw) in tracked PMT ridge segments.

| Set | ChanA v2 → v2.1 | ChanB v2 → v2.1 |
|---|---|---|
| gate > 0.5 | 16.0% → **13.7%** | 7.3% → **3.9%** |
| strongest 25% | 13.6% → **10.6%** | 4.8% → **2.0%** |

- Gate≈0 removed RMS = 0 on both channels (safety preserved).
- ChanB meets the &lt;5–10% strong-frame target; ChanA improved but is still slightly above on strongest frames.

500fr detections (stable):
- ChanA: `q≈14 / hi≈242`, fx ≈ ±10–38  
- ChanB: `q≈60 / hi≈196`, fx ≈ ±10–41  

## Critical finding: full-stack ChanA used wrong family

Existing production defringe (5400 frames) locked onto different `q` than the clear 500fr signature:

| Stack | 500fr test family | Full-stack v2 family |
|---|---|---|
| ChanA | **q=14** | **q=6** (weak; little/no removal on early frames) |
| ChanB | q=60 | q=53 (close; still useful) |

On the matching 500fr prefix, full ChanA defringe left **~100%** residual at the real q=14 ridge. ChanB full defringe left ~6–14% depending on family definition; SUPPORT then drops apparent ridge excess further (~1%).

**Implication:** re-defringe full ChanA/ChanB stacks with **v2.1** before another SUPPORT pass / retrain. Do not trust current `ChanA_stk_defringed.tif` as “clean” for the real fringe.

## Data locations

Base: `F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\`

| What | Path |
|---|---|
| Raw 500fr tests | `SUPPORT_ChanB\to build FFT deripple\raw test files\ChanA/B_raw_500fr.tif` |
| Cursor test outputs / PDFs | `SUPPORT_ChanB\to build FFT deripple\cursor tests\` |
| Full raw | `ChanA\ChanA_stk.tif`, `ChanB\ChanB_stk.tif` |
| Current (v2) defringed | `ChanA_defringe\ChanA_stk_defringed.tif`, `ChanB_defringe\...` |
| SUPPORT on defringed | `ChanA_defringe\SUPPORT\ChanA_stk_defringed_denoised.tif` (same for B) |
| Original SUPPORT (no defringe) | `SUPPORT_ChanA\denoised_cut.tif`, `SUPPORT_ChanB\denoised_cut.tif` |

## Suggested next steps (priority order)

1. **Re-defringe full 5400-frame ChanA + ChanB with v2.1**  
   - Prefer writing to new outputs (e.g. `*_defringed_v21.tif`) so old v2 results remain for comparison.  
   - Watch ChanA detection: should prefer ~q=14 family if present; investigate why full-stack median spectrum preferred q=6 (sampling / row_z / drift).
2. **Spot-check residual ridge power** on full stacks (or another 500fr slice mid-experiment), especially ChanA mid-FOV after SUPPORT.
3. **Retrain SUPPORT / `model_10` on v2.1-defringed data** — do not skip; old model was trained on fringed distribution and can unmask/amplify residual periods.
4. Optional ChanA push if still &gt;10% on strongest frames after full-stack v2.1: tighten residual pass or slightly raise high-confidence aggression **without** widening full rows / global threshold drops.
5. Repo streamline (deferred): prune bake-off clutter; keep v2.1 + notes + key runners.

## Reminder

**Retrain SUPPORT on defringed data** after v2.1 full-stack re-defringe.

## Key repo files

| Path | Role |
|---|---|
| `reference/gpt/pmt_fringe_raw_adaptive_v21.py` | Leading cleaner |
| `reference/gpt/pmt_fringe_raw_adaptive.py` | v2 baseline |
| `rescore_v21.py` | 500fr clean + residual rescore vs v2 |
| `stress_test_v2.py` | Injection / continuity (v2 defaults) |
| `compare_support_blocks.py` | SUPPORT block/seam compare |
| `notes/PROGRESS.md` | This handoff |

## Design constraints still in force

- Prefer not widening full FFT rows or globally lowering detection thresholds first.
- Optimize especially ChanA and mid-FOV residual visibility after SUPPORT.
- Keep “do nothing when fringe absent” as a hard safety property.
