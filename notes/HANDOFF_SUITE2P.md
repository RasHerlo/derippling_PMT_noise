# Handoff: suite2p agent

**From:** derippling_PMT_noise (2026-08-19)  
**Sandbox:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

## Preferred defringe input (new)

**v2.2 pack_D** (stronger ChanA residual cleanup than v2.1 pack_B):

```text
inputs/defringed_v22/ChanA/ChanA_stk_defringed_v22.tif
inputs/defringed_v22/ChanB/ChanB_stk_defringed_v22.tif
```

(If `defringed_v22` is still writing, watch `defringe_runs/v22_full_seeded500/STATUS.md`.)

## Still available (previous SOTA)

```text
inputs/defringed_v21/ChanA|B/*_stk_defringed_v21.tif
```

Use for A/B vs v2.2. **Do not** use `inputs/defringed` (legacy v2, superseded on
its scores). Its ChanA `q≈6` was long called wrong; that label is now retired —
dark-current controls confirm `q≈6` is real, time-varying structure
(`notes/DARKCURRENT.md` §3.1c). The demotion stands on the residual metrics
alone.

## Constraints

- Write MC under `mc_runs/<tag>/` only  
- Do not overwrite `inputs/raw|defringed|defringed_v21|support*`  
- Prefer same-length stacks when comparing (5400 full vs 5340 SUPPORT)

## Suggested compares

1. MC on `defringed_v22` vs `defringed_v21` vs raw  
2. Hold SUPPORT out until SUPPORT retrain on **v2.2** (or v21) completes  

Details: `notes/OPTIMIZATION_STATUS.md`, `notes/PROGRESS.md`
