# Handoff: SUPPORT agent

You are optimizing **SUPPORT denoising** in https://github.com/RasHerlo/SUPPORT  
Shared sandbox (do not invent a second data tree):

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

## Hard constraints

- Do **not** touch `mc_runs/` (suite2p in progress).
- Do **not** rename/overwrite `inputs/raw`, `inputs/defringed`, or `inputs/support`.
- Write new outputs under `support_runs/<tag>/` only.

## Defringe context (read-only)

- Current defringe winner: **`pack_B`** (v2.1 defaults)
- **Full stacks for suite2p / combo (preferred):**
  - `inputs/defringed_v21/ChanA/ChanA_stk_defringed_v21.tif`
  - `inputs/defringed_v21/ChanB/ChanB_stk_defringed_v21.tif`
  - (seeded from 500fr families: ChanA q=14, ChanB q=60)
- 500fr winner + avgs: `defringe_runs/v21_sweep_500fr/accepted/pack_B/`
- Status / pros-cons: repo `notes/OPTIMIZATION_STATUS.md`
- **Do not use** `inputs/defringed` (legacy v2; ChanA wrong q≈6)

## Requested task (inference compare first)

1. Run SUPPORT **inference** (existing pretrained / `model_10` OK) on:
   - raw: `inputs/raw/...` or `inputs/slices_500fr/raw/` for a fast path
   - defringed: prefer `inputs/defringed_v21/` (full) or `defringe_runs/v21_sweep_500fr/accepted/pack_B/` (500fr)
2. Save under e.g. `support_runs/packB_vs_raw/`.
3. Deliver:
   - same-frame montages: raw | defringed | SUPPORT(raw) | SUPPORT(defringed)
   - brief note on residual fringe visibility / blocky artifacts
   - any quantitative block/seam or similar metrics you already use
4. **Retrain** on `defringed_v21` when ready; models trained on fringed data may treat fringe as signal.

## Return for the overview repo

Keep a status file **inside the SUPPORT repo** (or under sandbox `support_runs/<tag>/STATUS.md`).  
The overview repo https://github.com/RasHerlo/figure_for_cAMP_Neu_paper will collect it — do not edit that overview repo from the SUPPORT agent unless the user asks there.

Include: approach tried, pros/cons vs SUPPORT-on-raw, artifact paths under `support_runs/`.
