# Handoff: SUPPORT agent

You are optimizing **SUPPORT denoising** in https://github.com/RasHerlo/SUPPORT  
Shared sandbox:

`F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`

## Hard constraints

- Do **not** touch `mc_runs/`  
- Do **not** rename/overwrite `inputs/raw`, `inputs/defringed`, `inputs/defringed_v21`, `inputs/support`  
- Write under `support_runs/<tag>/` only  

## Defringe context (2026-08-19)

| Tag | Path | Notes |
|---|---|---|
| **v2.2 pack_D (preferred for retrain)** | `inputs/defringed_v22/` | ChanA strong25 residual ~6.9% on 500fr (was ~9.3% pack_B); gate0=0 |
| v2.1 pack_B | `inputs/defringed_v21/` | Previous full-stack SOTA; keep for A/B |
| legacy v2 | `inputs/defringed/` | **Do not use** — superseded; ChanA was cleaned on a different `q` (see caveat below) |

500fr winner stacks: `defringe_runs/v22_sweep_500fr/accepted/pack_D/`  
Sweep summary: `defringe_runs/v22_sweep_500fr/SUMMARY.md`  
Status: `notes/OPTIMIZATION_STATUS.md`

Caveat (2026-08-25): legacy v2 ChanA used `q≈6` and was long described as the
"wrong q". Dark-current controls now show `q≈6` is real structure, so that label
was an untested inference — see `notes/DARKCURRENT.md` §3.1b. The
recommendation is unchanged (v2.2 pack_D scores better), but do not repeat
"wrong q" as established fact.

**Do not** recycle SUPPORT outputs into another defringe pass.

## Requested next work

1. **Retrain** ChanA (required) and ChanB (recommended) on **`inputs/defringed_v22`** once full-stack STATUS shows completed (fallback: `defringed_v21` if v22 still running).  
2. Inference bakeoff: SUPPORT(`model_10` on v22) vs new checkpoint on v22 — ridge amp + box/tile visuals.  
3. Optional quick 500fr compare: pack_B vs pack_D as SUPPORT inputs before full retrain finishes.  
4. Keep status in SUPPORT `notes/` for overview repo collection.

## Return for overview

Approach tried, pros/cons, artifact paths under `support_runs/`.
