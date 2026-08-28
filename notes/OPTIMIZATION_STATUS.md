# Defringe optimization status (for overview repo)

**Collected by (do not edit from here):** https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
This file + `optimization_manifest.json` are the handoff surface for that repo.

**Stage:** PMT fringe removal (pre-SUPPORT, pre-suite2p)  
**Repo:** https://github.com/RasHerlo/derippling_PMT_noise  
**Sandbox data:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`  
**Machine-readable twin:** [`optimization_manifest.json`](optimization_manifest.json)  
**Last updated:** 2026-08-28 evening (shutter-seed probe)

---

## Currently chosen approach

| Field | Value |
|---|---|
| **ID** | `gpt_raw_adaptive_v22_pack_D` |
| **Script** | `reference/gpt/pmt_fringe_raw_adaptive_v22.py` |
| **Previous** | v2.1 `pack_B` (still in `inputs/defringed_v21/` for A/B) |
| **Idea** | Same ridge-segment architecture; stronger residual pass + softer local excess ratios on high-confidence frames |
| **Pipeline slot** | `raw → defringe v2.2 → SUPPORT (retrain on v2.2) → suite2p` |

### pack_D defaults (v2.2)

```text
max_alpha=0.85  max_alpha_high=1.0  high_gate=0.95
high_strength=0.15  strength_span=0.12
residual_strength_min=0.03  residual_alpha=1.0
ratio_start=1.4  ratio_full=3.5
```

### Pros
- Raw-only; gate≈0 unmodified  
- Narrow ridges (no full-row widen)  
- ChanA strong25 residual **9.3% → 6.9%** vs pack_B; gate>0.5 **11.7% → 9.1%**  
- ChanB also slightly better; injection E_recovery flat/slightly down  
- Full stacks seeded from 500fr (ChanA then locks q≈14 rather than q≈6; both are
  now known to be real, so this is a choice of which family gets cleaned)

### Cons / risks
- Softening `ratio_*` notches more moderate bins — watch biology on strong frames (injection OK on 500fr)  
- Full-stack median detect still needs 500fr seed for ChanA  
- **Open (updated 2026-08-26):** the q≈6-vs-q≈14 choice on ChanA was never a
  measurement. Dark-current controls have now independently confirmed q≈6 as
  real, time-varying fringe structure in the same `fx` band as q≈14
  (`notes/DARKCURRENT.md` §3.1c). Seeding from a fringe-rich window is still
  sound; the "trap" framing is retired. Still open is whether the two are one
  comb, which decides whether seeding on q≈14 alone leaves fringe behind.  
- SUPPORT must still be **retrained**; do not recycle SUPPORT→defringe  

### Rejected / demoted

| ID | Why |
|---|---|
| Claude full-row / GPT point-notch / v2 alone | as before |
| v2.1 pack_B | superseded by pack_D on residual; keep for A/B |
| Widening fx / full rows | still deferred |

---

## Attempts log

| When | Attempt | Result | Artifacts |
|---|---|---|---|
| 2026-08-18 | pack_B sweep + full seeded v21 | ChanA ~9% strong25; promoted `defringed_v21` | `v21_*` |
| 2026-08-18 | Injection pack_B | E_rec modest; remain ~9%/2% | `v21_packB_injection` |
| 2026-08-19 | SUPPORT model_10 on v21 | ChanA ridge amp 1.59×; boxes | SUPPORT `fullstack_v21_model10` |
| 2026-08-19 | **v2.2 lever sweep → pack_D** | ChanA strong25 **6.9%**; promote | `v22_sweep_500fr` |
| 2026-08-19 | Full-stack v2.2 seeded | → `inputs/defringed_v22/` | `v22_full_seeded500` |
| 2026-08-25 | Dark-current controls (`darkcurrent/`) | First measured characterisation of the fringe layer on Shinano | `notes/DARKCURRENT.md` |
| 2026-08-26 | `darkcurrent confirm` battery | ChanA q≈6 confirmed real and time-varying, 0 false positives | `notes/DARKCURRENT.md` §3.1c |
| 2026-08-28 | `batch_defringe` v0.4.0 | Raster fingerprint (drop mag/µm; add scanMode/avg/alignment); library A/B/C; recurrent seed; track lock; failure PDF; single TIFF = batch | `fringe_library/catalog.json` |

---

## Where to look (sandbox)

```text
inputs/defringed_v22/     ← preferred for SUPPORT retrain + suite2p
inputs/defringed_v21/     ← previous SOTA (A/B)
defringe_runs/v22_sweep_500fr/
defringe_runs/v22_full_seeded500/
```

## Handoffs

- SUPPORT: `notes/HANDOFF_SUPPORT.md` — retrain on **v2.2**  
- suite2p: `notes/HANDOFF_SUITE2P.md` — MC on `defringed_v22` (vs v21/raw)  

## Next

1. Finish/confirm full-stack v2.2 promote  
2. SUPPORT retrain on `defringed_v22`  
3. suite2p MC bakeoff on v22 (± SUPPORT after retrain)  
4. Optional: harden full-stack detection (drop 500fr seed requirement)  
5. **Implemented 2026-08-28 (`batch_defringe` v0.4.0):** assume `mag`/`pixelSizeUM` do not affect PMT fringes (dropped from prior key). Library A/B/C in `fringe_library/catalog.json`. Failure PDF + inspect-only ladder. Recurrent seeds, track identity lock. Single TIFF = `process_stack`. Still need a **shuttered DC** at Haj Grant raster (`notes/DARKCURRENT.md` §6) before ChanA can be library-seeded.  
6. **2026-08-28 evening:** in-stack shutter 756–760 on Haj Grant is a same-file DC. ChanB q≈81 already matched it; ChanA seed is q≈10, not the z-ladder q=109. Pickup `notes/HANDOFF.md`. Do not promote ChanA until shutter flatten is spatially flat and live *removed* matches that pattern.  
