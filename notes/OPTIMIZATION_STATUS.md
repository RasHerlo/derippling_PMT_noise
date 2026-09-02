# Defringe optimization status (for overview repo)

**Collected by (do not edit from here):** https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
This file + `optimization_manifest.json` are the handoff surface for that repo.

**Stage:** PMT fringe removal (pre-SUPPORT, pre-suite2p)  
**Repo:** https://github.com/RasHerlo/derippling_PMT_noise  
**Sandbox data:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`  
**Machine-readable twin:** [`optimization_manifest.json`](optimization_manifest.json)  
**Last updated:** 2026-09-02 (v4 engine coded, not promoted)

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
| 2026-08-30 | Seed catalog + image-check plan | `darkcurrent` / `in_stack_shutter` / `live_clean`; Haj Grant 756–760 = quiet window not inventory | `notes/HANDOFF.md` |
| 2026-09-01 | Recursive image-domain check | Propose→notch→score `removed`; ridge-edge compact spots do not reject; per-frame applied FFT heatmap | `batch_defringe/image_check.py`, `tests/test_image_check.py` |
| 2026-09-01 | Spatial + fx seed **probe** | fy-only misses 160’s vertical fringes. Do not `max(fy,fx)` then fy-notch. H-trace P→qx steered 160 to fx q=20 (image-test PASS); greedy FFT fx q=40 failed ridges. Not in `process_stack`. | `batch_defringe/spatial_seed.py`, `tests/test_spatial_seed.py`, ChanA `defringe_v22/spatial_seed/` |
| 2026-09-01 | Congruent 4-cut seed | H, V, both diagonals must share one `(qy, qx)` or **none**. 160 = fx qx≈15.2; 756 = none (2-D ridge). Thin-peak mask, not full rows/columns. | `batch_defringe/congruence.py`, `tests/test_congruence.py` |
| 2026-09-01 | 10-frame seed compare | Original 4/10 PASS, linescan 4/10, **combo 6/10**. Combo adds fx on 160/1245; shutter still needs fy; 756 ranking picks q=10 over q=49. 700 still off. Not in `process_stack`. | `batch_defringe/seed_compare.py`, `tests/test_seed_compare.py`, ChanA `spatial_seed/seed_compare_10.pdf` |
| 2026-09-02 | v3 integrated pipeline | Shutter detect + congruence + leftover FFT + image-test + union apply. Writes `defringe_v3/` (not v22). Schematic `v3_pipeline_schematic.pdf`. Not a promote. | `batch_defringe/process_v3.py`, `v3_report.py`, `tests/test_process_v3.py` |
| 2026-09-02 | v4 per-frame engine | One growing mask: catalog + shutter-learn guesses, linescan peak+edges, leftover FFT. Seed-10 ChanA 160 RMS 17.9. Full stack writes `defringe_v4/`. Not a promote. | `batch_defringe/process_v4.py`, `v4_report.py`, `tests/test_process_v4.py` |

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
5. **Implemented 2026-08-28 (`batch_defringe` v0.4.0):** assume `mag`/`pixelSizeUM` do not affect PMT fringes (dropped from prior key). Library A/B/C in `fringe_library/catalog.json`. Failure PDF + inspect-only ladder. Recurrent seeds, track identity lock. Single TIFF = `process_stack`.  
6. **2026-08-28 evening:** Haj Grant frames 756–760 are a same-file shutter quiet window (ChanA q≈10, ChanB q≈81). Not an inventory of the stack. Pickup `notes/HANDOFF.md`.  
7. **Implemented 2026-08-30:** one seed catalog (`darkcurrent` / `in_stack_shutter` incomplete+`frames` / `live_clean`). Overview + `families.json` show catalog use. `darkcurrent/` not used for seeding.  
8. **Implemented 2026-09-01:** recursive image-domain check (`python -m batch_defringe.image_check`). Compact spots on a ridge field do not reject. Do not promote Haj Grant ChanA yet — live 160/700 still fy-gate=0.  
9. **Probe 2026-09-01 (not production):** spatial line-scans + fx-column families. Frame 160 vertical fringes are an fx family; line-scan qx beat the tallest FFT fx peak.  
10. **Probe 2026-09-01 evening:** congruent 4-cut seed + 10-frame compare. Combo 6/10 PASS vs original 4/10. Next: fy and fx as **separate** image-check candidates (congruence proposes fx; rank fy families by image-test). Do not promote Haj Grant ChanA — 700 still gate=0. Details: `notes/HANDOFF.md`.  
11. **Probe 2026-09-02:** v3 integrated pipeline (`process_v3`) — shutter + linescan + leftover FFT + image-test + union apply into `defringe_v3/`. Schematic `v3_pipeline_schematic.pdf`. Not a promote.  
12. **Coded 2026-09-02 evening:** v4 one-mask-per-frame (`process_v4`). Catalog/shutter-learn guesses; linescan thin-peak + chirp edges; leftover FFT last. Writes `defringe_v4/` (not v22). ChanA/ChanB not coupled. Not a promote. `notes/V4_PIPELINE.md`.  
13. **Later:** x-walk proposer, overlapping windows, **no tile cleans**.  
