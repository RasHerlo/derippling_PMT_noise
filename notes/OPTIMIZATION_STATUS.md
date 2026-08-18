# Defringe optimization status (for overview repo)

**Collected by (do not edit from here):** https://github.com/RasHerlo/figure_for_cAMP_Neu_paper  
This file + `optimization_manifest.json` are the handoff surface for that repo.

**Stage:** PMT fringe removal (pre-SUPPORT, pre-suite2p)  
**Repo:** https://github.com/RasHerlo/derippling_PMT_noise  
**Sandbox data:** `F:\bPACNewData2026\PreProcessing Optimization\Level3b copy`  
**Machine-readable twin:** [`optimization_manifest.json`](optimization_manifest.json)  
**Last updated:** 2026-08-18

---

## Currently chosen approach

| Field | Value |
|---|---|
| **ID** | `gpt_raw_adaptive_v21_pack_B` |
| **Script** | `reference/gpt/pmt_fringe_raw_adaptive_v21.py` |
| **Idea** | Detect paired PMT ridge families on **raw**, track `q` over time, attenuate only excess FFT amplitude on recurrent `fx` segments; soft-gate so weak frames are untouched; high-confidence frames get stronger attenuation + residual second pass |
| **Defaults** | `max_alpha=0.85`, `max_alpha_high=1.0`, `high_strength=0.18`, `strength_span=0.12`, `residual_strength_min=0.05`, `residual_alpha=0.95` |
| **Pipeline slot** | `raw → defringe → SUPPORT → (suite2p MC)` |

### Pros
- Raw-only (no SUPPORT needed before defringe)
- PMT-agnostic detection (ChanA ≈ q=14/242, ChanB ≈ q=60/196 found automatically on 500fr)
- Leaves absent-fringe frames unmodified (gate≈0 removed RMS = 0)
- Narrow ridge segments — less biological risk than full FFT-row notches
- Measurable residual fringe excess; ChanB already ≪10% on strong frames

### Cons / risks
- ChanA strong frames still ~9% residual ridge excess (inside ~10% band, not stretch &lt;5%)
- Full-stack **v2** and unseeded full-stack detect locked ChanA onto weak `q=6` — use **seeded** `inputs/defringed_v21` only
- Stronger high-confidence attenuation can slightly bite coefficients that also hold cell energy on strong-fringe frames
- SUPPORT trained on fringed data can still unmask leftover periods until retrain on defringed stacks

### Rejected / demoted alternatives

| ID | Why not leading |
|---|---|
| Claude full-row harmonic bands | Strong removal but over-filters weak frames; higher biology risk |
| GPT v1 point notches | Safer but under-removes dense ridge energy |
| GPT v2 (`max_alpha=0.85` only) | Best architecture, too conservative on strong frames (ChanA ~14–16% residual) |
| Global `max_alpha=0.95` without confidence scaling | Easy gain but less safe than confidence-scaled v2.1 |
| Widening full rows / lower detect thresholds | Deferred — artifact risk before aggression-on-known-ridges is exhausted |

---

## Attempts log (high level)

| When | Attempt | Result | Artifacts |
|---|---|---|---|
| earlier | Bake-off rowband vs point-notch vs v2 | v2 wins on safety | legacy cursor tests |
| earlier | v2 stress injection + SUPPORT block compare | v2 defaults OK; SUPPORT amplifies residual fringe | `stress_test_v2`, support_block_compare |
| 2026-08-13 | v2.1 + first rescore | ChanB good; ChanA partial | `rescore_v21` |
| 2026-08-18 | Sandbox baseline v2.1 500fr | A 10.6% / B 2.0% strong25 | `defringe_runs/v21_500fr/` |
| 2026-08-18 | High-confidence sweep → **pack_B** | A 9.3% / B 1.8% strong25; gate0=0 | `defringe_runs/v21_sweep_500fr/` |
| 2026-08-18 | Injection stress pack_B | α=1 residual remain A~9% / B~2%; E_rec modest | `defringe_runs/v21_packB_injection/` |
| 2026-08-18 | Full-stack seeded500 | ChanA q=14 / ChanB q=60 on 5400fr; promoted | `inputs/defringed_v21/` |

---

## Where to look (sandbox)

```text
Level3b copy/
  README_SANDBOX.md
  inputs/defringed_v21/                  ← SOTA full stacks for suite2p
  inputs/slices_500fr/raw/
  defringe_runs/
    v21_sweep_500fr/accepted/pack_B/     ← 500fr winner + avgs
    v21_packB_injection/
    v21_full_seeded500/STATUS.md
```

Repo docs: `notes/PROGRESS.md`, this file, `optimization_manifest.json`.

---

## Handoff cards for sibling agents

### SUPPORT agent (inference compare — not full retrain yet)

Read: `notes/HANDOFF_SUPPORT.md`.

Ask: SUPPORT on raw vs pack_B / `defringed_v21`; write `support_runs/`; don’t touch `mc_runs/` or overwrite `inputs/`.

### suite2p agent

**Ready now.** Use:

- `inputs/defringed_v21/ChanA/ChanA_stk_defringed_v21.tif`
- `inputs/defringed_v21/ChanB/ChanB_stk_defringed_v21.tif`

Do **not** use `inputs/defringed` (legacy v2 / ChanA q≈6). Do not rename `mc_runs/` or `inputs/raw|defringed|support`.

---

## Next in this repo

1. ~~Stack averages~~ / ~~injection~~ / ~~full-stack seeded promote~~ done
2. SUPPORT inference compare + later retrain on `defringed_v21`
3. Optional: harden full-stack detection so 500fr seed is unnecessary
4. Repo streamline (deferred)

### Injection snapshot (alpha=1)

| Channel | noise | E_recovery | remaining fringe-band |
|---|---|---:|---:|
| ChanA | packB_residual | 3.76 | 9.2% |
| ChanA | broad_spectral | 16.65 | 26.6% |
| ChanB | packB_residual | 4.93 | 2.2% |
| ChanB | broad_spectral | 18.48 | 6.7% |

Realistic `packB_residual`: low spoilage of I0 and remaining fringe matches on-data residual scores. Harder `broad_spectral` leaves more leftover (especially ChanA) — expected; not a reason to widen masks yet.

### Full-stack note

Naive full-stack detect prefers ChanA **q=6**; production path is **seeded from 500fr** (`run_full_v21_seeded500.py`) → `defringe_runs/v21_full_seeded500/` + `inputs/defringed_v21/`.
