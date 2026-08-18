# derippling_PMT_noise

Remove periodic PMT / scan-electronics fringe noise from imaging stacks, then
(optionally) denoise with SUPPORT.

## Leading method (current)

**Raw-only adaptive ridge-segment filter** (GPT **v2.1**):

`reference/gpt/pmt_fringe_raw_adaptive_v21.py`

Same architecture as v2 (`pmt_fringe_raw_adaptive.py`), plus:
- confidence-scaled `max_alpha` (0.85 → up to 1.00 when gate/strength are high; `pack_B` defaults)
- residual second pass on surviving paired ridges

Recommended production flow:

```text
raw TIFF  →  v2.1 defringe  →  SUPPORT / model_10
```

No SUPPORT step is required *before* defringing. Detect, track, and gate
directly on raw; leave weak/absent-fringe frames nearly untouched.

```bash
python reference/gpt/pmt_fringe_raw_adaptive_v21.py ChanA_stk.tif ^
  -o ChanA_defringe/ChanA_stk_defringed.tif ^
  --diagnostics ChanA_defringe/diagnostics
```

Re-score helper (500fr tests): `rescore_v21.py`

## Other code in this repo (pre-streamline)

| Path | Role |
|------|------|
| `deripple.py` | Claude-style full FFT-row harmonic notches |
| `reference/gpt/pmt_fringe_adaptive.py` | Earlier GPT point-notch adaptive filter |
| `reference/gpt/pmt_fringe_raw_adaptive.py` | v2 baseline (no confidence-scaled alpha) |
| `rescore_v21.py` | Run v2.1 + residual fringe-power vs v2 |
| `bakeoff_compare.py` | Compare methods on raw test stacks |
| `stress_test_v2.py` | Continuity + pseudo-ground-truth injection |
| `compare_support_blocks.py` | SUPPORT(raw) vs SUPPORT(defringed) block compare |
| `make_results_pdf.py` / `make_support_compare_pdf.py` | PDF reports |
| `notes/` | Design notes from Claude/GPT discussions |

Bake-off / stress / SUPPORT compare outputs were written next to the data under:

`...\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests\`

## Hierarchy (for minimal biological alteration)

`gpt_raw_adaptive_v21` ≥ `gpt_raw_adaptive_v2` > old point-notch > whole-row band

## Notes

- **Status for the overview repo** ([figure_for_cAMP_Neu_paper](https://github.com/RasHerlo/figure_for_cAMP_Neu_paper)): read `notes/OPTIMIZATION_STATUS.md` + `notes/optimization_manifest.json` (this repo only; overview collects).
- Progress handoff: `notes/PROGRESS.md`
- SUPPORT agent card: `notes/HANDOFF_SUPPORT.md`
- Claude excerpts: `notes/claude_share_excerpts.md`
- GPT reference scripts: `reference/gpt/`
