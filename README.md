# derippling_PMT_noise

Remove periodic PMT / scan-electronics fringe noise from imaging stacks, then
(optionally) denoise with SUPPORT.

## Leading method (current)

**Raw-only adaptive ridge-segment filter** (GPT v2):

`reference/gpt/pmt_fringe_raw_adaptive.py`

Recommended production flow:

```text
raw TIFF  →  v2 defringe  →  SUPPORT / model_10
```

No SUPPORT step is required *before* defringing. Detect, track, and gate
directly on raw; leave weak/absent-fringe frames nearly untouched.

```bash
python reference/gpt/pmt_fringe_raw_adaptive.py ChanA_stk.tif ^
  -o ChanA_defringe/ChanA_stk_defringed.tif ^
  --diagnostics ChanA_defringe/diagnostics
```

## Other code in this repo (pre-streamline)

| Path | Role |
|------|------|
| `deripple.py` | Claude-style full FFT-row harmonic notches |
| `reference/gpt/pmt_fringe_adaptive.py` | Earlier GPT point-notch adaptive filter |
| `bakeoff_compare.py` | Compare methods on raw test stacks |
| `stress_test_v2.py` | Continuity + pseudo-ground-truth injection |
| `compare_support_blocks.py` | SUPPORT(raw) vs SUPPORT(defringed) block compare |
| `make_results_pdf.py` / `make_support_compare_pdf.py` | PDF reports |
| `notes/` | Design notes from Claude/GPT discussions |

Bake-off / stress / SUPPORT compare outputs were written next to the data under:

`...\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests\`

## Hierarchy (for minimal biological alteration)

`gpt_raw_adaptive_v2` > old point-notch > whole-row band

## Notes

- Claude excerpts: `notes/claude_share_excerpts.md`
- GPT reference scripts: `reference/gpt/`
