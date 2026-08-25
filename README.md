# derippling_PMT_noise

Remove periodic PMT / scan-electronics fringe noise from imaging stacks, then
(optionally) denoise with SUPPORT.

## Leading method (current)

**Raw-only adaptive ridge-segment filter** (GPT **v2.2** / `pack_D`):

`reference/gpt/pmt_fringe_raw_adaptive_v22.py`

Same architecture as v2.1, with stronger residual pass + softer `ratio_*`
(2026-08-19 sweep). Prior v2.1/`pack_B`: `pmt_fringe_raw_adaptive_v21.py`.

Recommended production flow:

```text
raw TIFF  →  v2.2 defringe  →  SUPPORT / retrained model  →  suite2p
```

### Setup (venv)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### Single stack

```bash
python reference/gpt/pmt_fringe_raw_adaptive_v22.py ChanA_stk.tif ^
  -o ChanA_defringe/ChanA_stk_defringed_v22.tif ^
  --diagnostics ChanA_defringe/diagnostics
```

### Batch over an experiment root — `batch_defringe` v0.2.0

Package: `batch_defringe/` (`python -m batch_defringe`).

**Default run:** opens a folder dialog, then walks the chosen root for
`DATA/**/ChanA_stk.tif` and `DATA/**/ChanB_stk.tif`, runs **v2.2 / pack_D**, and
writes `Chan*_stk_defringed_v22.tif` next to each raw stack. Existing defringed
files are skipped (and reported). Soft priors are keyed by microscope
(`Experiment.xml` `<Computer>`) × channel.

```bash
python -m batch_defringe
# or non-interactive root:
python -m batch_defringe --root "F:\bPACNewData2026\AC_cAMP_Neu_Ca_C1_C2"
python -m batch_defringe --root "..." --dry-run
```

**Expected layout**

```text
.../<trial>/
  Experiment.xml          # preferred (microscope prior)
  DATA/
    ChanA/ChanA_stk.tif
    ChanB/ChanB_stk.tif
    # also searches subfolders under DATA/
```

If `Experiment.xml` is missing: console warning + `DEFRINGE_WARNING_NO_EXPERIMENT_XML.txt`
beside the stack, and a **fresh seed** (no microscope prior).

Outputs: `DATA/.../Chan*_stk_defringed_v22.tif` (raw left untouched).  
Priors + run logs: `{root}/.defringe_cache/`.

#### Why it is designed this way

| Choice | Reason |
|--------|--------|
| **Separate from single-stack CLI** | Batch discovery, microscope caching, and QC are a different job from cleaning one TIFF; keeps `pmt_fringe_raw_adaptive_v22.py` simple. |
| **Soft priors, not hard locks** | Within a group, scans share PMT/settings but `q` still drifts (even within one stack). Priors guide tracking; each stack can reseed if QC fails. |
| **Cache by `Computer` × channel** | `Experiment.xml` `<Computer name="..."/>` marks which microscope recorded the trial (e.g. `THORLABS_30_016` vs `USER-PC`). ChanA/ChanB use different PMTs, so priors are never shared across channels. |
| **LSM fingerprint check** | Same computer with a large change in `frameRate` / `fieldSize` / pixel size invalidates pixel-`q` priors — reseeds instead of forcing a wrong family. |
| **Fringe-rich seed on long stacks** | Full-stack median detection can pick a weak/wrong ridge (seen on ChanA). First seed (and reseeds) prefer strong blocks, so dedicated 500fr seed files are not required. |
| **Per-stack sanity checks** | Track-update fraction and `q` drift vs prior; failures go to `needs_review` in the run log rather than silently promoting bad cleans. |
| **Root-local `.defringe_cache/`** | Priors travel with the dataset; other workstations reuse them after the same `pip install -r requirements.txt` setup. |
| **Skip existing outputs** | Safe re-runs on large trees; already-defringed stacks are reported as skipped. |

Sandbox helper that still uses an explicit 500fr seed: `run_full_v22_seeded500.py`.  
Status for overview: `notes/OPTIMIZATION_STATUS.md`

## Dark-current controls — `darkcurrent/` v0.1.0

Calibration recordings (no sample) used to measure the fringe layer itself, so
seeding can verify a known family instead of searching blindly. Kept separate
from `batch_defringe`: these are controls, not data to clean, and nothing here
writes into experiment folders or the defringe prior cache.

```bash
python -m darkcurrent census --root "...\DarkCurrent" --registry
python -m darkcurrent characterize --root "...\DarkCurrent" --metrics
```

`census` reports acquisition settings per trial and groups them by scan
configuration. `characterize` measures ridge families, field-of-view dependence,
and temporal drift, writing figures next to the data plus a small JSON history in
the repo (`darkcurrent/registry.json`, `darkcurrent/measurements.json`) so
repeated recordings can be tracked over time.

Two acquisition conventions this tooling relies on:

- `THORLABS_30_016` in `Experiment.xml` is the rig **Shinano**; the alias lives in
  `darkcurrent/metadata.py`.
- **Pockels level is read from the trial folder name** (`PC050` → 50). ThorImage's
  `<Pockels start>` does not track the live setting, so it is reported only as a
  known-unreliable field. Name new trials accordingly.

Report and current state: [`notes/DARKCURRENT.md`](notes/DARKCURRENT.md) —
resume point in §9.

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

`gpt_raw_adaptive_v22` ≥ `gpt_raw_adaptive_v21` ≥ `gpt_raw_adaptive_v2` > old point-notch > whole-row band

## Notes

- **Status for the overview repo** ([figure_for_cAMP_Neu_paper](https://github.com/RasHerlo/figure_for_cAMP_Neu_paper)): read `notes/OPTIMIZATION_STATUS.md` + `notes/optimization_manifest.json` (this repo only; overview collects).
- Dark-current controls report: `notes/DARKCURRENT.md`
- Progress handoff: `notes/PROGRESS.md`
- SUPPORT agent card: `notes/HANDOFF_SUPPORT.md`
- Claude excerpts: `notes/claude_share_excerpts.md`
- GPT reference scripts: `reference/gpt/`
