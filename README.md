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

Same path as batch (`process_stack`): fringe-rich seed, library/cache prior, QC,
`defringe_v22/` report (including `needs_review`).

```bash
python reference/gpt/pmt_fringe_raw_adaptive_v22.py ChanA_stk.tif
# optional explicit output:
python reference/gpt/pmt_fringe_raw_adaptive_v22.py ChanA_stk.tif ^
  -o ChanA/defringe_v22/ChanA_stk_defringed_v22.tif
```

### Batch over an experiment root — `batch_defringe` v0.4.0

Package: `batch_defringe/` (`python -m batch_defringe`).

**Default run:** opens a folder dialog, then walks the chosen root for
`DATA/**/ChanA_stk.tif` and `DATA/**/ChanB_stk.tif`, runs **v2.2 / pack_D**, and
writes each channel into a `defringe_v22/` folder beside the raw stack. Existing
cleaned stacks in that folder are skipped (and reported). Soft priors come from
the committed library (`fringe_library/catalog.json`, branches A/B/C) and from
`{root}/.defringe_cache/` (Computer × channel). Raster match ignores objective
`mag` / `pixelSizeUM`. Failed seeds still write `defringe_v22/overview.pdf`.

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

Outputs (raw left untouched), under `DATA/.../ChanA/defringe_v22/` (same for ChanB):

```text
ChanA_stk_defringed_v22.tif   # cleaned stack
ChanA_stk_removed_v22.tif     # raw − cleaned (float32 remainder)
per_frame.csv                 # gate / q / RMS per frame
families.json                 # FFT ridge mask definition + summary
mask_fft.tif                  # 2-D FFT-domain support of seeded families
mean_raw.tif / mean_cleaned.tif / mean_removed.tif
overview.pdf                  # averages, mask, cleaning heaviness vs frame
                              # (also written on needs_review, with inspect-only ladder)
overview.png                  # same page as a preview
ladder.json                   # only on needs_review: extra candidate rungs (not applied)
```

Priors + run logs: `{root}/.defringe_cache/`. Learned geometry also appends to
`fringe_library/catalog.json` (q/fx only, never amplitude).

#### Why it is designed this way

| Choice | Reason |
|--------|--------|
| **Single-stack CLI = batch** | `pmt_fringe_raw_adaptive_v22.py` calls the same `process_stack` as `python -m batch_defringe`. |
| **Soft priors, not hard locks** | Within a group, scans share PMT/settings but `q` still drifts (even within one stack). Priors guide tracking; each stack can reseed if QC fails. Library families must still show fx support on **this** stack. |
| **Cache by `Computer` × channel** | `Experiment.xml` `<Computer name="..."/>` marks which microscope recorded the trial (e.g. `THORLABS_30_016` vs `USER-PC`). ChanA/ChanB use different PMTs, so priors are never shared across channels. |
| **Raster fingerprint** | Match `pixelX/Y`, listed `frameRate`, `fieldSize`, `scanMode`, averaging N, flyback, and two-way alignment when `scanMode=0`. **Not** `mag` / `pixelSizeUM`. Effective stack fps = listed / N. |
| **Fringe-rich recurrent seed** | Families that recur across 50-frame blocks are preferred over a single strongest block. |
| **Track lock** | `track_search=10` with identity lock so one family cannot hop onto another's `q`. |
| **Inspect-only ladder** | If the safe paired seed fails, extra rungs (standalone, lower z, library q) are written to the PDF — not applied. |
| **Per-stack sanity checks** | Track-update fraction and `q` drift vs prior; failures go to `needs_review` **with** a report rather than silently promoting bad cleans. |
| **Root-local `.defringe_cache/`** | Priors travel with the dataset; other workstations reuse them after the same `pip install -r requirements.txt` setup. |
| **Skip existing outputs** | Safe re-runs on large trees; already-defringed stacks in `defringe_v22/` are reported as skipped. |
| **Readout next to the clean** | Every successful clean writes the remainder stack, per-frame numbers, FFT mask, and a one-page PDF so later processing can track what was removed. |

Sandbox helper that still uses an explicit 500fr seed: `run_full_v22_seeded500.py`.  
Status for overview: `notes/OPTIMIZATION_STATUS.md`

## Dark-current controls — `darkcurrent/` v0.2.0

Calibration recordings (no sample) used to measure the fringe layer itself, so
seeding can verify a known family instead of searching blindly. Kept separate
from `batch_defringe`: these are controls, not data to clean, and nothing here
writes into experiment folders or the defringe prior cache.

```bash
python -m darkcurrent census --root "...\DarkCurrent" --registry
python -m darkcurrent characterize --root "...\DarkCurrent" --metrics
python -m darkcurrent confirm --root "...\DarkCurrent"
```

`census` reports acquisition settings per trial and groups them by scan
configuration. `characterize` measures ridge families, field-of-view dependence,
and temporal drift, writing figures next to the data plus a small JSON history in
the repo (`darkcurrent/registry.json`, `darkcurrent/measurements.json`) so
repeated recordings can be tracked over time.

`confirm` re-tests ridge candidates without borrowing any background from
DC-adjacent rows, which is what a family close to DC needs before it can be
trusted. It suppresses the DC skirt (high-pass along y plus Hann apodisation,
with the filter gain calibrated against a synthetic tone), takes its `fx`
background from within the same row, and picks the support on one half of the
sampled frames while measuring it on the other. A separate temporal test tracks
the complex FFT coefficient over consecutive frames, which separates a fringe
that drifts from structure that is merely fixed — static structure and DC leakage
both peak at zero temporal frequency. Thresholds are calibrated per channel
against an empirical null over every row not under test, and the fraction of
those empty rows that still pass is reported as the false-positive rate.
Verdicts: `fringe_confirmed`, `static_structure`, `structure_uncharacterized`
(real but temporal character unresolved), `moving_not_localized`, `noise_like`.

`python -m darkcurrent.validate_confirm` is the battery's self-test: it plants
known components in synthetic stacks and checks each one gets the verdict it
should, including a static ridge sitting at the same near-DC frequency as a real
family. Needs no data; run it after editing `darkcurrent/confirm.py`.

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
