# Dark-current fringe controls — first report

**Date:** 2026-08-25
**Code:** `darkcurrent/` (`python -m darkcurrent`)
**Data:** `F:\bPACNewData2026\DataTrials260216_CavscAMP_PNAS\DarkCurrent\260214`
**Artifacts:** `<data root>\.darkcurrent_analysis\20260825T160621Z\`
**Machine-readable:** `darkcurrent/registry.json` (what exists), `darkcurrent/measurements.json` (what was measured, appended per run)

Purpose: measure the PMT fringe layer directly, with no biology competing, so
seeding can *verify a known family* instead of searching blindly. This is
calibration material — nothing here writes into experiment folders or the
`batch_defringe` prior cache.

---

## 1. What is actually in the folder

Four trials on **Shinano** (`Computer="THORLABS_30_016"` — confirmed as the same
rig, 2026-08-25), recorded 2026-02-14, 512×512, 29.595 Hz, fieldSize 178,
pixelSizeUM 0.935, `scanMode=0`, `twoWayAlignment=0`, `averageMode=0`,
`averageNum=2`, PMT gain **50/50** on both channels.

| Trial | Pockels setting | Frames | Assembled stacks | Note |
|---|---:|---:|---|---|
| `PC050` | 50 | 1600 | ChanA, ChanB | usable |
| `PC150` | 150 | 5000 | — | aborted, excluded |
| `PC150_001` | 150 | 1600 | ChanA, ChanB | usable (retake) |
| `PC250` | 250 | 1600 | ChanA, ChanB | usable |

Raw per-frame TIFFs (1600 per channel) plus an assembled
`DATA/Chan{A,B}/Chan*_stk.tif`, which is the layout the analysis uses.

### Settled points

**`THORLABS_30_016` is Shinano.** One rig, so the shared tag across this set, the
Level3b sandbox and the Haj Grant trial is correct and the prior cache is not
merging different microscopes. `darkcurrent/metadata.py` now carries the alias.

**Pockels level comes from the folder name, not the XML.** The trial names give
the real settings (50 / 150 / 250). ThorImage stored `<Pockels start="20.5">` for
`PC050`, `PC250` and `PC150`, and `0` for `PC150_001`, which tracks nothing
useful. The tooling now reads the level from the label and reports the XML value
beside it only as a known-unreliable field. **Any future analysis should key on
the folder name.**

**PMT gain is identical (50/50) everywhere**, so this set contains no gain series.

---

## 2. How the measurements are made

Dark current is mostly white detector noise, so the raw power inside any ridge
mask mostly reports *how many bins the mask has*. Every number below is
therefore an **excess over a matched control** placed on fringe-free rows, using
the same estimator as the production residual score (`rescore_v21.score_frame`):
excess linear amplitude inside the ridge's `fx` support, measured against a local
row background.

An early version of this analysis used masked-field RMS and produced nearly
identical values for both channels — that was measuring mask size, not fringe.
The control-referenced version separates them (tile SNR now ranges from ~1 to
>10³ instead of being constant).

Detection itself reuses the production detector from
`reference/gpt/pmt_fringe_raw_adaptive.py`, at both production thresholds and a
relaxed diagnostic pass, so "nothing detected" stays distinguishable from
"nothing present".

---

## 3. Findings

### 3.1 The ChanA `q≈6` family is real

This is the question the whole conservative seeding posture rests on, and these
recordings answer it: **in a stack with no biology, ChanA shows a strong paired
family at `q≈5–7`** (period 73–102 px), with narrow `fx` support at `|fx|≈11–41`
— exactly the band the repo attributes to the PMT ridge.

| Trial / channel | Chosen `q` | Excess | Control | SNR |
|---|---:|---:|---:|---:|
| PC050 / ChanA | 7 | 13437 | 76 | 177 |
| PC250 / ChanA | 6 | 35323 | 132 | **269** |
| PC050 / ChanB | 54 | 56901 | 93 | **612** |
| PC250 / ChanB | 55 | 30247 | 83 | 363 |
| PC150_001 / ChanA | 204 | 13711 | 138 | 100 |
| PC150_001 / ChanB | 7 | 6570 | 9 | 744 |

So `q≈6` is **not** purely a shading artifact of structured biology. It is
measurable structure with ridge-segment `fx` support. That does not automatically
make it safe to notch on real data — but "avoid `q≈6` because it is an artifact"
is no longer a supportable justification, and the reject-band idea floated during
the 2026-08-25 policy discussion should be dropped.

Caveat, now resolved: `q≈6` sits close to DC, which is the hardest place to
measure, and the row background for `q=6` reaches rows adjacent to DC. That
caveat was well founded — it turned out to be worse than "adjacent" (§3.1c) — but
the conclusion survives an independent re-measurement that never touches a
DC-adjacent row. See §3.1c.

### 3.1b Why the pipeline warned about `q≈6` in the first place

Worth recording, because the answer is *not* "we measured biology contamination".
The origin is a single comparative judgement, in `run_full_v21_seeded500.py`:

> Why: median spectrum over all 5400 frames can prefer a weak spurious family
> (ChanA q=6) over the clear 500fr signature (q=14).

Two candidates competed on sandbox ChanA. The median over all 5400 frames
preferred `q=6`; detection on the fringe-rich 500-frame slice gave `q=14`, which
looked strong and cleanly paired. `q=6` was therefore called **weak** and
**spurious**, and the fix was to seed from the 500 fr slice.

The attribution to biology was an inference, never a test. It was plausible:
averaging 5400 frames mixes in a lot of varying structure and low-frequency
shading, `q=6` sits near DC where exactly that power lives, and no fringe-only
recording existed to check against. But it was a *relative* statement — weaker
than `q=14` on one window — which then hardened into *categorical* language as it
propagated: "wrong q" in `HANDOFF_SUPPORT.md`, "**Do not** use" in
`HANDOFF_SUITE2P.md`, "the ChanA q=6 trap" in `OPTIMIZATION_STATUS.md`. Nothing
new was measured between those statements.

These controls are the closest available test, because they share the sandbox
scan configuration (29.595 Hz, 0.935 µm, fieldSize 178, mag 27.77, `scanMode=0`;
differing in `twoWayAlignment`, `averageMode` and gain). They show real ridge
structure at `q≈5–7`. The `fx` support also matches: `|fx|≈11–41` here versus
`|fx|≈10–38` for the accepted sandbox `q=14` family.

Leading interpretation: **`q=6` and `q=14` are both real and live in the same
`fx` band** — plausibly the same phenomenon sampled at different times, or two
members of one comb — and the full-stack median surfaced whichever dominates on
average while the fringe-rich window surfaced whichever dominates when the fringe
is strong. On that reading `q=6` was never spurious; it was just weaker in the
window that was inspected.

This does **not** mean the 500 fr seeding decision was wrong in effect. Seeding
from a fringe-rich window is still the right instinct, and the resulting cleans
scored well. What was wrong was the *reason* recorded for it, and the categorical
warnings built on top.

### 3.1c Independent confirmation of the near-DC family

**Date:** 2026-08-26
**Code:** `darkcurrent/confirm.py` (`python -m darkcurrent confirm`)
**Artifacts:** `<data root>\.darkcurrent_analysis\confirm_20260826T081515Z\confirm.json`

This is §9 item 1, closed. The near-DC ChanA family is **real, narrow in `fx`, and
time-varying**, measured without any DC-adjacent background.

**First, the defect was worse than §3.1 assumed.** The legacy estimator's
background for a candidate at `q` is offsets ±5-9 from row `cy±q`. For `q=6` that
set is rows `cy-3 … cy+1` — it **contains the DC row itself**. The probe reports
this per candidate, and every near-DC candidate in every trial came back
`nearest_bg_dy = 0`, `uses_dc_row = True`. The contamination also reached the
*support*: `ridge_z_at_row` builds its background from offsets ±4-10, so the
`|fx|≈11-41` figure quoted in §3.1 was itself derived across DC. Neither number
was independent.

**What replaced it.** Four changes, each forced by a control that failed:

1. Frames are high-passed along y and Hann-apodised before the FFT, killing the
   offset and `q≲1` structure that feed the DC pedestal. Without this, the steep
   pedestal's *curvature* alone produced a "ridge" at `q≈6` with `z≈160`.
2. The background along `fx` is a wide median filter **within the same row**, so
   no other row is consulted at all.
3. Support is chosen on one half of the sampled frames and measured on the other.
   Without the split, a pure-noise row scored 3-15× the noise floor instead of the
   0.4 it should, because every row got credit for its own best-looking bins.
4. The filter's gain at each `q` is calibrated with a synthetic sinusoid of known
   amplitude, so a near-DC candidate is not penalised for sitting near the
   high-pass corner (measured gain 0.22-0.30, against 0.25 for an unattenuated
   tone).

Thresholds are calibrated per channel against an empirical null over all ~194
rows not under test, with robust rejection of the rows that carry structure. The
false-positive rate on 24 verified-empty rows was **0/24 in all six channels**.

| Trial / channel | `q` | Period px | Amp SNR | (thresh) | Prominence | (thresh) | Temporal peak | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PC050 / ChanA | 5 | 102.4 | 165 | 8.1 | 178 | 10.8 | +0.045 c/fr (+1.33 Hz) | **confirmed** |
| PC050 / ChanA | 8 | 64.0 | 82 | 8.1 | 80 | 10.8 | −0.014 (−0.40 Hz) | **confirmed** |
| PC250 / ChanA | 6 | 85.3 | 153 | 5.4 | 20 | 9.8 | +0.129 (+3.81 Hz) | **confirmed** |
| PC250 / ChanA | 9 | 56.9 | 110 | 5.4 | 52 | 9.8 | −0.020 (−0.58 Hz) | **confirmed** |
| PC150_001 / ChanA | 3 | 170.7 | 132 | 30.8 | 6351 | 12.7 | −0.389 (−11.50 Hz) | **confirmed** |
| PC050 / ChanB | 3 | 170.7 | 6.9 | 11.9 | 6.9 | 11.9 | — | noise-like |
| PC150_001 / ChanB | 7 | 73.1 | 12.9 | 51.5 | 27.1 | 12.9 | +0.166 | moving, not localised |

The near-DC candidates are the **strongest** structures in ChanA, at 20-30× their
own channel's threshold. Two of them are confirmed per trial, spaced by 3
(`5 & 8`, `6 & 9`) — worth remembering as a possible sideband pair rather than two
independent families.

**The decisive part is the temporal test**, because it needs no spatial
background model. The complex FFT coefficient at each support bin is tracked over
512 consecutive frames and its two-sided temporal power spectrum averaged over
bins. Static structure — including anything that is really DC leakage — peaks at
`f=0`, because it inherits DC's behaviour. A drifting fringe peaks elsewhere.
Every confirmed candidate peaks at a **non-zero** frequency, while the `fy=0`
reference row peaks at exactly `f=0` in three of six channels. So the near-DC
ChanA structure is not static shading and is not DC leakage: it moves.

The method was validated on synthetic stacks with known contents before being
trusted here (`python -m darkcurrent.validate_confirm`, three scenarios, all
passing). Each stack carries a steep static shading pedestal, so the battery has
to survive a DC skirt it did not build. The case that matters most is the second
scenario: a ridge planted at `q=6` that does *not* move is reported as static
structure rather than as a fringe, while the same `q=6` planted with drift — and
again with 0.8 rad of per-frame phase jitter, which is the regime §3.3 measured
on real data — is confirmed. So the test discriminates the exact confound §3.1b
worried about, and does so without needing a clean phase. Across the three
scenarios: every planted component got its expected verdict and 0/24 null rows
passed.

**What this does not establish.** The battery is deliberately conservative, and
several known-real families fail it:

- ChanB `q=54` (PC250) confirms cleanly, but `q=95` (PC050) lands in
  `structure_uncharacterized` — solid spatial evidence, temporal peak not
  resolvable. That outcome exists on purpose: §3.3 measured a phase that wanders
  1.1-1.8 rad per frame, which spreads temporal power, so demanding a sharp
  temporal peak would reject real families. Absence of a temporal peak is not
  evidence against a family here.
- `PC150_001` ChanB rejects both its near-DC candidate and its own strongest
  family (`q=98`, `row_z=53.6`) on amplitude, because in that channel the *median*
  empty row already scores 12.4 and 14 rows carry ridge-like structure. Nothing
  stands out there against its own background. This is the anomalous near-dark
  trial flagged in §3.4 and is a finding, not a bug.
- Weak far-from-DC families (`q=24`, `27`, `38`, all `row_z<4.5`) do not clear
  their channel thresholds.

**One loose thread worth a cheap check.** In `PC150_001` ChanA both the `q=3` row
and the `fy=0` reference peak at ∓0.3887 cycles/frame (∓11.50 Hz) with enormous
prominence — same frequency, opposite sign, so a global oscillation of the whole
frame rather than a spatial family. At 29.595 Hz frame rate that aliases to
`11.50 + 3×29.595 = 100.3 Hz`, suspiciously close to twice mains. It is 0.3 Hz
off, which exceeds the 0.06 Hz frequency resolution, so this is a hypothesis and
not a measurement: it would require the true frame rate to be ~29.50 rather than
the nominal 29.595. Testable against a recording with a known mains environment.

### 3.2 The fringe is strongly edge-weighted along x — the most actionable result

Fringe excess is **not spatially uniform**. Measured in full-height column strips
(so vertical frequency resolution is preserved), it concentrates at the extreme
left and right of the field:

```
PC050 ChanA: 1518  536  216  100  214   76  524  321  332  528   89  189  102  235  426 1473
PC050 ChanB: 2287 1100  316   90  234  118  810  466  488  803  118  241   92  334  904 2319
PC250 ChanB: 1923  912  250   77  221  104  675  395  412  638  103  199   81  285  740 1918
```

Edge windows run **15–26× higher** than the weakest central window, with a
roughly symmetric profile and a secondary central bump. This is the expected
resonant-scanner signature: the fast axis dwells longest at the turnarounds.

Implication for masking: the current cleaner attenuates ridge segments in the
**global** 2D FFT, which is uniform across x by construction. On this evidence it
necessarily over-treats the middle of the field relative to the edges. An
x-aware treatment (windowed or x-weighted) could remove more fringe at the edges
while touching the centre less — the direction the question about "defringing
more efficiently" was pointing at.

### 3.3 Frequency drifts slowly; phase does not stay put

The 1600-frame recordings do capture drift, and the two behave very differently.

**`q` drift is slow and stepwise — on ChanB.** PC050 ChanB tracks a clean
staircase: `q=55` for roughly frames 0–200, `54` for 200–1100, `53` for
1100–1600 — total span 2 over ~54 s. A soft prior plus per-block tracking is
therefore the right architecture for ChanB: the mask *location* is predictable.

**ChanA is not established.** PC250 ChanA stayed within `q=5–8` (span 3), but
PC050 ChanA reported `q=5–15`, which is **exactly the full width of the ±8 search
window** around its `q0=7`, clipped at the `q≥5` floor. A tracker that visits
every candidate it is allowed to visit has told us nothing about drift; it means
tracking is unstable this close to DC. Do not read the PC050 ChanA range as a
measured 5→15 drift. Re-run ChanA with a wider window and a stability criterion
before claiming anything about its drift. (Noted because `q=14` falls inside that
window — see §3.1b — so this run cannot be used as evidence either way.)

**Phase is not stable frame to frame.** Measured on consecutive frames, the
median step is ~1.1–1.8 rad. The unwrapped trace drifts smoothly overall, so it
is not pure noise, but it moves far too much to treat as fixed. This rules out
building a static fringe template and subtracting it, and it supports the
existing per-frame Fourier attenuation design.

### 3.4 Fringe amplitude follows the excitation light, non-monotonically in setting

Temporal mean excess, ordered by the Pockels setting from the folder names:

| Channel | PC 50 | PC 150 (`PC150_001`) | PC 250 |
|---|---:|---:|---:|
| ChanA | 37499 | **11642** | 28198 |
| ChanB | 25916 | **2887** | 25971 |

The middle setting is the *weakest*, by ~9× on ChanB. Intensity statistics agree:
`PC150_001` ChanA maxes at 232 ADU versus 2340 in `PC050` — essentially no
photons. The x-edge dominance also disappears there (edge/middle 0.65 and 1.03,
versus 1.85–2.27 in the other two).

So the response is **not monotonic in the nominal setting**. Two candidate
explanations, both testable and not yet distinguished:

- **A Pockels transmission null near 150.** A Pockels cell's transmission goes as
  `sin²` of the applied voltage, so a three-point series can straddle a minimum
  and read high–low–high. This would make `PC150_001` a genuine
  near-zero-light control by accident, which is useful.
- **A blocked path in that trial** (shutter, or laser not on) — a plain
  acquisition fault, in which case `PC150_001` is a dark control but tells us
  nothing about Pockels response.

A finer sweep (say 6–8 settings from 0 to 300) would separate these immediately
and is cheap to record.

Either way, the physics conclusion holds: **the dominant fringe in `PC050` and
`PC250` requires excitation light to be present**, so it is not a PMT-only
electronics artifact, and these are not purely dark-current recordings despite
the folder name. Only ridge *geometry* can transfer from a control to real data;
amplitude cannot, so each stack must keep deciding its own attenuation, as the
cleaner already does.

One anomaly: `PC150_001` ChanB still peaks at 2546 ADU and shows the single
strongest detection in the set (`row_z=53.6` at `q=98`) despite having the least
light. Either light leaks into ChanB, or ChanB's PMT contributes something of its
own. Worth a look.

### 3.5 ChanB carries a family comb

ChanB is consistently strongest at `q≈53–55` (period ~9.5 px) and also shows
`q≈95–98` (~5.2 px) and `q≈157–161` (~3.2 px). The `hi` partners the detector
reports (202, 158, …) are just `cy − q`, so the independent families are roughly
`q ≈ 54`, `98`, and possibly `21` and `7`.

---

## 4. Configuration coverage — the gap that matters

These controls match the **Level3b sandbox** configuration on every field the
prior-compatibility check inspects, and do **not** match the Haj Grant Example
trial whose ChanA failed to seed:

| Field | DarkCurrent | Level3b sandbox | Haj Grant Example |
|---|---:|---:|---:|
| frameRate | 29.595 | 29.595 | **15.136** |
| pixelSizeUM | 0.935 | 0.935 | **1.623** |
| fieldSize | 178 | 178 | 178 |
| magnification | 27.77 (25x) | 27.77 (25x) | **16.0 (16X)** |
| scanMode | 0 | 0 | **1** |
| twoWayAlignment | **0** | −17 | −12 |
| averageMode × Num | **0 × 2** | 1 × 2 | 1 × 6 |
| PMT gain A/B | **50 / 50** | 60 / 60 | 40 / 40 |

`fingerprint_compatible` would reject these controls for the Haj Grant trial on
frame rate (Δ14.5 versus tolerance 0.5) and pixel size (Δ0.69 versus 0.05). So
**this set cannot seed the stack that actually failed.**

Also note `twoWayAlignment=0` here versus −17 and −12 in real recordings. That
parameter is the bidirectional line-phase correction, which is precisely the kind
of line-to-line timing that could shape a fringe. Even for the sandbox
configuration, these controls differ from the data on that field.

---

## 5. Limits of this analysis

- **Picking `q` by best SNR is fragile.** Where the control estimate is tiny the
  ratio explodes: `PC150_001` ChanB chose `q=7` on SNR 744 (control 8.8) while
  the detector's strongest family was `q=98` at `row_z=53.6`. SNR needs a floor
  on the control, or corroboration by `row_z`, before it drives any decision.
- **Near-DC measurements** (`q≈5–7`) share rows with DC leakage; treat §3.1 as
  strong but not yet independently confirmed.
- **Per-tile "assumption-free" `q`** uses 128 px tiles, so its frequency
  resolution is 4× coarser and many central tiles have weak dominance
  (`z` down to 1.5). The reported `q` spread of 124–216 is suggestive of
  character varying across the field, not proof.
- **ChanA `q` tracking saturated its search window** in PC050 (see §3.3), so this
  set does not constrain ChanA drift.
- **One session only** (2026-02-14), one scan configuration, three Pockels
  settings of which one delivered almost no light, one gain. No repeat over time
  yet, so nothing here speaks to stability across days or hardware service.

---

## 6. What to record next

Highest value first, given the aim of a pipeline that runs the same way every
time:

1. **Controls at the configurations actually used for data**, especially the Haj
   Grant style (16X, 15.136 Hz, 1.623 µm, `scanMode=1`, `averageMode=1×6`,
   gain 40). Without this the calibration cannot help the case that failed.
2. **Match `twoWayAlignment` to the experimental value** rather than 0, so the
   control shares the line-phase behaviour of real recordings.
3. **A genuine laser-blanked control** (shutter closed) *paired* with a light-on
   control at the same settings, since §3.4 shows these measure different things.
   Label the pair unambiguously.
4. **A finer Pockels series**, 6–8 levels from 0 to 300, to resolve the
   non-monotonic response in §3.4 and settle whether 150 sits on a transmission
   null or whether that trial simply had a blocked path. Record the level in the
   folder name as before — the XML does not capture it.
5. **A gain series** (e.g. 40 / 50 / 60) at one configuration, to confirm gain
   changes amplitude but not `q` — which is the assumption behind excluding gain
   from the calibration key.
6. **Longer or repeated recordings** if drift across a whole session matters;
   1600 frames (~54 s) shows a 2-step `q` staircase, so a several-minute
   recording would bound the drift range far better.

A repeat of the same configuration on a later date would also start the time
series the registry is built for.

---

## 7. Reproducing

```bash
python -m darkcurrent census --root "F:\bPACNewData2026\DataTrials260216_CavscAMP_PNAS\DarkCurrent" --registry
python -m darkcurrent characterize --root "F:\bPACNewData2026\DataTrials260216_CavscAMP_PNAS\DarkCurrent" \
  --sample-n 128 --tiles 4 --temporal-every 8 --phase-count 300 --metrics
python -m darkcurrent confirm --root "F:\bPACNewData2026\DataTrials260216_CavscAMP_PNAS\DarkCurrent"
```

Per-channel figures are written as `<trial>_<channel>_summary.png`: FOV tile
excess, tile SNR, excess versus x, per-tile dominant `q`, excess over time,
consecutive-frame phase, tracked `q` per block, and candidate evidence.

`confirm` writes `confirm.json` only (no figures) and takes ~1 min for six
channels. It re-tests candidates without DC-adjacent backgrounds (§3.1c) and
reports, per channel, the empirical null and its false-positive rate. Use
`--q 6,14` to force specific rows, `--only`/`--channel` to restrict, and
`--temporal-frames` to trade run time against temporal frequency resolution.

The battery's own self-test needs no data and takes ~30 s:

```bash
python -m darkcurrent.validate_confirm
```

It plants known components in synthetic stacks and fails loudly if any verdict
changes (§3.1c). Run it after touching `darkcurrent/confirm.py` — the thresholds
in there are easy to move by accident, and this is what catches it.

---

## 8. Open questions for the pipeline

Nothing in `batch_defringe` has been changed on the basis of this report. The
findings that would feed a seed-policy change:

- `q≈6` on ChanA is real structure, **independently confirmed** without
  DC-adjacent backgrounds and shown to be time-varying rather than static
  shading (§3.1c), so it should not be reject-banded on the assumption that it is
  an artifact. The historical "spurious / trap" language was an untested
  inference (§3.1b); it has now been retired from `HANDOFF_SUPPORT.md`,
  `HANDOFF_SUITE2P.md`, `OPTIMIZATION_STATUS.md` and `PROGRESS.md`, each of which
  points here instead.
- Mask location is predictable on ChanB (slow stepwise `q` drift), which supports
  verifying a known family rather than searching — but only where a control
  exists for that configuration. ChanA is not yet established.
- Fringe is edge-weighted in x, which a global FFT mask cannot express.
- Amplitude is light-dependent, so only geometry may ever be imported from a
  control recording.

---

## 9. State pinned for the next session

**Where this stands.** The `darkcurrent/` package does census, characterisation
and independent confirmation. One dataset has been analysed end to end. No
production code has been touched: `batch_defringe` behaves exactly as before, and
nothing writes into experiment folders or the defringe prior cache.

**Settled, do not re-litigate.**

- `THORLABS_30_016` = Shinano, one rig. Prior sharing across Level3b, Haj Grant
  and these controls is legitimate.
- Pockels level comes from the trial folder name; the XML field is unreliable and
  is now labelled as such in code and CLI output.
- `PC150` (5000 frames, no `DATA/`) is aborted and excluded everywhere.
- The `q≈6` warning originated as a relative "weaker than q=14 on one window"
  judgement, not a measurement of biology contamination (§3.1b).
- **The near-DC ChanA family is real and moves** (§3.1c). Confirmed with in-row
  backgrounds, split-half support selection, per-channel calibrated thresholds
  and 0/24 false positives; a static ridge planted at the same `q` in synthetic
  validation is correctly rejected. Do not re-open "is `q≈6` an artifact".
- Every family also appears at its Nyquist partner `cy − q` (§3.5). Treat
  `{q, cy−q}` as one family; a "control" at `q=250` *is* the `q=6` candidate.

**Best current interpretation, not yet confirmed.** `q≈6` and `q≈14` on ChanA are
both real and share the same `fx` band, so they are likely one phenomenon or one
comb rather than a true-versus-false pair.

**Next steps, in order.**

1. ~~Independently confirm the near-DC `q≈5–7` measurement.~~ **Done, §3.1c.**
2. Re-run ChanA `q` tracking with a wider search window and a stability
   criterion, so §3.3 can say something about ChanA (currently window-saturated).
   `confirm` now gives a per-channel amplitude null that the tracker could use as
   its stability criterion instead of a fixed threshold.
3. Put a floor on the control estimate in the SNR ranking, or require `row_z`
   corroboration, before SNR is allowed to pick `q` (§5). Partly addressed:
   `confirm` normalises by a robust noise scale that cannot be zero, after the
   old ratio produced SNR values of order 10¹⁶ on quiet rows. `characterize`
   still uses the old estimator.
4. Test whether `q≈6` and `q≈14` co-occur on the sandbox ChanA stack — that is
   the direct check on §3.1b, and it needs no new recordings. `confirm --q 6,14`
   does exactly this run.
5. Only then revisit seed policy in `batch_defringe`. The `needs_review` outcome
   on Haj Grant ChanA is still correct behaviour: no control exists at that
   configuration (§4), so there is nothing to verify a family against.

**Two cautions for whoever picks this up.**

- `row_z` (95th percentile across `fx`) is blind to narrow ridges: a 6-bin
  segment out of ~480 valid bins does not move it. It is fine for finding the
  broad families but must not be used to decide that a row is *empty*.
- Selecting an `fx` support and measuring its excess on the same spectrum
  inflates everything, including any null built that way. Split the frames.

**New recordings would unblock:** items in §6, most importantly a control at the
Haj Grant configuration.

**Artifacts.** `darkcurrent/registry.json` (acquisition census),
`darkcurrent/measurements.json` (run history, appended per run), and figures
under `<root>/.darkcurrent_analysis/<run>/`.

**Which run to trust.**

For `characterize`: only `20260825T160621Z` — the one recorded in
`darkcurrent/measurements.json`. Four earlier directories from the same afternoon
(`20260825T1554…` through `20260825T1604…`) are development runs made *before*
the amplitude metric was fixed; they report masked-field RMS instead of spectral
excess and their FOV and temporal numbers are not comparable. They were left on
disk rather than deleted; ignore them, or clear them by hand.

For `confirm`: only `confirm_20260826T081515Z`. The earlier `confirm_smoke` and
`confirm_20260826T0752…` through `…T0812…` directories are development runs from
the same morning, each superseded because a control caught a flaw in the method —
in order: the row-permutation null inflated the `fy` axis and put a real family
*below* its own null; the positive control was drawn from a candidate's Nyquist
partner; and support was selected and measured on the same spectrum. Their
verdicts are wrong and their thresholds are not comparable. Same disposition:
ignore, or clear by hand.
