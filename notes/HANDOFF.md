# Pickup — 2026-08-28 evening

Continue here next session (Sat/Sun). Do **not** re-run full-stack Haj Grant
cleans until the shutter-seed path is wired in; ChanB’s existing clean is fine.

## Where we are

Haj Grant Example (`F:\bPACNewData2026\Haj Grant Example`):

| Channel | Production `defringe_v22` | Why |
|---|---|---|
| ChanB | **ok**, q≈81 | Library/live prior; overview now has original\|cleaned\|removed inspection pages |
| ChanA | **needs_review** | Stack-wide seed never found a safe pair; **do not** treat the z-ladder trial-clean as a real subtract |

The useful signal is an **in-stack shutter stretch**: frames **756–760** (761 is
already opening). Auto-detected as the only low-std run on both channels.
Mean barely drops; **std collapses** (ChanA ~220→26, ChanB ~115→32). Those
frames *are* the PMT pattern (no biology).

Learned from that stretch (not from 50-frame live blocks):

- ChanA: **q≈10** (harmonic 30), fx comb ~13–69
- ChanB: **q≈81** (harmonic 162), fx comb ~11–62 — same family the successful clean already used

Live transfer with that comb + q-track ±8:

- ChanA **1061**: q 10→12, gate on, **removed looks like shutter fringes, not cells**
- ChanB **160**: q 81→89, same
- Several live frames stay gate=0 (ridge weak under cells) — conservative, not a seed failure
- ChanA shutter flatten is **incomplete** (ghost bands; std 28→14). ChanB shutter flatten is close to grain (std 33→15)

Phase-locking / subtracting the shutter *image* is still the wrong model.
Amplitude notch + short q-track is the right class. Failure was seeding from
the wrong frames and evaluating detector z-ladders instead of “make 756–760
flat, then check live *removed* vs that pattern.”

The old ChanA eval (q=109, z=4.5–3.0) was a red herring: at z=4.5–3.5
removed rms on 760 was **0**. Frame 760 looked “successfully cleaned” because
it *is* the pattern, not because we subtracted it.

## Artifacts to inspect (not in git)

```
F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\defringe_v22\overview.pdf
F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\defringe_v22\shutter_seed_test\overview.pdf
F:\bPACNewData2026\Haj Grant Example\DATA\ChanB\defringe_v22\overview.pdf          (inspection pages rebuilt)
F:\bPACNewData2026\Haj Grant Example\DATA\ChanB\defringe_v22\shutter_seed_test\overview.pdf
```

Probe command (does not overwrite production TIFFs):

`python -m batch_defringe.shutter_seed_test`

Rebuild a success PDF only:

`python -m batch_defringe --rebuild-overview "<channel>\defringe_v22"`

## Code landed (this commit)

- `batch_defringe/eval_report.py` — needs_review original\|trial-cleaned\|removed + z-ladder
- Success `overview.pdf` — same inspection at anchors 160/1061 + strongest/weakest removal
- `batch_defringe/shutter_seed_test.py` — shutter-seed probe
- Library: Haj Grant ChanB live_clean q=81 (branch C) in `fringe_library/catalog.json`

## Pick up here

1. Finish ChanA shutter **756–760 to spatially flat** with the **smallest extra fx peaks** (not a wider row). Ghost diagonal remains.
2. On live frames where the tracked ridge **gates on**, use the same flatten depth as the shutter (pack_D leftover is milder than shutter flatten).
3. Wire in-stack shutter detection into `process_stack` (low-std run → seed q/fx comb → tight q-track). Dialog to confirm/mark ranges can wait until auto-detect is trusted.
4. New eval metric: (a) shutter std after flatten; (b) live *removed* must resemble the shutter pattern; (c) do **not** score “dominant stripes on frame 160” as the fringe when FFT says q=0 biology.
5. Only then re-run Haj Grant ChanA for a real cleaned TIFF.

SUPPORT / suite2p handoffs unchanged; do not touch `DATA\SUPPORT_*`.
