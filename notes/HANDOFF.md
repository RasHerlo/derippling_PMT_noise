# Pickup — 2026-08-30

Continue here. **First implementation: the recursive image-domain check.**
Do **not** start the FOV x-scan until that loop is in and reviewed. Do **not**
re-run a production Haj Grant ChanA clean until the check exists.

Haj Grant ChanB’s existing `defringe_v22` clean is still fine. SUPPORT / suite2p
handoffs unchanged; do not touch `DATA\SUPPORT_*`.

## Where we are

Protocol slot is still `raw → v2.2 pack_D → SUPPORT retrain → suite2p`.
v2.2 worked on the Level3b sandbox; Haj Grant ChanA did not seed. That is a
coverage / seed problem, not a reason to treat one stack’s quiet frames as the
definition of the fringe.

| Channel (Haj Grant Example) | Production `defringe_v22` | Why |
|---|---|---|
| ChanB | **ok**, q≈81 | Branch C live_clean in the catalog; tracking held |
| ChanA | **needs_review** | No catalog hit at this raster with fx support; safe paired seed empty. Do **not** apply the z-ladder trial-clean |

On this Haj Grant stack, frames **756–760** are a mid-experiment shutter
(low-std run; 761 is opening). They are a **same-file quiet window**: no
biology, PMT pattern visible. From those frames we saw ChanA **q≈10** and
ChanB **q≈81** (ChanB matches the successful clean). That is **one
configuration at one moment**, not an inventory of every family later in the
stack. Do not call those frames “the fringe.”

## Catalog (landed 2026-08-30)

One seed file: `fringe_library/catalog.json`. `darkcurrent/` stays lab notes
and is not consulted for seeding.

| `source` | Branch | Default `complete` |
|---|---|---|
| `darkcurrent` | A same day, else B | true |
| `in_stack_shutter` | same A/B split | **false** (set `frames`) |
| `live_clean` | C | true |

Within a branch, complete wins, then newest date. Amplitude is never copied.
Library `q` must still show `fx` support on **this** stack.

Every run’s `overview.pdf` (ok and `needs_review`) prints a catalog line;
`families.json` / `eval.json` store the same block. Rebuild:
`python -m batch_defringe --rebuild-overview "<channel>\defringe_v22"`.

No in-stack shutter rows are in the catalog yet (schema only). Haj Grant
ChanB `q=81` is the only live_clean. Existing DC rows are Shinano 29.595 Hz /
`scanMode=0` and **do not** match Haj Grant.

## Seed (agreed — do not re-litigate)

A seed is a **hint** (look near this `q` / `fx` first), not ground truth.
Locking one pair does **not** guarantee all fringe is caught, and does **not**
guarantee everything inside that pair is non-biology.

What currently limits biology: narrow mask, gate=0 when the ridge is weak,
paired high-z cut, `needs_review` instead of a guess. What is still missing:
an image-domain stop/accept rule on **`removed`**.

Without a seed the code already searches 50-frame blocks for recurrent paired
ridges. That is not “the most likely structure in the FOV”; on a live frame
the strongest FFT peak is often cells.

## Implement tomorrow — 1) recursive check (do this first)

Propose (FFT / catalog / shutter hint) → notch only that candidate → judge
**`removed` in image domain** → look at leftover raw → repeat or stop.

Score **traits**, not a template (no fixed period, no shutter-image match):

- `removed` covers most of the FOV (or the region under test)
- left–right even; biology is local, fringe is a field pattern
- many parallel ridges with locally similar spacing (spacing may change
  across x; it must not look like somata)
- not blob-like

Accept if those hold. Reject if `removed` is patchy, one-sided, or cell-shaped
— even if FFT liked the peak. Then, if leftover raw still shows a grating,
propose another candidate; if the next `removed` fails the image test, **stop**.

Write the verdict on the overview (next to the catalog line). Do not promote a
clean that fails this check.

## Implement later — 2) x-scan, no tiles

One global `q` is why the mid-FOV wide bands come off and the finer edge
lines stay: period changes along the resonant (x) axis; amplitude is also
edge-heavy. After the recursive check works, walk **x** only to **propose**
local period/strength. Apply one global or **smoothly weighted** notch.

**Do not** clean independent tiles (SUPPORT-style seams). Overlapping
windows, never abutting patches. The scan must not become a second spatial
denoiser.

## Cautions

- In-stack shutter = branch A **geometry for families those frames show**,
  flagged incomplete. Rank: dedicated same-day DC (complete A) beats shutter;
  shutter beats live_clean (C). Still scan/track the rest of the stack.
- Do not wire shutter detection into `process_stack` as a hard seed until the
  image-domain check can veto a bad transfer.
- A quiet true DarkCurrent may have no usable ridge; then it must not force a
  seed.
- Every `ok` currently appends branch C. A bad success pollutes the next seed.
  The image-domain check is meant to be the gate before catalog write.
- Inspect-only z-ladder can make a shutter frame look “cleaned” because that
  frame *is* the pattern (Haj Grant 760 at z=4.5–3.5, removed RMS 0). Do not
  promote ladder trials.
- Phase-lock / subtract the shutter *image* is the wrong model. Amplitude
  notch + short q-track stays the filter class.
- Do not flatten Haj Grant 756–760 as the definition of success. Flattening
  those frames is only a probe that a candidate can remove the quiet-window
  pattern; live `removed` must still pass the image-domain traits.

## Haj Grant probe artifacts (not in git)

```
F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\defringe_v22\overview.pdf
F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\defringe_v22\shutter_seed_test\overview.pdf
F:\bPACNewData2026\Haj Grant Example\DATA\ChanB\defringe_v22\overview.pdf
F:\bPACNewData2026\Haj Grant Example\DATA\ChanB\defringe_v22\shutter_seed_test\overview.pdf
```

`python -m batch_defringe.shutter_seed_test` — probe only; does not overwrite
production TIFFs.

## Code landed this session (not yet the recursive loop)

- `batch_defringe/library.py` — `in_stack_shutter`, `complete`, `frames`;
  complete-before-incomplete lookup; `catalog_status` / `format_catalog_line`
- Overview + `families.json` / `eval.json` show catalog use
- Tests in `tests/test_batch_defringe.py`
- Spec: `notes/DARKCURRENT.md` §6.3
