# Claude share excerpts (inspiration)

Source: https://claude.ai/share/de19610d-d99a-4f20-a65a-cdef8ceb80c2  
Saved from user paste (share page itself not machine-readable).

## Key findings (Claude)

- Pattern is highly periodic; FFT peaks at roughly `dy=±61, dx=±14` and harmonics `±122, ±195`.
- Peak **locations are stable** across frames; visual strength/phase shifts frame-to-frame.
- Approach: auto-detect outlier harmonic spikes vs local neighborhood median; replace those bins with local background; inverse FFT.
- Script name mentioned: `pmt_fringe_denoise.py`
  - CLI: `python pmt_fringe_denoise.py your_stack.tif -o cleaned.tif --diagnostics diag.png`
  - Auto-detects harmonics from the stack; optional `--reference` for pooled detection stats.
  - Knobs: `--sigma-y`, `--height-sigma`, `--dc-col-protect`
  - Importable: `detect_noise_harmonics`, `clean_stack`
- Residual “blocky” artifacts in SUPPORT-preprocessed frames were **not** the fringe itself.

## User pipeline insight (important)

Noisy blocks are artifacts from a **SUPPORT denoising** step applied to make fringes more visible.

Optimal idea:
1. Detect fringe frequencies on the SUPPORT-denoised stack (clearer peaks).
2. Apply that FFT notch / harmonic mask to the **raw** stack.
3. Then denoise the defringed raw (avoid SUPPORT inventing blocks on fringed data).

Conversation cut off when Claude hit free-message limit before finishing the raw-transfer approach.

## Claude files on disk

`F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\from Claude\`

- `pmt_fringe_denoise.py` — reference implementation (kept on F:; logic ported into repo `deripple.py`)
- `Frame1t10_cleaned.tif`
- `Frame1t10_diagnostics.png`

### Method difference vs sparse peak notches

Claude detects **harmonic rows** (`dy`) from a max-residual row profile, then notches
**full-width horizontal bands** in FFT space and replaces magnitude with a local
median background while keeping phase. That covers sidebands across `fx` and
avoids blocky ringing from incomplete point notches.

Repo `deripple.py` now uses this row-band method as the primary cleaner.
