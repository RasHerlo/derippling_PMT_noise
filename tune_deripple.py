"""Try stronger / hybrid dynamic deripple and save previews."""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from deripple import deripple_frame, deripple_stack, detect_interference_peaks, build_notch_mask

base = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple"
)
out = Path(r"c:\Users\rasmu\Projects\Repos\derippling_PMT_noise\examples")

raw = tifffile.imread(base / "Frame1t10.tif").astype(np.float32)


def to_u8(img, p=(1, 99.5)):
    lo, hi = np.percentile(img, p)
    if hi <= lo:
        hi = lo + 1
    x = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def fft_log_u8(frame):
    F = np.fft.fftshift(np.fft.fft2(frame - frame.mean()))
    x = np.log1p(np.abs(F))
    return to_u8(x, (50, 99.95))


# Aggressive per-frame settings
params = dict(r_min=12.0, snr=4.0, max_peaks=120, notch_radius=5.0, soften=0.0)
print("Running aggressive per-frame deripple...")
clean = deripple_stack(raw, **params)

# Hybrid: shared peak support from median spectrum + per-frame refinement
print("Building median-spectrum peak support...")
mags = []
for i in range(raw.shape[0]):
    F = np.fft.fftshift(np.fft.fft2(raw[i] - raw[i].mean()))
    mags.append(np.abs(F))
med_mag = np.median(np.stack(mags, axis=0), axis=0)
shared_peaks = detect_interference_peaks(med_mag, r_min=12.0, snr=4.0, max_peaks=80)
print("shared peaks", len(shared_peaks))

hybrid = np.empty_like(raw)
for i in range(raw.shape[0]):
    frame = raw[i]
    mean = float(frame.mean())
    F = np.fft.fftshift(np.fft.fft2(frame - mean))
    mag = np.abs(F)
    frame_peaks = detect_interference_peaks(mag, r_min=12.0, snr=4.5, max_peaks=80)
    # merge peaks (unique)
    peaks = list({*shared_peaks, *frame_peaks})
    mask = build_notch_mask(frame.shape, peaks, notch_radius=5.0, soften=0.0)
    hybrid[i] = np.fft.ifft2(np.fft.ifftshift(F * mask)).real.astype(np.float32) + mean
    print(f"  hybrid frame {i+1}: {len(peaks)} peaks")

# Extra: also suppress vertical frequency ridges near fx~±14 (known family)
print("Running family-ridge notch (dynamic amplitude, fixed freq bands)...")


def ridge_mask(shape, dx_centers=(13, 14, 16), dy_centers=(61, 195), half_w=3, half_h=4):
    h, w = shape
    cy, cx = h // 2, w // 2
    mask = np.ones(shape, dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    for dy in dy_centers:
        for dx in dx_centers:
            for sy in (+1, -1):
                for sx in (+1, -1):
                    y0 = cy + sy * dy
                    x0 = cx + sx * dx
                    d2 = (yy - y0) ** 2 / (half_h**2) + (xx - x0) ** 2 / (half_w**2)
                    mask = np.minimum(mask, (1.0 - np.exp(-d2)).astype(np.float32))
                    # stronger core
                    core = ((yy - y0) ** 2 <= half_h**2) & ((xx - x0) ** 2 <= half_w**2)
                    mask = np.where(core, 0.0, mask).astype(np.float32)
    mask[cy, cx] = 1.0
    return mask


# Also auto-detect dy ridges: for each frame find top columns in mag
ridge = np.empty_like(raw)
base_mask = ridge_mask(raw.shape[-2:])
for i in range(raw.shape[0]):
    frame = raw[i]
    mean = float(frame.mean())
    F = np.fft.fftshift(np.fft.fft2(frame - mean))
    mag = np.abs(F)
    peaks = detect_interference_peaks(mag, r_min=12.0, snr=4.0, max_peaks=100)
    mask = build_notch_mask(frame.shape, peaks, notch_radius=4.0, soften=0.0)
    mask = np.minimum(mask, base_mask)
    ridge[i] = np.fft.ifft2(np.fft.ifftshift(F * mask)).real.astype(np.float32) + mean

# Save outputs
tifffile.imwrite(base / "Frame1t10_derippled_aggressive.tif", clean)
tifffile.imwrite(base / "Frame1t10_derippled_hybrid.tif", hybrid.astype(np.float32))
tifffile.imwrite(base / "Frame1t10_derippled_ridge.tif", ridge.astype(np.float32))

for name, arr in [("aggressive", clean), ("hybrid", hybrid), ("ridge", ridge)]:
    Image.fromarray(to_u8(arr[0])).save(out / f"clean_{name}_f0.png")
    Image.fromarray(to_u8(arr[9])).save(out / f"clean_{name}_f9.png")
    Image.fromarray(fft_log_u8(raw[0])).save(out / "fft_raw_f0.png")
    Image.fromarray(fft_log_u8(arr[0])).save(out / f"fft_{name}_f0.png")
    print(name, "std", arr.std(), "resid energy", np.mean((raw - arr) ** 2) / np.mean((raw - raw.mean()) ** 2))

Image.fromarray(to_u8(raw[0])).save(out / "raw_f0.png")
print("saved")
