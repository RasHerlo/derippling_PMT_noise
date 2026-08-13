"""Quick look at per-frame FFT peaks to tune adaptive deripple."""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

base = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple"
)
out = Path(r"c:\Users\rasmu\Projects\Repos\derippling_PMT_noise\examples")
out.mkdir(exist_ok=True)

stack = tifffile.imread(base / "Frame1t10.tif").astype(np.float32)
print("stack", stack.shape, stack.dtype)


def fft_mag(frame):
    f = frame - frame.mean()
    F = np.fft.fftshift(np.fft.fft2(f))
    return np.abs(F)


def to_u8_log(mag, p_hi=99.9):
    x = np.log1p(mag)
    hi = np.percentile(x, p_hi)
    x = np.clip(x / max(hi, 1e-8), 0, 1)
    return (x * 255).astype(np.uint8)


for i in [0, 4, 9]:
    mag = fft_mag(stack[i])
    Image.fromarray(to_u8_log(mag)).save(out / f"fft_mag_f{i}.png")
    cy, cx = np.array(mag.shape) // 2
    # zero out a small DC/low-freq disk
    yy, xx = np.ogrid[: mag.shape[0], : mag.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    m = mag.copy()
    m[r < 8] = 0
    # also ignore exact center cross lightly? keep for now
    flat = m.ravel()
    idx = np.argpartition(flat, -20)[-20:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    print(f"\nframe {i} top peaks (rel to center):")
    for k in idx[:15]:
        y, x = np.unravel_index(k, m.shape)
        print(
            f"  dy={y - cy:4d}, dx={x - cx:4d}, r={r[y, x]:5.1f}, "
            f"mag={m[y, x]:.1f}, angle={np.degrees(np.arctan2(y - cy, x - cx)):6.1f}"
        )

# Compare peak locations frame0 vs frame9 via normalized cross-corr of log mag (highpass)
m0 = np.log1p(fft_mag(stack[0]))
m9 = np.log1p(fft_mag(stack[9]))
cy, cx = np.array(m0.shape) // 2
yy, xx = np.ogrid[: m0.shape[0], : m0.shape[1]]
r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
mask = (r >= 8) & (r <= 200)
a = m0[mask]
b = m9[mask]
a = (a - a.mean()) / (a.std() + 1e-8)
b = (b - b.mean()) / (b.std() + 1e-8)
print("\ncorr of log-FFT magnitudes (r=8..200):", float((a * b).mean()))
