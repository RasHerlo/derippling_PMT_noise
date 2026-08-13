import tifffile
import numpy as np
from pathlib import Path
from PIL import Image

base = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple"
)
out = Path(r"c:\Users\rasmu\Projects\Repos\derippling_PMT_noise\examples")
out.mkdir(exist_ok=True)


def to_u8(img, p_lo=1, p_hi=99.5):
    lo, hi = np.percentile(img, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1
    x = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


for name in ["Frame1.tif", "Frame1t10.tif"]:
    path = base / name
    with tifffile.TiffFile(path) as tif:
        print("=" * 60)
        print(name)
        print("pages:", len(tif.pages))
        print("series:", len(tif.series))
        for i, s in enumerate(tif.series):
            print(f"  series[{i}] shape={s.shape} dtype={s.dtype} axes={s.axes}")
        arr = tif.asarray()
    print("asarray shape:", arr.shape, "dtype:", arr.dtype)
    print(
        "min/max/mean/std:",
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
        float(arr.std()),
    )
    frames = arr[None, ...] if arr.ndim == 2 else arr
    if frames.ndim > 3:
        frames = frames.reshape(-1, *frames.shape[-2:])
    print("n_frames used:", frames.shape[0], "H,W:", frames.shape[-2:])
    for i in range(min(frames.shape[0], 10)):
        f = frames[i]
        print(
            f"  frame {i}: min={f.min()} max={f.max()} mean={f.mean():.2f} "
            f"std={f.std():.2f} p1={np.percentile(f, 1):.1f} p99={np.percentile(f, 99):.1f}"
        )

f1 = tifffile.imread(base / "Frame1.tif")
if f1.ndim == 3:
    f1 = f1[0]
Image.fromarray(to_u8(f1)).save(out / "preview_frame1.png")

stack = tifffile.imread(base / "Frame1t10.tif")
if stack.ndim == 2:
    stack = stack[None]
print("stack shape final:", stack.shape)

mean = stack.mean(axis=0)
std = stack.std(axis=0)
Image.fromarray(to_u8(mean)).save(out / "preview_mean_1t10.png")
Image.fromarray(to_u8(std, 1, 99)).save(out / "preview_std_1t10.png")
Image.fromarray(to_u8(stack[0])).save(out / "preview_stack_f0.png")
if stack.shape[0] > 1:
    Image.fromarray(to_u8(stack[-1])).save(out / "preview_stack_f9.png")
    Image.fromarray(
        to_u8(
            np.abs(stack[0].astype(np.float32) - stack[-1].astype(np.float32)),
            1,
            99,
        )
    ).save(out / "preview_absdiff_f0_f9.png")

print("\nProjection diagnostics on frame0:")
f = stack[0].astype(np.float32)
row_mean = f.mean(axis=1)
col_mean = f.mean(axis=0)
print("row_mean std:", row_mean.std(), "col_mean std:", col_mean.std())

F = np.fft.rfft2(f - f.mean())
mag = np.abs(F)
mag[0, 0] = 0
iy, ix = np.unravel_index(np.argmax(mag), mag.shape)
print("strongest FFT peak at (fy,fx)=", iy, ix, "mag=", mag[iy, ix])
flat = mag.ravel()
idx = np.argpartition(flat, -10)[-10:]
idx = idx[np.argsort(flat[idx])[::-1]]
print("top FFT peaks:")
for k in idx:
    y, x = np.unravel_index(k, mag.shape)
    print(f"  fy={y}, fx={x}, mag={mag[y, x]:.1f}")

# Temporal correlation of demeaned frames (noise stability)
X = stack.astype(np.float32)
X = X - X.mean(axis=(1, 2), keepdims=True)
norms = np.linalg.norm(X.reshape(X.shape[0], -1), axis=1) + 1e-8
corr = (X[0].ravel() @ X.reshape(X.shape[0], -1).T) / (norms[0] * norms)
print("corr of frame0 vs each frame:", np.round(corr, 3))

print("\nSaved previews to", out)
