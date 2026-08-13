from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

base = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple"
)
out = Path(r"c:\Users\rasmu\Projects\Repos\derippling_PMT_noise\examples")

raw = tifffile.imread(base / "Frame1t10.tif").astype(np.float32)
cln = tifffile.imread(base / "Frame1t10_derippled.tif").astype(np.float32)


def to_u8(img, lo=None, hi=None):
    if lo is None or hi is None:
        lo, hi = np.percentile(img, (1, 99.5))
    if hi <= lo:
        hi = lo + 1
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


for i in [0, 4, 9]:
    lo, hi = np.percentile(raw[i], (1, 99.5))
    Image.fromarray(to_u8(raw[i], lo, hi)).save(out / f"compare_raw_f{i}.png")
    Image.fromarray(to_u8(cln[i], lo, hi)).save(out / f"compare_clean_f{i}.png")
    # also auto-scaled clean
    Image.fromarray(to_u8(cln[i])).save(out / f"compare_clean_autoscale_f{i}.png")

print("raw mean/std", raw.mean(), raw.std())
print("cln mean/std", cln.mean(), cln.std())
print("residual energy ratio", np.mean((raw - cln) ** 2) / np.mean((raw - raw.mean()) ** 2))
for i in [0, 4, 9]:
    print(
        f"f{i} raw std={raw[i].std():.2f} clean std={cln[i].std():.2f} "
        f"corr(raw,clean)={np.corrcoef(raw[i].ravel(), cln[i].ravel())[0,1]:.3f}"
    )
