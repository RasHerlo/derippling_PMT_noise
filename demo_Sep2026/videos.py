"""Write 300 fps real-space and Fourier-space movies of the demo ChanA stack.

``python -m demo_Sep2026.videos``
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .common import _percentile_limits, fft_log_amp, iter_frames, load_frame
from .paths import CHANA_TIF, DEMO_DIR, VIDEO_FPS


def _sample_limits(n_frames: int, n_sample: int = 32) -> dict:
    idxs = np.unique(np.linspace(0, n_frames - 1, n_sample, dtype=int))
    raws = []
    ffts = []
    for i in idxs:
        raw, _ = load_frame(int(i))
        raws.append(raw)
        ffts.append(fft_log_amp(raw))
    raw_stack = np.stack(raws, axis=0)
    fft_stack = np.stack(ffts, axis=0)
    rlo, rhi = _percentile_limits(raw_stack)
    flo, fhi = _percentile_limits(fft_stack, (3.0, 99.7))
    return {
        "sample_frames": [int(i) for i in idxs],
        "real_display_limits": [float(rlo), float(rhi)],
        "fft_display_limits": [float(flo), float(fhi)],
    }


def _to_uint8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scale = max(float(hi) - float(lo), 1e-12)
    x = np.clip((np.asarray(img, dtype=np.float64) - float(lo)) / scale, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def _open_writer(path, fps: float, width: int, height: int):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)), True)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    return writer, cv2


def write_videos(*, fps: float = VIDEO_FPS, max_frames: int | None = None) -> dict:
    import cv2

    first, n_frames = load_frame(0)
    h, w = first.shape
    n_write = n_frames if max_frames is None else min(int(max_frames), n_frames)
    limits = _sample_limits(n_write)
    rlo, rhi = limits["real_display_limits"]
    flo, fhi = limits["fft_display_limits"]

    real_path = DEMO_DIR / f"vid00_real_space_{int(fps)}fps.mp4"
    fft_path = DEMO_DIR / f"vid00_fourier_space_{int(fps)}fps.mp4"
    real_w, cv2 = _open_writer(real_path, fps, w, h)
    fft_w, _ = _open_writer(fft_path, fps, w, h)
    try:
        for i, raw, _n in iter_frames():
            if i >= n_write:
                break
            real_u8 = _to_uint8(raw, rlo, rhi)
            fft_u8 = _to_uint8(fft_log_amp(raw), flo, fhi)
            real_w.write(cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR))
            fft_w.write(cv2.cvtColor(fft_u8, cv2.COLOR_GRAY2BGR))
            if i == 0 or (i + 1) % 200 == 0 or i + 1 == n_write:
                print(f"  video frames {i + 1}/{n_write}", flush=True)
    finally:
        real_w.release()
        fft_w.release()

    numbers = {
        "stack": str(CHANA_TIF),
        "n_frames_in_stack": n_frames,
        "n_frames_written": n_write,
        "shape_hw": [h, w],
        "fps": float(fps),
        "duration_s": n_write / float(fps),
        "real_video": str(real_path),
        "fourier_video": str(fft_path),
        "fourier_note": "each frame is log(1+|FFT|) after subtracting that frame's median, centred (same as v4 fft_log_amp)",
        **limits,
    }
    meta = DEMO_DIR / "vid00_stack.json"
    meta.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    numbers["written"] = [str(real_path), str(fft_path), str(meta)]
    return numbers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=float, default=VIDEO_FPS)
    ap.add_argument("--max-frames", type=int, default=None, help="Encode only the first N frames (debug)")
    args = ap.parse_args(argv)
    numbers = write_videos(fps=args.fps, max_frames=args.max_frames)
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
