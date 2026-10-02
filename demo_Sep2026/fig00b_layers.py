"""Figure 0b: biological signal and PMT fringes as two tilted real-space layers.

Lower layer: raw frame 292 (biology).
Upper layer: shutter frame 758, high-contrast grayscale (fringes only).
Right: raw frame 1288, untilted, as the mixed recording.

``python -m demo_Sep2026.fig00b_layers``
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import _percentile_limits, boxed_text, load_frame, new_figure, save_figure
from .paths import CHANA_TIF

BIO_FRAME = 292
SHUTTER_FRAME = 758
MERGE_FRAME = 1288


def _stretch_rgb(img: np.ndarray, *, gamma: float = 0.40, percentiles=(1.0, 99.2)) -> np.ndarray:
    """Percentile + gamma so dim two-photon structure is visible on a slide."""
    x = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(x, percentiles)
    y = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    y = np.power(y, float(gamma))
    return np.stack([y, y, y], axis=-1)


def _shutter_rgb(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(img, dtype=np.float64)
    x = x - float(np.median(x))
    lim = float(np.percentile(np.abs(x), 99.2)) or 1.0
    y = np.clip((x + lim) / (2.0 * lim), 0.0, 1.0)
    return np.stack([y, y, y], axis=-1), -lim, lim


def _warp_card(rgb01: np.ndarray, out_wh: tuple[int, int], shear_px: int) -> np.ndarray:
    import cv2

    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = rgb_u8[:, :, ::-1]
    h, w = bgr.shape[:2]
    out_w, out_h = out_wh
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[float(shear_px), 0.0], [float(out_w - 1), 0.0], [0.0, float(out_h - 1)]])
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(
        bgr,
        M,
        (int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped[:, :, ::-1]


def _paste(canvas: np.ndarray, card: np.ndarray, xy: tuple[int, int]) -> None:
    x0, y0 = xy
    h, w = card.shape[:2]
    y1, x1 = y0 + h, x0 + w
    patch = canvas[y0:y1, x0:x1]
    src = card.astype(np.float32)
    white = np.all(src > 250, axis=2, keepdims=True)
    alpha = np.where(white, 0.0, 1.0)
    patch[...] = np.clip(src * alpha + patch.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def _card_corners(xy, out_wh, shear_px) -> np.ndarray:
    x0, y0 = xy
    out_w, out_h = out_wh
    return np.array(
        [
            [x0 + shear_px, y0],
            [x0 + out_w - 1, y0],
            [x0 + out_w - 1 - shear_px, y0 + out_h - 1],
            [x0, y0 + out_h - 1],
        ],
        dtype=float,
    )


def _draw_outline_on_canvas(canvas: np.ndarray, xy, out_wh, shear_px, color_rgb, thickness: int = 4) -> None:
    """Stroke a parallelogram into the canvas (RGB). Later pastes can cover it."""
    import cv2

    pts = np.round(_card_corners(xy, out_wh, shear_px)).astype(np.int32)
    bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.polylines(canvas, [pts], isClosed=True, color=bgr, thickness=int(thickness), lineType=cv2.LINE_AA)


def draw(
    *,
    bio_frame: int = BIO_FRAME,
    shutter_frame: int = SHUTTER_FRAME,
    merge_frame: int = MERGE_FRAME,
) -> dict:
    bio, n_frames = load_frame(bio_frame, CHANA_TIF)
    shut, _ = load_frame(shutter_frame, CHANA_TIF)
    merge, _ = load_frame(merge_frame, CHANA_TIF)
    if bio.shape != shut.shape or merge.shape != bio.shape:
        raise ValueError("layer shapes differ")
    h, w = bio.shape

    bio_rgb = _stretch_rgb(bio)
    shut_rgb, shut_lo, shut_hi = _shutter_rgb(shut)

    card_w, card_h = 860, 390
    shear = 185
    bio_card = _warp_card(bio_rgb, (card_w, card_h), shear)
    shut_card = _warp_card(shut_rgb, (card_w, card_h), shear)

    canvas_h, canvas_w = 860, 1280
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    bio_xy = (30, 300)
    shut_xy = (250, 20)
    green_rgb = (31, 115, 77)
    _paste(canvas, bio_card, bio_xy)
    _draw_outline_on_canvas(canvas, bio_xy, (card_w, card_h), shear, green_rgb, thickness=5)
    _paste(canvas, shut_card, shut_xy)
    _draw_outline_on_canvas(canvas, shut_xy, (card_w, card_h), shear, (50, 50, 50), thickness=3)

    green = (0.12, 0.45, 0.30)
    fig = new_figure()
    fig.suptitle("Figure 0b  ·  Two layers in the same pixels  ·  ChanA", fontsize=16, fontweight="bold", y=0.97)

    ax = fig.add_axes([0.03, 0.20, 0.58, 0.66])
    ax.imshow(canvas, interpolation="nearest")
    ax.set_axis_off()
    boxed_text(fig, [0.03, 0.155, 0.55, 0.04], f"Biological signal  (frame {bio_frame})", fontsize=10, color=green)
    boxed_text(
        fig,
        [0.03, 0.875, 0.58, 0.04],
        f"PMT fringes  (shutter frame {shutter_frame}; no sample fluorescence)",
        fontsize=10,
        color="0.25",
    )

    axm = fig.add_axes([0.66, 0.24, 0.28, 0.52])
    mlo, mhi = _percentile_limits(merge)
    axm.imshow(merge, cmap="gray", vmin=mlo, vmax=mhi, interpolation="nearest")
    axm.set_title(f"Recorded frame {merge_frame}\n(both layers, mixed)", fontsize=11)
    axm.set_xlabel("x (pixels)")
    axm.set_ylabel("y (pixels)")

    boxed_text(
        fig,
        [0.03, 0.02, 0.94, 0.12],
        (
            f"Lower tilted plane: raw frame {bio_frame}.\n"
            f"Upper tilted plane: raw shutter frame {shutter_frame}, median-centred grayscale, "
            "stretched to the 99.2nd percentile of |pixel|.\n"
            f"Right: raw frame {merge_frame}, untilted. Both contributions occupy the same "
            "(x, y) pixels; the recording mixes them."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "bio_frame": bio_frame,
        "shutter_frame": shutter_frame,
        "merge_frame": merge_frame,
        "source": str(CHANA_TIF),
        "n_frames": n_frames,
        "shape_hw": [h, w],
        "bio_display": "percentile 1–99.2, then gamma 0.40",
        "shutter_display_limits": [shut_lo, shut_hi],
        "shutter_contrast": "median-centred grayscale, ±99.2 percentile of |pixel|",
        "merge_display_limits": [mlo, mhi],
        "cleaned": False,
    }
    try:
        written = save_figure(fig, "fig00b_layers", numbers)
    except OSError:
        written = save_figure(fig, "fig00b_layers_new", numbers)
    numbers["written"] = [str(p) for p in written]
    return numbers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bio-frame", type=int, default=BIO_FRAME)
    ap.add_argument("--shutter-frame", type=int, default=SHUTTER_FRAME)
    ap.add_argument("--merge-frame", type=int, default=MERGE_FRAME)
    args = ap.parse_args(argv)
    numbers = draw(bio_frame=args.bio_frame, shutter_frame=args.shutter_frame, merge_frame=args.merge_frame)
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
