"""Compare linescan congruence, original fy seeding, and a combination.

Score-and-print only. Does not overwrite production TIFFs.

``python -m batch_defringe.seed_compare`` defaults to 10 Haj Grant ChanA
frames: 160, 1061, 700, two shutter frames (756, 760), and five RNG-picked
others.

Methods
  linescan  — congruent (qy, qx) from H/V/diagonals, then a pack_D notch on
              that axis (fx-column or fy-row) with TRACK_SEARCH.
  original  — fy-row ``detect_families`` on this frame (leftover z cuts), then
              a fy notch. This is the production seeder; it cannot propose fx.
  combo     — union of original fy families and the congruent axis+q. Notch
              each; keep the image-test winner (PASS, else gated, else none).
              Does **not** blend fy/fx into one gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from .congruence import congruent_seed
from .image_check import DETECT_LEFTOVER
from .readout import _jsonable, _percentile_limits, _signed_limit
from .seed import TRACK_SEARCH, hydrate_families
from .shutter_seed_test import _is_nyquist_self_pair
from .spatial_seed import (
    OUTPUT_SUBDIR,
    fft_axis_scores,
    fft_log_amp,
    fx_family_from_q,
    fy_family_from_q,
    notch_fx,
    notch_fy,
)

from pmt_fringe_raw_adaptive import detect_families  # noqa: E402

RNG_SEED = 20260901
ANCHOR_LIVE = (160, 1061, 700)
SHUTTER_PAIR = (756, 760)
N_RANDOM = 5
DEFAULT_TIF = Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\ChanA_stk.tif")


def pick_frames(n_pages: int, *, rng_seed: int = RNG_SEED) -> list[tuple[int, str]]:
    """Required anchors + two shutter frames + five reproducible random others."""
    reserved = set(ANCHOR_LIVE) | set(SHUTTER_PAIR)
    missing = [i for i in list(ANCHOR_LIVE) + list(SHUTTER_PAIR) if i < 0 or i >= n_pages]
    if missing:
        raise ValueError(f"required frames {missing} out of range 0..{n_pages - 1}")
    pool = [i for i in range(n_pages) if i not in reserved]
    if len(pool) < N_RANDOM:
        raise ValueError(f"need {N_RANDOM} random frames, only {len(pool)} free")
    rng = np.random.default_rng(int(rng_seed))
    random = sorted(int(x) for x in rng.choice(pool, size=N_RANDOM, replace=False))
    out: list[tuple[int, str]] = []
    for i in ANCHOR_LIVE:
        out.append((int(i), "live_anchor"))
    for i in SHUTTER_PAIR:
        out.append((int(i), "shutter"))
    for i in random:
        out.append((int(i), "random"))
    return out


def _search_for(role: str) -> int:
    return 2 if role == "shutter" else int(TRACK_SEARCH)


def _seed_light(seed: dict) -> dict:
    return {
        "ok": bool(seed.get("ok")),
        "winner": seed.get("winner"),
        "qy": seed.get("qy"),
        "qx": seed.get("qx"),
        "score": seed.get("score"),
        "measured": seed.get("measured"),
    }


def _trial_light(tr: dict | None) -> dict:
    if tr is None:
        return {
            "axis": None,
            "q_pred": None,
            "q": None,
            "strength": None,
            "gate": 0.0,
            "removed_rms": 0.0,
            "passed": False,
            "status": "none",
            "traits": None,
        }
    score = tr.get("score")
    gate = float(tr.get("gate") or 0.0)
    passed = bool(tr.get("passed"))
    if passed:
        status = "PASS"
    elif gate > 0:
        status = "FAIL"
    else:
        status = "off"
    traits = None
    if score is not None:
        traits = [
            {"name": t["name"], "passed": bool(t["passed"]), "detail": t.get("detail")}
            for t in score.get("traits", [])
        ]
    return {
        "axis": tr.get("axis"),
        "q_pred": tr.get("q_pred"),
        "q": tr.get("q"),
        "strength": tr.get("strength"),
        "gate": gate,
        "removed_rms": float(tr.get("removed_rms") or 0.0),
        "passed": passed,
        "status": status,
        "traits": traits,
    }


def _rank_trial(tr: dict) -> tuple:
    return (
        1 if tr.get("passed") else 0,
        1 if float(tr.get("gate") or 0.0) > 0 else 0,
        float(tr.get("removed_rms") or 0.0),
    )


def _detect_fy_families(frame: np.ndarray) -> list[dict]:
    logamp = fft_log_amp(frame)
    families, _, _ = detect_families(logamp, **DETECT_LEFTOVER)
    if not families:
        return []
    hydrated = hydrate_families(families, logamp, x_z_thresh=float(DETECT_LEFTOVER["x_z_thresh"]))
    height = int(logamp.shape[0])
    return [f for f in hydrated if not _is_nyquist_self_pair(f, height)]


def eval_linescan(frame: np.ndarray, *, search: int) -> dict:
    seed = congruent_seed(frame)
    logamp = fft_log_amp(frame)
    trials: list[dict] = []
    if seed.get("ok") and seed.get("winner") in ("fx", "tilted") and seed.get("qx") is not None:
        trials.append(notch_fx(frame, fx_family_from_q(logamp, seed["qx"]), seed["qx"], search=search))
    if seed.get("ok") and seed.get("winner") in ("fy", "tilted") and seed.get("qy") is not None:
        trials.append(notch_fy(frame, fy_family_from_q(logamp, seed["qy"]), seed["qy"], search=search))
    trial = max(trials, key=_rank_trial) if trials else None
    return {
        "seed": _seed_light(seed),
        "notch": _trial_light(trial),
        "removed": None if trial is None else trial["removed"],
    }


def eval_original(frame: np.ndarray, *, search: int) -> dict:
    logamp = fft_log_amp(frame)
    axes = fft_axis_scores(logamp)
    fams = _detect_fy_families(frame)
    trial = None
    if fams:
        fam = max(fams, key=lambda f: float(f.get("row_score") or 0.0))
        trial = notch_fy(frame, fam, float(fam["q"]), search=search)
    return {
        "fft": {
            "fy_q": float(axes["row_peak_q"]),
            "fy_score": float(axes["row_peak"]),
            "fx_q": float(axes["col_peak_q"]),
            "fx_score": float(axes["col_peak"]),
        },
        "detect_q": [float(f["q"]) for f in fams],
        "detect_row_score": [float(f.get("row_score") or 0.0) for f in fams],
        "notch": _trial_light(trial),
        "removed": None if trial is None else trial["removed"],
    }


def eval_combo(frame: np.ndarray, *, search: int, linescan: dict, original: dict) -> dict:
    """Union of fy detect families and the congruent axis; keep the best notch."""
    logamp = fft_log_amp(frame)
    tried: list[tuple[str, dict]] = []
    for q in original.get("detect_q") or []:
        tried.append(("original_fy", notch_fy(frame, fy_family_from_q(logamp, q), q, search=search)))
    seed = linescan.get("seed") or {}
    if seed.get("ok"):
        qx = seed.get("qx")
        qy = seed.get("qy")
        if seed.get("winner") in ("fx", "tilted") and qx is not None:
            tried.append(("linescan_fx", notch_fx(frame, fx_family_from_q(logamp, qx), qx, search=search)))
        if seed.get("winner") in ("fy", "tilted") and qy is not None:
            if not any(abs(float(q) - float(qy)) < 3.0 for q in (original.get("detect_q") or [])):
                tried.append(("linescan_fy", notch_fy(frame, fy_family_from_q(logamp, qy), qy, search=search)))
    if not tried:
        return {"source": None, "notch": _trial_light(None), "removed": None, "n_tried": 0}
    source, best = max(tried, key=lambda item: _rank_trial(item[1]))
    return {
        "source": source,
        "notch": _trial_light(best),
        "removed": best["removed"],
        "n_tried": len(tried),
    }


def compare_frame(frame: np.ndarray, *, index: int, role: str) -> dict:
    search = _search_for(role)
    linescan = eval_linescan(frame, search=search)
    original = eval_original(frame, search=search)
    combo = eval_combo(frame, search=search, linescan=linescan, original=original)
    return {
        "frame": int(index),
        "role": role,
        "search": search,
        "raw": np.asarray(frame),
        "linescan": linescan,
        "original": original,
        "combo": combo,
    }


def frame_public(rec: dict) -> dict:
    return {
        "frame": rec["frame"],
        "role": rec["role"],
        "search": rec["search"],
        "linescan": {"seed": rec["linescan"]["seed"], "notch": rec["linescan"]["notch"]},
        "original": {
            "fft": rec["original"]["fft"],
            "detect_q": rec["original"]["detect_q"],
            "detect_row_score": rec["original"]["detect_row_score"],
            "notch": rec["original"]["notch"],
        },
        "combo": {
            "source": rec["combo"]["source"],
            "notch": rec["combo"]["notch"],
            "n_tried": rec["combo"]["n_tried"],
        },
    }


def _status_color(status: str) -> str:
    return {"PASS": "0.15", "FAIL": "C1", "off": "0.45", "none": "0.55"}.get(status, "0.3")


def _fmt(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "-"
    try:
        if not np.isfinite(float(v)):
            return "-"
    except (TypeError, ValueError):
        return "-"
    return f"{float(v):.{nd}f}"


def summarize(rows: list[dict]) -> dict:
    methods = ("original", "linescan", "combo")
    counts = {m: {"PASS": 0, "FAIL": 0, "off": 0, "none": 0} for m in methods}
    for rec in rows:
        pub = rec if "original" in rec and "notch" in rec["original"] else rec
        for m in methods:
            st = pub[m]["notch"]["status"]
            counts[m][st] = counts[m].get(st, 0) + 1
    n = len(rows)
    return {"n": n, "counts": counts}


def write_compare_pdf(path: Path, rows: list[dict], *, source: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pub = [frame_public(r) for r in rows]
    summary = summarize(pub)

    def _notch_line(name: str, rec: dict) -> str:
        n = rec[name]["notch"]
        extra = ""
        if name == "linescan":
            s = rec["linescan"]["seed"]
            extra = f"  seed={s['winner'] or 'none'} qy={_fmt(s['qy'])} qx={_fmt(s['qx'])}"
        elif name == "original":
            det = rec["original"]["detect_q"]
            fx = rec["original"]["fft"]
            extra = (
                f"  detect_q={det or '[]'}  "
                f"FFT fy q={_fmt(fx['fy_q'], 0)}/{_fmt(fx['fy_score'], 2)}  "
                f"fx q={_fmt(fx['fx_q'], 0)}/{_fmt(fx['fx_score'], 2)}"
            )
        elif name == "combo":
            extra = f"  from={rec['combo']['source'] or 'none'}  tried={rec['combo']['n_tried']}"
        return (
            f"{name:<9} {n['status']:<4}  {n.get('axis') or '-':<8}  "
            f"q {_fmt(n['q_pred'], 0)}->{_fmt(n['q'], 0)}  "
            f"gate={_fmt(n['gate'], 2)}  rms={_fmt(n['removed_rms'], 3)}{extra}"
        )

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, "Seed method compare  ·  Haj Grant ChanA", fontsize=13, fontweight="bold", va="top")
        fig.text(
            0.06,
            0.935,
            f"{source}   ·   {summary['n']} frames   ·   RNG {RNG_SEED}   ·   not production",
            fontsize=8,
            va="top",
            color="0.35",
        )
        counts = summary["counts"]
        lines = [
            "Image-test of pack_D `removed` (coverage / even / ridges / blobs).",
            "linescan = congruent (qy,qx) then notch that axis.  original = fy-row detect_families.",
            "combo = union of those proposals; keep the best image-test.  fy and fx stay separate.",
            "",
            f"{'method':<10} {'PASS':>5} {'FAIL':>5} {'off':>5} {'none':>5}",
        ]
        for m in ("original", "linescan", "combo"):
            c = counts[m]
            lines.append(f"{m:<10} {c['PASS']:>5} {c['FAIL']:>5} {c['off']:>5} {c['none']:>5}")
        lines.append("")
        lines.append("frame  role         original  linescan  combo     combo from")
        for rec in pub:
            lines.append(
                f"{rec['frame']:<5}  {rec['role']:<11}  "
                f"{rec['original']['notch']['status']:<8}  "
                f"{rec['linescan']['notch']['status']:<8}  "
                f"{rec['combo']['notch']['status']:<8}  "
                f"{rec['combo']['source'] or '-'}"
            )
        fig.text(0.06, 0.90, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig, dpi=140)

        for rec, full in zip(pub, rows):
            fig.clear()
            fig.patch.set_facecolor("white")
            fig.text(
                0.06,
                0.97,
                f"Frame {rec['frame']}  ·  {rec['role']}  ·  search ±{rec['search']}",
                fontsize=13,
                fontweight="bold",
                va="top",
            )
            body = "\n".join(_notch_line(m, rec) for m in ("original", "linescan", "combo"))
            fig.text(0.06, 0.93, body, fontsize=7.5, va="top", family="monospace")
            raw = np.asarray(full["raw"], dtype=np.float64)
            vmin, vmax = _percentile_limits(raw)
            gs = fig.add_gridspec(2, 2, left=0.05, right=0.98, top=0.72, bottom=0.05, hspace=0.22, wspace=0.12)
            panels = (
                ("raw", raw, False),
                ("original removed", full["original"]["removed"], True),
                ("linescan removed", full["linescan"]["removed"], True),
                ("combo removed", full["combo"]["removed"], True),
            )
            for i, (title, img, signed) in enumerate(panels):
                ax = fig.add_subplot(gs[i // 2, i % 2])
                ax.set_title(title, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                if img is None:
                    ax.set_facecolor("0.92")
                    ax.text(0.5, 0.5, "no notch", ha="center", va="center", transform=ax.transAxes, color="0.4")
                    continue
                arr = np.asarray(img, dtype=np.float64)
                if signed:
                    lim = _signed_limit(arr)
                    ax.imshow(arr, cmap="gray", norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), interpolation="nearest")
                else:
                    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument("--frames", type=str, default=None, help="Comma list, e.g. 160,700. Default: 10-frame set.")
    ap.add_argument("--rng", type=int, default=RNG_SEED)
    args = ap.parse_args(argv)
    tif = args.tif or DEFAULT_TIF
    if not tif.is_file():
        print(f"MISSING {tif}")
        return 1
    with tifffile.TiffFile(tif) as tf:
        n_pages = len(tf.pages)
        if args.frames:
            picked = [(int(x.strip()), "manual") for x in args.frames.split(",") if x.strip()]
        else:
            picked = pick_frames(n_pages, rng_seed=args.rng)
        print(f"compare {tif}  n={n_pages}  frames={[i for i, _ in picked]}", flush=True)
        rows = []
        for idx, role in picked:
            print(f"  frame {idx} ({role}) …", flush=True)
            frame = np.asarray(tf.pages[int(idx)].asarray())
            rec = compare_frame(frame, index=idx, role=role)
            pub = frame_public(rec)
            for name in ("original", "linescan", "combo"):
                n = pub[name]["notch"]
                print(
                    f"    {name:<9} {n['status']:<4}  {n.get('axis') or '-':<8} "
                    f"q {_fmt(n['q_pred'], 0)}->{_fmt(n['q'], 0)}  "
                    f"gate={_fmt(n['gate'], 2)}  rms={_fmt(n['removed_rms'], 3)}",
                    flush=True,
                )
            rows.append(rec)
    out_dir = tif.parent / "defringe_v22" / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = write_compare_pdf(out_dir / "seed_compare_10.pdf", rows, source=str(tif))
    payload = {
        "tif": str(tif),
        "rng_seed": int(args.rng),
        "frames": [frame_public(r) for r in rows],
        "summary": summarize([frame_public(r) for r in rows]),
    }
    json_path = out_dir / "seed_compare_10.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"  wrote {pdf}", flush=True)
    print(f"  wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
