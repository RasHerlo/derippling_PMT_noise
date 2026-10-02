"""v4 per-frame engine: one growing mask, ranked lines, soft α, predicted check.

Default: full stack into ``<channel>/defringe_v4/`` (not v22/v3).
With ``--out``, mirror that folder onto a destination drive and never write the
source tree (``E:\\Rasmus-Guillermo`` is refused as a write target).

``python -m batch_defringe.process_v4 --root <experiment> --out F:\\CollectedData``
``--seed10``: same 10 indices as seed_compare (ChanA convenience set).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import detect_families, fft_log_amp, search_q  # noqa: E402
from pmt_fringe_raw_adaptive_v21 import _attenuate_family_on_amp  # noqa: E402

from .congruence import congruent_seed, seed_peak_mask
from .discover import discover_stacks, job_for_stack
from .experiment_xml import fingerprint_compatible
from .image_check import DETECT_LEFTOVER, score_removed
from .library import catalog_status, lookup_prior
from .priors import load_prior
from .process import PACK_D, ProcessResult
from .readout import _jsonable, write_mean_tif
from .seed import Q_CLUSTER_TOL, TRACK_SEARCH, hydrate_families, library_family_supported
from .v4_export import (  # noqa: E402
    CONSECUTIVE_IO_STOP,
    DRIVE_WAIT_SEC,
    DriveKeepAlive,
    IO_RETRIES,
    IO_RETRY_SLEEP,
    USB_POWER_HINT,
    V4_DIR,
    artifacts_complete,
    assert_writable,
    copy_experiment_xml,
    forbidden_root_for,
    is_drive_io_error,
    pack_frame_export,
    path_reachable,
    save_mask_patterns,
    sparsify,
    v4_dest_dir,
    wait_until_reachable,
    BatchProgress,
)
from .v4_catchup import write_catchup
from .seed_compare import pick_frames
from .shutter_detect import detect_shutter_windows, format_shutter_span, scan_frame_stats, shutter_public
from .shutter_seed_test import _is_nyquist_self_pair, learn_shutter_families
from .spatial_seed import (
    GATE_HIGH,
    GATE_LOW,
    _attenuate_fx_family_on_amp,
    _xvalid,
    _yvalid,
    fft_axis_scores,
    fx_family_from_q,
    search_fx,
    spectral_peak_mask,
)

MAX_LINES = 10
MAX_INSPECT = 8
ALPHA_LIVE = (0.28, 0.55, 0.85)
ALPHA_SHUTTER = (0.28, 0.55, 0.85, 1.00)
AGREE_MIN = 0.40
Y_RADIUS = 2
Q_MIN = 4.0


def v4_out_dir(tif_path: Path) -> Path:
    return Path(tif_path).parent / V4_DIR


def v4_cleaned_path(tif_path: Path) -> Path:
    return v4_out_dir(tif_path) / f"{Path(tif_path).stem}_defringed_v4.tif"


def v4_removed_path(tif_path: Path) -> Path:
    return v4_out_dir(tif_path) / f"{Path(tif_path).stem}_removed_v4.tif"


def _refresh_catchup(dest_root: Path) -> None:
    try:
        write_catchup(
            dest_root=dest_root,
            notes_path=_REPO / "notes" / "V4_DEST_CATCHUP.md",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  catchup list not refreshed: {exc}", flush=True)


def pick_inspect_frames(rows: list[dict], shutter: dict) -> list[tuple[int, str]]:
    """Inspect set from *this* channel's shutter window and metrics.

    ChanA seed_compare indices (160, 1061, …) are not required. The two PMTs
    are not coupled; a shared shutter *time* window is the same experiment,
    not a shared fringe family.
    """
    chosen: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(idx: int | None, why: str) -> None:
        if idx is None:
            return
        i = int(idx)
        if i in seen:
            return
        seen.add(i)
        chosen.append((i, why))

    shut = [int(i) for i in (shutter.get("frames") or [])]
    if shut:
        add(shut[len(shut) // 2], "shutter_mid")
        add(shut[0], "shutter_start")
        add(shut[-1], "shutter_end")

    live = [r for r in rows if str(r.get("role") or "") != "shutter"]
    pool = live or list(rows)
    if pool:
        add(int(max(pool, key=lambda r: float(r.get("removed_rms") or 0))["frame"]), "strong")
        nonempty = [r for r in pool if not r.get("empty")]
        if nonempty:
            add(int(min(nonempty, key=lambda r: float(r.get("removed_rms") or 0))["frame"]), "weak")
        empty = [r for r in pool if r.get("empty")]
        if empty:
            add(int(empty[len(empty) // 2]["frame"]), "empty")
        brake = [r for r in pool if r.get("brake")]
        if brake:
            add(int(max(brake, key=lambda r: float(r.get("removed_rms") or 0))["frame"]), "brake")
        add(int(max(pool, key=lambda r: int(r.get("n_lines") or 0))["frame"]), "most_lines")
    return chosen[:MAX_INSPECT]


@dataclass
class Line:
    axis: str
    q: float
    family: dict
    source: str
    kind: str = "ridge"  # peak = thin conjugate blobs; ridge = pack_D local excess
    peak_mask: np.ndarray | None = None
    tier: str = "dubious"
    note: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.axis, int(round(float(self.q))))


def _near(a: float, b: float | None, tol: float = Q_CLUSTER_TOL) -> bool:
    if b is None:
        return False
    return abs(float(a) - float(b)) < float(tol)


def _line_public(line: Line) -> dict:
    return {
        "axis": line.axis,
        "q": float(line.q),
        "source": line.source,
        "kind": line.kind,
        "tier": line.tier,
        "note": line.note,
    }


def _catalog_line(src: dict, logamp: np.ndarray) -> Line | None:
    if not library_family_supported(src, logamp):
        return None
    hydrated = hydrate_families([src], logamp)
    if not hydrated:
        return None
    fam = dict(hydrated[0])
    return Line("fy", float(fam["q"]), fam, "catalog", kind="ridge", note="library A/B/C")


def _seg_qs(seed: dict, axis: str) -> list[tuple[float, str]]:
    """Segment and L–C–R sliding qs from the same rloess traces as congruence."""
    tr = ((seed.get("cuts") or {}).get("traces") or {})
    pack = (tr.get("horizontal") if axis == "fx" else tr.get("vertical")) or {}
    out: list[tuple[float, str]] = []
    for s in pack.get("segs") or []:
        q = s.get("q")
        if q is None:
            continue
        qf = float(q)
        if not np.isfinite(qf) or abs(qf) < Q_MIN:
            continue
        out.append((qf, f"seg {s.get('i')}"))
    qs = pack.get("qs")
    if qs is None:
        return out
    arr = np.asarray(qs, dtype=float)
    n = arr.size
    if n < 9:
        return out
    thirds = (
        ("L", arr[: n // 3]),
        ("C", arr[n // 3 : 2 * n // 3]),
        ("R", arr[2 * n // 3 :]),
    )
    for name, sl in thirds:
        good = sl[np.isfinite(sl)]
        if good.size:
            out.append((float(np.median(good)), f"slide {name}"))
    return out


def collect_fft_leftover(frame: np.ndarray) -> list[Line]:
    """FFT fy ridges + fx column peak on leftover. Not linescan, not priors."""
    logamp = fft_log_amp(frame)
    h = int(logamp.shape[0])
    lines: list[Line] = []
    raw_fy, _, _ = detect_families(logamp, **DETECT_LEFTOVER)
    if raw_fy:
        for fam in hydrate_families(raw_fy, logamp, x_z_thresh=float(DETECT_LEFTOVER["x_z_thresh"])):
            if _is_nyquist_self_pair(fam, h):
                continue
            lines.append(Line("fy", float(fam["q"]), dict(fam), "fft", kind="ridge", note="leftover fy"))
    scores = fft_axis_scores(logamp)
    col_q = float(scores["col_peak_q"])
    if float(scores["col_peak"]) >= GATE_LOW:
        fam = fx_family_from_q(logamp, col_q)
        lines.append(
            Line("fx", col_q, fam, "fft", kind="ridge", note=f"leftover fx {scores['col_peak']:.2f}")
        )
    return unique_lines(lines)


def collect_lines(
    frame: np.ndarray,
    *,
    catalog_families: list[dict] | None = None,
    shutter_families: list[dict] | None = None,
) -> tuple[list[Line], dict]:
    """Add order: catalog → shutter-learn → linescan center → edge/segment qs.

    Congruence none skips linescan pieces; leftover FFT is a later grow pass.
    """
    seed = congruent_seed(frame)
    logamp = fft_log_amp(frame)
    h, w = int(logamp.shape[0]), int(logamp.shape[1])
    lines: list[Line] = []

    for src in catalog_families or []:
        ln = _catalog_line(src, logamp)
        if ln is not None:
            lines.append(ln)

    for fam in shutter_families or []:
        f = dict(fam)
        lines.append(
            Line("fy", float(f["q"]), f, "shutter_hint", kind="ridge", note="in-stack shutter learn")
        )

    if seed.get("ok"):
        win = seed.get("winner")
        if win in ("fx", "fy", "tilted"):
            mask = seed_peak_mask(h, w, seed)
            if float(np.max(mask)) > 0:
                if win == "fy":
                    axis, q = "fy", float(seed["qy"])
                else:
                    axis, q = "fx", float(seed["qx"])
                lines.append(
                    Line(
                        axis,
                        q,
                        {},
                        "linescan",
                        kind="peak",
                        peak_mask=mask,
                        note=f"congruence {win} center",
                    )
                )
            axes_for_edges: list[str] = []
            if win in ("fx", "tilted"):
                axes_for_edges.append("fx")
            if win in ("fy", "tilted"):
                axes_for_edges.append("fy")
            q_center = {"fx": seed.get("qx"), "fy": seed.get("qy")}
            for axis in axes_for_edges:
                q0 = q_center.get(axis)
                for q, tag in _seg_qs(seed, axis):
                    if q0 is not None and _near(q, q0):
                        continue
                    if axis == "fx":
                        emask = spectral_peak_mask(h, w, qx=q)
                    else:
                        emask = spectral_peak_mask(h, w, qy=q)
                    if float(np.max(emask)) <= 0:
                        continue
                    lines.append(
                        Line(
                            axis,
                            float(q),
                            {},
                            "linescan",
                            kind="peak",
                            peak_mask=emask,
                            note=f"linescan {tag}",
                        )
                    )

    return unique_lines(lines), seed


def unique_lines(lines: list[Line]) -> list[Line]:
    """First source wins at (axis, q). Collect order is the priority."""
    out: list[Line] = []
    seen: list[tuple[str, int]] = []
    for ln in lines:
        k = ln.key
        if any(a == k[0] and abs(q - k[1]) < Q_CLUSTER_TOL for a, q in seen):
            continue
        out.append(ln)
        seen.append(k)
    return out


def rank_lines(lines: list[Line], seed: dict) -> list[Line]:
    """Keep collect order. Tag tiers for the PDF; do not re-sort by FFT agreement."""
    _ = seed
    for ln in lines:
        note = (ln.note or "").lower()
        if ln.source == "catalog":
            ln.tier = "catalog"
        elif ln.source == "shutter_hint":
            ln.tier = "shutter"
        elif ln.source == "linescan" and ("seg" in note or "slide" in note):
            ln.tier = "extra"
        elif ln.source == "linescan":
            ln.tier = "core"
        else:
            ln.tier = "dubious"
    return lines[:MAX_LINES]


def _identity(frame: np.ndarray) -> dict:
    x = np.asarray(frame, dtype=np.float32)
    z = np.zeros_like(x)
    return {
        "cleaned": np.asarray(frame),
        "removed": z.copy(),
        "predicted": z.copy(),
        "applied": np.zeros(x.shape, dtype=np.float32),
        "removed_rms": 0.0,
        "n_active": 0,
        "max_alpha": 0.0,
        "agree": 1.0,
    }


def apply_lines(
    frame: np.ndarray,
    lines: list[Line],
    *,
    max_alpha: float,
    search: int,
) -> dict:
    """One FFT; peak pieces thin-blob, ridge pieces pack_D excess; take max attenuation."""
    if not lines or max_alpha <= 0:
        return _identity(frame)
    params = dict(PACK_D)
    params.update(y_sigma=1.0, y_radius=Y_RADIUS)
    orig_dtype = frame.dtype
    x = np.asarray(frame, dtype=np.float32)
    offset = float(np.median(x))
    x0 = x - offset
    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j * np.angle(F))
    logamp = np.log1p(amp)
    h, w = x.shape
    att_peak = np.zeros_like(amp, dtype=np.float64)
    newamp_ridge = amp.copy()
    n_active = 0
    xvalid = _xvalid(w)
    yvalid = _yvalid(h)
    for ln in lines:
        if ln.kind == "peak":
            wmask = ln.peak_mask
            if wmask is None:
                continue
            wmask = np.asarray(wmask, dtype=np.float64)
            if wmask.shape != amp.shape or float(np.max(wmask)) <= 0:
                continue
            n_active += 1
            att_peak = np.maximum(att_peak, wmask)
            continue
        if ln.axis == "fy":
            q, strength = search_q(logamp, float(ln.q), True, xvalid, int(search))
        else:
            q, strength = search_fx(logamp, float(ln.q), True, yvalid, int(search))
        gate = float(np.clip((strength - GATE_LOW) / max(1e-9, GATE_HIGH - GATE_LOW), 0.0, 1.0))
        if gate <= 0:
            continue
        n_active += 1
        if ln.axis == "fy":
            _attenuate_family_on_amp(
                amp,
                newamp_ridge,
                ln.family,
                q,
                gate,
                max_alpha=float(max_alpha),
                ratio_start=params["ratio_start"],
                ratio_full=params["ratio_full"],
                y_sigma=params["y_sigma"],
                y_radius=int(params["y_radius"]),
            )
        else:
            _attenuate_fx_family_on_amp(
                amp,
                newamp_ridge,
                ln.family,
                q,
                gate,
                max_alpha=float(max_alpha),
                ratio_start=params["ratio_start"],
                ratio_full=params["ratio_full"],
            )
    att_ridge = np.clip((amp - newamp_ridge) / (amp + 1e-12), 0.0, 1.0)
    att = np.maximum(att_ridge, float(max_alpha) * att_peak)
    newamp = amp * (1.0 - att)
    applied = np.clip(att, 0.0, 1.0).astype(np.float32)
    predicted = np.real(np.fft.ifft2(np.fft.ifftshift((amp - newamp) * phase))).astype(np.float32)
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(newamp * phase))) + offset
    removed = x - cleaned
    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_w = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_w = cleaned.astype(orig_dtype)
    rem = np.asarray(removed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    return {
        "cleaned": cleaned_w,
        "removed": rem.astype(np.float32),
        "predicted": predicted,
        "applied": applied,
        "removed_rms": float(np.sqrt(np.mean(rem * rem))),
        "n_active": n_active,
        "max_alpha": float(max_alpha),
        "agree": _agree_pred_rem(pred, rem),
    }


def _agree_pred_rem(pred: np.ndarray, rem: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(rem, dtype=np.float64)
    pr = float(np.sqrt(np.mean(p * p)))
    rr = float(np.sqrt(np.mean(r * r)))
    d = float(np.sqrt(np.mean((r - p) ** 2)))
    if max(pr, rr) < 1e-9:
        return 1.0
    return float(max(0.0, 1.0 - d / (pr + rr + 1e-12)))


def _ridges_ok(score: dict) -> bool:
    return bool((score.get("by_name") or {}).get("ridges", {}).get("passed", False))


def accept_step(trial: dict, prev: dict, *, shutter: bool) -> tuple[bool, str]:
    rem = np.asarray(trial["removed"], dtype=np.float64)
    inc = rem - np.asarray(prev["removed"], dtype=np.float64)
    inc_rms = float(np.sqrt(np.mean(inc * inc)))
    if trial["n_active"] <= 0 or trial["removed_rms"] < 1e-8:
        return False, "no gate"
    score = score_removed(trial["removed"])
    by = score.get("by_name") or {}
    cov = bool(by.get("coverage", {}).get("passed", False))
    even = bool(by.get("even", {}).get("passed", False))
    blob = bool(by.get("blob", {}).get("passed", True))
    ridges = _ridges_ok(score)
    inc_score = score_removed(inc.astype(np.float32)) if inc_rms > 1e-6 else {"passed": True, "by_name": {}}
    inc_ridges = _ridges_ok(inc_score)
    inc_blob = bool((inc_score.get("by_name") or {}).get("blob", {}).get("passed", True))
    if shutter:
        return True, "shutter push"
    # Predicted ≈ removed is the v4 check: ridge traits are fy-centric and can
    # fail on a clean vertical (fx) grating that is still a fringe.
    if trial["agree"] >= AGREE_MIN and cov and even and (blob or ridges):
        return True, "ok"
    if not score["passed"] and not ridges:
        return False, "removed failed traits (not ridges)"
    if inc_rms > 1e-6 and not inc_score["passed"] and not inc_ridges:
        return False, "increment failed traits"
    if trial["agree"] < AGREE_MIN and trial["removed_rms"] > 0.5 and not inc_blob:
        return False, "removed diverged from predicted"
    return True, "ok"


def _snap(trial: dict, *, kept: bool, why: str, reason: str, alpha: float, lines: list[Line], keep_images: bool) -> dict:
    rec = {
        "kept": kept,
        "why": why,
        "reason": reason,
        "alpha": float(alpha),
        "n_lines": len(lines),
        "lines": list(lines),
        "removed_rms": trial["removed_rms"],
        "agree": trial["agree"],
        "n_active": trial["n_active"],
        "applied_sparse": sparsify(trial["applied"]),
    }
    if keep_images:
        rec["applied"] = np.asarray(trial["applied"])
        rec["predicted"] = np.asarray(trial["predicted"])
        rec["removed"] = np.asarray(trial["removed"])
        rec["cleaned"] = np.asarray(trial["cleaned"])
    return rec


def grow_frame(
    frame: np.ndarray,
    *,
    role: str,
    catalog_families: list[dict] | None = None,
    shutter_families: list[dict] | None = None,
    keep_images: bool = True,
) -> dict:
    """Full v4 loop on one frame."""
    shutter = role == "shutter"
    alphas = list(ALPHA_SHUTTER if shutter else ALPHA_LIVE)
    search = 2 if shutter else TRACK_SEARCH
    collected, seed = collect_lines(
        frame,
        catalog_families=catalog_families,
        shutter_families=shutter_families,
    )
    ranked = rank_lines(collected, seed)
    current = _identity(frame)
    accepted: list[Line] = []
    kept_steps: list[dict] = []
    undone: list[dict] = []
    history: list[dict] = []
    brake = False

    def consider(new_lines: list[Line], alpha: float, why: str) -> bool:
        nonlocal current, brake
        trial = apply_lines(frame, new_lines, max_alpha=alpha, search=search)
        ok, reason = accept_step(trial, current, shutter=shutter)
        rec = _snap(
            trial,
            kept=ok,
            why=why,
            reason=reason,
            alpha=alpha,
            lines=new_lines,
            keep_images=keep_images,
        )
        history.append(rec)
        if ok:
            current = trial
            kept_steps.append(rec)
            return True
        undone.append(rec)
        if not shutter and reason != "no gate":
            brake = True
        return False

    for ln in ranked:
        if consider(accepted + [ln], alphas[0], f"add {ln.tier} {ln.source}/{ln.axis} q={ln.q:.1f}"):
            accepted.append(ln)
        if len(accepted) >= MAX_LINES:
            break

    if accepted:
        leftover_img = np.asarray(current["cleaned"])
        more = collect_fft_leftover(leftover_img)
        used = [ln.key for ln in accepted]
        for ln in more:
            if any(a == ln.key[0] and abs(q - ln.key[1]) < Q_CLUSTER_TOL for a, q in used):
                continue
            ln.tier = "dubious"
            ln.note = (ln.note + " leftover").strip()
            if consider(accepted + [ln], alphas[0], f"leftover {ln.source}/{ln.axis} q={ln.q:.1f}"):
                accepted.append(ln)
                used.append(ln.key)
            if len(accepted) >= MAX_LINES:
                break

    if accepted:
        for a in alphas[1:]:
            if not consider(accepted, a, f"raise a={a:.2f}"):
                break

    return {
        "role": role,
        "search": search,
        "seed": {
            "ok": bool(seed.get("ok")),
            "winner": seed.get("winner"),
            "qy": seed.get("qy"),
            "qx": seed.get("qx"),
            "score": seed.get("score"),
        },
        "seed_full": seed if keep_images else None,
        "ranked": [_line_public(ln) for ln in ranked],
        "accepted": [_line_public(ln) for ln in accepted],
        "accepted_lines": accepted,
        "n_lines": len(accepted),
        "max_alpha": current["max_alpha"],
        "removed_rms": current["removed_rms"],
        "agree": current["agree"],
        "brake": brake,
        "empty": len(accepted) == 0,
        "core_only": len(accepted) == 1 and accepted[0].tier == "core",
        "cleaned": current["cleaned"],
        "removed": current["removed"],
        "predicted": current["predicted"],
        "applied": current["applied"],
        "steps": kept_steps,
        "undone": undone,
        "history": history,
        "raw": np.asarray(frame) if keep_images else None,
    }


def _resolve_catalog(
    *,
    computer: str,
    channel: str,
    fingerprint: dict | None,
    recording_date: str | None,
    batch_root: Path | None,
) -> tuple[list[dict], dict | None]:
    fp = fingerprint or {}
    lib_hit = lookup_prior(
        computer=computer,
        channel=channel,
        fingerprint=fp,
        recording_date=recording_date,
        batch_root=batch_root,
    )
    catalog_families = list((lib_hit or {}).get("families") or [])
    if not catalog_families and batch_root is not None:
        prior = load_prior(batch_root, computer, channel)
        if (
            prior
            and prior.get("families")
            and fingerprint_compatible(prior.get("fingerprint"), fp)
        ):
            catalog_families = list(prior["families"])
    return catalog_families, lib_hit


def process_seed10(
    tif_path: Path,
    *,
    channel: str,
    computer: str,
    fingerprint: dict | None = None,
    recording_date: str | None = None,
    batch_root: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    from .v4_report import write_v4_seed10_pdf

    tif_path = Path(tif_path)
    out_dir = Path(out_dir) if out_dir is not None else tif_path.parent / V4_DIR / "seed10"
    assert_writable(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_families, lib_hit = _resolve_catalog(
        computer=computer,
        channel=channel,
        fingerprint=fingerprint,
        recording_date=recording_date,
        batch_root=batch_root,
    )
    cat_qs = [float(f["q"]) for f in catalog_families if f.get("q") is not None]
    print(f"v4 seed10  {tif_path}  {computer}/{channel}", flush=True)
    print(f"  catalog qs={cat_qs or 'none'}  branch={(lib_hit or {}).get('branch')}", flush=True)
    with tifffile.TiffFile(tif_path) as tf:
        n = int(tf.series[0].shape[0])
        framespec = pick_frames(n)
        shutter_det = detect_shutter_windows(scan_frame_stats(tf))
        shutter_pub = shutter_public(shutter_det)
        shutter_idx = [int(i) for i in (shutter_pub.get("frames") or [])]
        shutter_frames = [np.asarray(tf.pages[i].asarray()) for i in shutter_idx]
        shutter_families: list[dict] = []
        if shutter_frames:
            shutter_families, _ = learn_shutter_families(shutter_frames)
        shut_qs = [float(f["q"]) for f in shutter_families]
        print(f"  shutter: {format_shutter_span(shutter_det)}  learn qs={shut_qs or 'none'}", flush=True)
        results = []
        acc_raw = acc_cl = acc_rm = acc_pr = None
        for idx, role in framespec:
            frame = np.asarray(tf.pages[int(idx)].asarray())
            print(f"  frame {idx} ({role}) …", flush=True)
            rec = grow_frame(
                frame,
                role=role,
                catalog_families=catalog_families,
                shutter_families=shutter_families,
            )
            rec["frame"] = int(idx)
            results.append(rec)
            rawf = np.asarray(frame, dtype=np.float64)
            if acc_raw is None:
                acc_raw = np.zeros_like(rawf)
                acc_cl = np.zeros_like(rawf)
                acc_rm = np.zeros_like(rawf)
                acc_pr = np.zeros_like(rawf)
            acc_raw += rawf
            acc_cl += np.asarray(rec["cleaned"], dtype=np.float64)
            acc_rm += np.asarray(rec["removed"], dtype=np.float64)
            acc_pr += np.asarray(rec["predicted"], dtype=np.float64)
            print(
                f"    lines={rec['n_lines']} a={rec['max_alpha']:.2f} "
                f"rms={rec['removed_rms']:.3g} agree={rec['agree']:.2f} "
                f"brake={rec['brake']} empty={rec['empty']}",
                flush=True,
            )
    inv = 1.0 / max(len(results), 1)
    summary = {
        "n_frames": len(results),
        "n_empty": int(sum(r["empty"] for r in results)),
        "n_core_only": int(sum(r["core_only"] for r in results)),
        "n_brake": int(sum(r["brake"] for r in results)),
        "median_removed_rms": float(np.median([r["removed_rms"] for r in results])),
        "median_agree": float(np.median([r["agree"] for r in results])),
        "catalog_qs": cat_qs,
        "shutter_learn_qs": shut_qs,
        "catalog_branch": (lib_hit or {}).get("branch"),
    }
    payload = {
        "version": "v4_seed10",
        "source_tif": str(tif_path),
        "channel": channel,
        "computer": computer,
        "shutter": shutter_pub,
        "frames": [
            {
                "frame": r["frame"],
                "role": r["role"],
                "ranked": r["ranked"],
                "accepted": r["accepted"],
                "n_lines": r["n_lines"],
                "max_alpha": r["max_alpha"],
                "removed_rms": r["removed_rms"],
                "agree": r["agree"],
                "brake": r["brake"],
                "empty": r["empty"],
                "seed": r["seed"],
                "n_steps_kept": len(r["steps"]),
                "n_undone": len(r["undone"]),
            }
            for r in results
        ],
        "summary": summary,
    }
    (out_dir / "seed10.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    pdf = write_v4_seed10_pdf(
        out_dir / "overview.pdf",
        tif_path=tif_path,
        channel=channel,
        computer=computer,
        shutter=shutter_pub,
        results=results,
        mean_raw=acc_raw * inv,
        mean_cleaned=acc_cl * inv,
        mean_removed=acc_rm * inv,
        mean_predicted=acc_pr * inv,
        summary=summary,
    )
    print(f"  overview: {pdf}", flush=True)
    return pdf


def _frame_row(fi: int, role: str, rec: dict) -> dict:
    seed = rec.get("seed") or {}
    acc = rec.get("accepted") or []
    rem = rec.get("removed")
    mean_abs = 0.0
    p99_abs = 0.0
    if rem is not None:
        arr = np.abs(np.asarray(rem, dtype=np.float64))
        if arr.size:
            mean_abs = float(np.mean(arr))
            p99_abs = float(np.percentile(arr, 99.0))
    hist = rec.get("history") or []
    return {
        "frame": int(fi),
        "role": role,
        "n_lines": int(rec.get("n_lines") or 0),
        "max_alpha": float(rec.get("max_alpha") or 0),
        "removed_rms": float(rec.get("removed_rms") or 0),
        "removed_mean_abs": mean_abs,
        "removed_p99_abs": p99_abs,
        "agree": float(rec.get("agree") or 0),
        "brake": bool(rec.get("brake")),
        "empty": bool(rec.get("empty")),
        "core_only": bool(rec.get("core_only")),
        "seed_winner": seed.get("winner"),
        "qx": seed.get("qx"),
        "qy": seed.get("qy"),
        "accepted": ";".join(
            f"{a.get('source')}/{a.get('kind')}/{a.get('axis')} q={float(a.get('q') or 0):.1f}"
            for a in acc
        ),
        "n_history": len(hist),
        "n_undone": int(sum(1 for s in hist if not s.get("kept"))),
    }


CSV_FIELDS = [
    "frame",
    "role",
    "n_lines",
    "max_alpha",
    "removed_rms",
    "removed_mean_abs",
    "removed_p99_abs",
    "agree",
    "brake",
    "empty",
    "core_only",
    "seed_winner",
    "qx",
    "qy",
    "accepted",
    "n_history",
    "n_undone",
]


def resolve_v4_out_dir(
    tif_path: Path,
    *,
    source_root: Path | None = None,
    dest_root: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    if out_dir is not None:
        return assert_writable(Path(out_dir))
    if dest_root is not None:
        if source_root is None:
            raise ValueError("source_root is required when dest_root is set")
        return assert_writable(v4_dest_dir(tif_path, source_root, dest_root))
    return assert_writable(Path(tif_path).parent / V4_DIR)


def process_stack_v4(
    tif_path: Path,
    *,
    channel: str,
    computer: str,
    fingerprint: dict | None = None,
    recording_date: str | None = None,
    batch_root: Path | None = None,
    skip_existing: bool = False,
    source_root: Path | None = None,
    dest_root: Path | None = None,
    out_dir: Path | None = None,
    xml_path: Path | None = None,
    data_dir: Path | None = None,
    slim_pdf: bool | None = None,
    write_removed: bool | None = None,
    write_means: bool | None = None,
    progress: object | None = None,
) -> ProcessResult:
    """Full stack: one growing mask per frame. Does not overwrite v22/v3.

    With ``dest_root``, write only under the destination (cleaned TIFF, slim PDF,
    per_frame.csv, families.json, mask_recipe.json, mask_patterns.npz). Source
    trees under ``E:\\Rasmus-Guillermo`` are refused as write targets.
    """
    from .v4_report import write_v4_report

    tif_path = Path(tif_path)
    dest_mode = dest_root is not None
    if slim_pdf is None:
        slim_pdf = dest_mode
    if write_removed is None:
        write_removed = not dest_mode
    if write_means is None:
        write_means = not dest_mode

    out_dir = resolve_v4_out_dir(
        tif_path, source_root=source_root, dest_root=dest_root, out_dir=out_dir
    )
    out_tif = out_dir / f"{tif_path.stem}_defringed_v4.tif"
    removed_tif = out_dir / f"{tif_path.stem}_removed_v4.tif"
    if skip_existing:
        if dest_mode and artifacts_complete(out_dir, out_tif):
            return ProcessResult(
                status="skipped",
                message=f"exists: {out_dir}",
                out_tif=out_tif,
                out_dir=out_dir,
            )
        if not dest_mode and out_tif.is_file():
            return ProcessResult(
                status="skipped",
                message=f"exists: {out_dir.name}/{out_tif.name}",
                out_tif=out_tif,
                out_dir=out_dir,
                removed_tif=removed_tif if removed_tif.is_file() else None,
            )

    xml_copied = None
    if dest_mode and source_root is not None and xml_path is not None:
        xml_copied = copy_experiment_xml(
            xml_path, source_root, dest_root, data_dir=data_dir
        )

    catalog_families, lib_hit = _resolve_catalog(
        computer=computer,
        channel=channel,
        fingerprint=fingerprint,
        recording_date=recording_date,
        batch_root=batch_root,
    )
    cat_qs = [float(f["q"]) for f in catalog_families if f.get("q") is not None]

    with tifffile.TiffFile(tif_path) as tf:
        shape = tf.series[0].shape
        if len(shape) != 3:
            return ProcessResult(status="error", message=f"bad shape {shape}")
        n, h, w = (int(shape[0]), int(shape[1]), int(shape[2]))
        dtype = tf.pages[0].dtype
        shutter_det = detect_shutter_windows(scan_frame_stats(tf))
        shutter_pub = shutter_public(shutter_det)
        shutter_idx = {int(i) for i in (shutter_pub.get("frames") or [])}
        shutter_frames = [np.asarray(tf.pages[i].asarray()) for i in sorted(shutter_idx)]
        shutter_families: list[dict] = []
        if shutter_frames:
            shutter_families, _ = learn_shutter_families(shutter_frames)
        shut_qs = [float(f["q"]) for f in shutter_families]
        print(f"v4 stack  {tif_path}  {computer}/{channel}", flush=True)
        print(f"  dest: {out_dir}", flush=True)
        print(f"  catalog qs={cat_qs or 'none'}  branch={(lib_hit or {}).get('branch')}", flush=True)
        print(f"  shutter: {format_shutter_span(shutter_det)}  learn qs={shut_qs or 'none'}", flush=True)
        if progress is not None:
            progress.set_n_frames(n)

        assert_writable(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        est = n * h * w * np.dtype(dtype).itemsize
        rem_b = n * h * w * 4
        rows: list[dict] = []
        acc_raw = np.zeros((h, w), dtype=np.float64)
        acc_cl = np.zeros((h, w), dtype=np.float64)
        acc_rm = np.zeros((h, w), dtype=np.float64)
        acc_pr = np.zeros((h, w), dtype=np.float64)
        catalog_used = False
        finals: list = []
        history_masks: list = []
        history_frame: list[int] = []
        history_i: list[int] = []
        history_kept: list[int] = []
        history_alpha: list[float] = []
        peaks_all: list = []
        recipes: list[dict] = []
        keep_images = not slim_pdf
        cleaned_target = out_dir / f"{out_tif.stem}.partial.tif" if dest_mode else out_tif
        twr = None
        try:
            tw = tifffile.TiffWriter(cleaned_target, bigtiff=est > 3_500_000_000)
            if write_removed:
                twr = tifffile.TiffWriter(removed_tif, bigtiff=rem_b > 3_500_000_000)
            with tw:
                for fi in range(n):
                    frame = np.asarray(tf.pages[fi].asarray())
                    role = "shutter" if fi in shutter_idx else "live"
                    rec = grow_frame(
                        frame,
                        role=role,
                        catalog_families=catalog_families,
                        shutter_families=shutter_families,
                        keep_images=keep_images,
                    )
                    search = int(rec.get("search") or (2 if role == "shutter" else TRACK_SEARCH))
                    packed = pack_frame_export(fi, rec, search=search)
                    tw.write(packed.cleaned, contiguous=True)
                    if twr is not None:
                        twr.write(rec["removed"], contiguous=True)
                    acc_raw += np.asarray(frame, dtype=np.float64)
                    acc_cl += np.asarray(rec["cleaned"], dtype=np.float64)
                    acc_rm += np.asarray(rec["removed"], dtype=np.float64)
                    acc_pr += np.asarray(rec["predicted"], dtype=np.float64)
                    rec["removed_mean_abs"] = packed.removed_mean_abs
                    rec["removed_p99_abs"] = packed.removed_p99_abs
                    row = _frame_row(fi, role, rec)
                    row["removed_mean_abs"] = packed.removed_mean_abs
                    row["removed_p99_abs"] = packed.removed_p99_abs
                    rows.append(row)
                    finals.append(packed.final)
                    for step in packed.history_meta:
                        history_masks.append(packed.history_masks[step["i"]])
                        history_frame.append(fi)
                        history_i.append(int(step["i"]))
                        history_kept.append(1 if step["kept"] else 0)
                        history_alpha.append(float(step["alpha"]))
                    for axis, q, smask in packed.peaks:
                        peaks_all.append((fi, axis, q, smask))
                    recipes.append(packed.recipe)
                    if any(a.get("source") == "catalog" for a in packed.accepted):
                        catalog_used = True
                    if (fi + 1) % 50 == 0 or fi == n - 1:
                        extra = (
                            f"lines={row['n_lines']} a={row['max_alpha']:.2f} "
                            f"rms={row['removed_rms']:.3g}"
                        )
                        if progress is not None:
                            progress.tick_frame(fi + 1, n, extra=extra)
                        else:
                            print(f"  frames {fi + 1}/{n}  {extra}", flush=True)
                    del rec
                    del packed
            if twr is not None:
                twr.close()
                twr = None
        except Exception:
            if twr is not None:
                try:
                    twr.close()
                except Exception:
                    pass
            if dest_mode and cleaned_target.is_file() and cleaned_target != out_tif:
                try:
                    cleaned_target.unlink()
                except OSError:
                    pass
            raise

        inspect_spec = pick_inspect_frames(rows, shutter_pub)
        inspect: list[dict] = []
        if not slim_pdf:
            print(f"  inspect frames: {inspect_spec}", flush=True)
            for idx, why in inspect_spec:
                frame = np.asarray(tf.pages[int(idx)].asarray())
                role = "shutter" if int(idx) in shutter_idx else "live"
                rec = grow_frame(
                    frame,
                    role=role,
                    catalog_families=catalog_families,
                    shutter_families=shutter_families,
                    keep_images=True,
                )
                rec["frame"] = int(idx)
                rec["role"] = f"{role}/{why}"
                inspect.append(rec)
        else:
            print(f"  inspect (slim, not re-grown): {inspect_spec}", flush=True)

    inv = 1.0 / max(n, 1)
    csv_path = out_dir / "per_frame.csv"
    assert_writable(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        wri.writeheader()
        for r in rows:
            wri.writerow(r)

    n_empty = int(sum(1 for r in rows if r["empty"]))
    n_brake = int(sum(1 for r in rows if r["brake"]))
    n_core = int(sum(1 for r in rows if r["core_only"]))
    n_active = int(sum(1 for r in rows if r["n_lines"] > 0))
    rms = np.array([r["removed_rms"] for r in rows], dtype=float)
    summary = {
        "n_frames": n,
        "n_empty": n_empty,
        "n_core_only": n_core,
        "n_brake": n_brake,
        "n_active": n_active,
        "frac_frames_any_active": float(n_active / max(n, 1)),
        "median_removed_rms": float(np.median(rms)),
        "max_removed_rms": float(np.max(rms)) if rms.size else 0.0,
        "median_agree": float(np.median([r["agree"] for r in rows])),
        "catalog_qs": cat_qs,
        "shutter_learn_qs": shut_qs,
        "catalog_branch": (lib_hit or {}).get("branch"),
        "inspect": [{"frame": i, "why": w} for i, w in inspect_spec],
        "slim_pdf": bool(slim_pdf),
        "dest_mode": bool(dest_mode),
    }
    cat = catalog_status(lib_hit, used=catalog_used, reseeded=False, cache_used=False)
    params = {
        "alpha_live": list(ALPHA_LIVE),
        "alpha_shutter": list(ALPHA_SHUTTER),
        "agree_min": AGREE_MIN,
        "y_radius": Y_RADIUS,
        "max_lines": MAX_LINES,
        "pack_d": _jsonable(PACK_D),
        "q_cluster_tol": Q_CLUSTER_TOL,
        "track_search": TRACK_SEARCH,
    }
    payload = {
        "version": "v4",
        "status": "ok",
        "source_tif": str(tif_path),
        "dest_tif": str(out_tif),
        "dest_dir": str(out_dir),
        "experiment_xml": str(xml_copied) if xml_copied else (str(xml_path) if xml_path else None),
        "channel": channel,
        "computer": computer,
        "fingerprint": _jsonable(fingerprint or {}),
        "shape": [n, h, w],
        "shutter": shutter_pub,
        "catalog": cat,
        "params": params,
        "summary": summary,
    }
    families_path = out_dir / "families.json"
    assert_writable(families_path)
    families_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    recipe_payload = {
        "version": "v4_mask_recipe",
        "engine": "process_v4",
        "source_tif": str(tif_path),
        "dest_tif": str(out_tif),
        "channel": channel,
        "computer": computer,
        "shape": [n, h, w],
        "params": params,
        "shutter": {"frames": shutter_pub.get("frames"), "runs": shutter_pub.get("runs")},
        "catalog_qs": cat_qs,
        "shutter_learn_qs": shut_qs,
        "frames": recipes,
    }
    recipe_path = out_dir / "mask_recipe.json"
    assert_writable(recipe_path)
    recipe_path.write_text(json.dumps(_jsonable(recipe_payload), indent=2), encoding="utf-8")
    npz_path = save_mask_patterns(
        out_dir / "mask_patterns.npz",
        shape_nwh=(n, h, w),
        finals=finals,
        history_masks=history_masks,
        history_frame=history_frame,
        history_i=history_i,
        history_kept=history_kept,
        history_alpha=history_alpha,
        peaks=peaks_all,
        meta={
            "source_tif": str(tif_path),
            "dest_tif": str(out_tif),
            "channel": channel,
            "computer": computer,
            "params": params,
        },
    )
    if write_means:
        write_mean_tif(out_dir / "mean_raw.tif", acc_raw * inv)
        write_mean_tif(out_dir / "mean_cleaned.tif", acc_cl * inv)
        write_mean_tif(out_dir / "mean_removed.tif", acc_rm * inv)
        write_mean_tif(out_dir / "mean_predicted.tif", acc_pr * inv)
    pdf_path = out_dir / "overview.pdf"
    assert_writable(pdf_path)
    pdf = write_v4_report(
        pdf_path,
        tif_path=tif_path,
        channel=channel,
        computer=computer,
        shutter=shutter_pub,
        rows=rows,
        inspect=inspect,
        mean_raw=acc_raw * inv,
        mean_cleaned=acc_cl * inv,
        mean_removed=acc_rm * inv,
        mean_predicted=acc_pr * inv,
        summary=summary,
        slim=bool(slim_pdf),
        dest_note=f"wrote {out_dir}" if dest_mode else None,
    )
    if dest_mode and cleaned_target != out_tif and cleaned_target.is_file():
        if out_tif.is_file():
            out_tif.unlink()
        cleaned_target.replace(out_tif)
    print(f"  overview: {pdf}", flush=True)
    print(f"  masks: {npz_path}", flush=True)
    return ProcessResult(
        status="ok",
        message=(
            f"v4 active on {100 * summary['frac_frames_any_active']:.1f}% frames; "
            f"median RMS {summary['median_removed_rms']:.3g}"
        ),
        out_tif=out_tif,
        out_dir=out_dir,
        removed_tif=removed_tif if write_removed and removed_tif.is_file() else None,
        overview_pdf=pdf,
        used_prior=catalog_used,
        prior_branch=(lib_hit or {}).get("branch"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination root. Required for trees under E:\\Rasmus-Guillermo.",
    )
    ap.add_argument("--seed10", action="store_true", help="10-frame probe only (ChanA seed_compare indices)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--full-pdf", action="store_true", help="Write inspect/rung PDF pages (default slim when --out)")
    ap.add_argument("--write-removed", action="store_true", help="Also write *_removed_v4.tif (off when --out)")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N stacks")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-keepalive", action="store_true", help="Do not poke source/dest drives every 20s")
    args = ap.parse_args(argv)
    jobs = []
    if args.tif is not None:
        jobs = [job_for_stack(args.tif, root=args.root)]
    elif args.root is not None:
        jobs = discover_stacks(args.root)
    else:
        print("Need --root or --tif")
        return 1
    if not jobs:
        print("No ChanA/ChanB stacks found")
        return 1
    if args.limit is not None:
        jobs = jobs[: max(0, int(args.limit))]

    dest_root = args.out.resolve() if args.out is not None else None
    source_root = args.root.resolve() if args.root is not None else None
    if dest_root is not None:
        assert_writable(dest_root)
        if source_root is not None:
            try:
                dest_root.relative_to(source_root)
                print(f"Dest {dest_root} is inside source {source_root}")
                return 1
            except ValueError:
                pass
    else:
        blocked = [j.tif_path for j in jobs if forbidden_root_for(j.tif_path) is not None]
        if blocked:
            print(
                "These stacks sit under a read-only source tree. Pass --out "
                r"(e.g. F:\CollectedData)."
            )
            for p in blocked[:8]:
                print(f"  {p}")
            return 1

    skip_existing = dest_root is not None
    if args.no_skip_existing:
        skip_existing = False
    elif args.skip_existing:
        skip_existing = True

    slim_pdf = dest_root is not None and not args.full_pdf
    write_removed = bool(args.write_removed) if dest_root is not None else True

    print(f"Found {len(jobs)} channel stacks.", flush=True)
    for j in jobs[:12]:
        print(f"  - {j.tif_path}  [{j.computer}/{j.channel}]", flush=True)
    if len(jobs) > 12:
        print(f"  ... and {len(jobs) - 12} more", flush=True)
    if args.dry_run:
        print("Dry-run only - no defringe performed.")
        return 0

    run_dir = None
    logfh = None
    if dest_root is not None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = dest_root / ".defringe_v4_runs" / run_id
        assert_writable(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "jobs.json").write_text(
            json.dumps(
                [
                    {
                        "tif": str(j.tif_path),
                        "channel": j.channel,
                        "computer": j.computer,
                        "xml": str(j.xml_path) if j.xml_path else None,
                    }
                    for j in jobs
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        logfh = open(run_dir / "summary.jsonl", "w", encoding="utf-8")
        print(f"Run log: {run_dir}", flush=True)
        print(USB_POWER_HINT, flush=True)

    progress_paths = [_REPO / ".defringe_progress.txt"]
    if run_dir is not None:
        progress_paths.append(run_dir / "progress.txt")
    progress = BatchProgress(n_jobs=len(jobs), paths=progress_paths)
    print(f"Progress file: {_REPO / '.defringe_progress.txt'}", flush=True)
    if run_dir is not None:
        print(f"             also {run_dir / 'progress.txt'}", flush=True)

    def _log_line(payload: dict) -> None:
        if logfh is None:
            return
        try:
            logfh.write(json.dumps(payload) + "\n")
            logfh.flush()
        except OSError as exc:
            print(f"  [warn] run log write failed: {exc}", flush=True)

    def _wait_drives() -> bool:
        if source_root is not None and not path_reachable(source_root):
            print(
                f"Source {source_root} unreachable; waiting up to {DRIVE_WAIT_SEC:.0f}s …",
                flush=True,
            )
            if not wait_until_reachable(source_root, timeout=DRIVE_WAIT_SEC):
                print("Source still gone; stopping so a later run can resume.", flush=True)
                return False
        if dest_root is not None and not path_reachable(dest_root):
            print(
                f"Destination {dest_root} unreachable; waiting up to {DRIVE_WAIT_SEC:.0f}s …",
                flush=True,
            )
            if not wait_until_reachable(dest_root, timeout=DRIVE_WAIT_SEC):
                print("Destination still gone; stopping so a later run can resume.", flush=True)
                return False
        return True

    rc = 0
    counts = {"ok": 0, "skipped": 0, "error": 0}
    consecutive_io = 0
    ka_read = [] if args.no_keepalive else [source_root]
    ka_write = [] if args.no_keepalive else [dest_root]
    with DriveKeepAlive(read_paths=ka_read, write_paths=ka_write):
        try:
            for si, job in enumerate(jobs, 1):
                if not _wait_drives():
                    rc = 1
                    break
                batch_root = source_root if source_root is not None else job.trial_dir
                print(f"\n[{si}/{len(jobs)}] {job.tif_path}  ({job.computer}/{job.channel})", flush=True)
                progress.begin_job(si, str(job.tif_path), channel=job.channel)
                status = "error"
                message = ""
                out_dir = None
                last_exc: BaseException | None = None
                for attempt in range(1, IO_RETRIES + 1):
                    try:
                        if args.seed10:
                            seed_out = None
                            if dest_root is not None and source_root is not None:
                                seed_out = v4_dest_dir(job.tif_path, source_root, dest_root) / "seed10"
                            process_seed10(
                                job.tif_path,
                                channel=job.channel,
                                computer=job.computer,
                                fingerprint=job.fingerprint,
                                recording_date=job.date_utc,
                                batch_root=batch_root,
                                out_dir=seed_out,
                            )
                            status, message, out_dir = "ok", "seed10", seed_out
                        else:
                            result = process_stack_v4(
                                job.tif_path,
                                channel=job.channel,
                                computer=job.computer,
                                fingerprint=job.fingerprint,
                                recording_date=job.date_utc,
                                batch_root=batch_root,
                                skip_existing=skip_existing,
                                source_root=source_root,
                                dest_root=dest_root,
                                xml_path=job.xml_path,
                                data_dir=job.data_dir,
                                slim_pdf=slim_pdf,
                                write_removed=write_removed,
                                progress=progress,
                            )
                            status, message = result.status, result.message
                            out_dir = result.out_dir
                            print(f"  {status}: {message}", flush=True)
                        last_exc = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        if is_drive_io_error(exc, source_root=source_root, dest_root=dest_root) and attempt < IO_RETRIES:
                            print(
                                f"  I/O ({exc}); retry {attempt}/{IO_RETRIES} in "
                                f"{IO_RETRY_SLEEP:.0f}s",
                                flush=True,
                            )
                            time.sleep(IO_RETRY_SLEEP)
                            if not _wait_drives():
                                break
                            continue
                        break
                if last_exc is None:
                    consecutive_io = 0
                    counts[status] = counts.get(status, 0) + 1
                    progress.end_job(status)
                    _log_line(
                        {
                            "i": si,
                            "tif": str(job.tif_path),
                            "channel": job.channel,
                            "computer": job.computer,
                            "status": status,
                            "message": message,
                            "out_dir": str(out_dir) if out_dir else None,
                        }
                    )
                else:
                    print(f"  ERROR {job.tif_path}: {last_exc}", flush=True)
                    counts["error"] += 1
                    rc = 1
                    progress.end_job("error")
                    _log_line(
                        {
                            "i": si,
                            "tif": str(job.tif_path),
                            "channel": job.channel,
                            "status": "error",
                            "message": str(last_exc),
                        }
                    )
                    if is_drive_io_error(last_exc, source_root=source_root, dest_root=dest_root):
                        consecutive_io += 1
                        if consecutive_io >= CONSECUTIVE_IO_STOP:
                            print(
                                f"{consecutive_io} consecutive drive I/O failures; "
                                "stopping so a later run can resume.",
                                flush=True,
                            )
                            break
                    else:
                        consecutive_io = 0
                    if dest_root is not None:
                        _refresh_catchup(dest_root)
        finally:
            if logfh is not None:
                try:
                    logfh.close()
                except OSError:
                    pass
    print(
        f"\nDone. ok={counts.get('ok', 0)}  skipped={counts.get('skipped', 0)}  "
        f"error={counts.get('error', 0)}",
        flush=True,
    )
    if run_dir is not None:
        print(f"  log: {run_dir}", flush=True)
    if dest_root is not None:
        _refresh_catchup(dest_root)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
