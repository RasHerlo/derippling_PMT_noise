"""v3 stack defringe: shutter detect, linescan+FFT proposals, image-test leftover, union apply.

Writes ``<channel>/defringe_v3/`` and does **not** overwrite ``defringe_v22``.
Does not append the production catalog.

``python -m batch_defringe.process_v3 --root <experiment>``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import detect_families, fft_log_amp, search_q  # noqa: E402
from pmt_fringe_raw_adaptive_v21 import (  # noqa: E402
    _attenuate_family_on_amp,
    _effective_max_alpha,
)

from .congruence import congruent_seed
from .discover import discover_stacks, job_for_stack
from .experiment_xml import fingerprint_compatible
from .image_check import (
    DETECT_LEFTOVER,
    MAX_CONSECUTIVE_REJECTS,
    MAX_ROUNDS,
    FrameEval,
    _aggregate_verdict,
)
from .library import catalog_status, lookup_prior
from .priors import load_prior
from .process import PACK_D, ProcessResult
from .readout import _jsonable, tracking_row
from .seed import (
    EVAL_ANCHOR_FRAMES,
    FORBIDDEN_Q_RADIUS,
    MAX_FAMILIES,
    Q_CLUSTER_TOL,
    TRACK_SEARCH,
    hydrate_families,
    library_family_supported,
)
from .shutter_detect import (
    detect_shutter_windows,
    format_shutter_span,
    scan_frame_stats,
    shutter_public,
)
from .shutter_seed_test import _is_nyquist_self_pair, learn_shutter_families
from .spatial_seed import (
    GATE_HIGH,
    GATE_LOW,
    _attenuate_fx_family_on_amp,
    _xvalid,
    _yvalid,
    fx_family_from_q,
    fy_family_from_q,
    notch_fx,
    notch_fy,
    search_fx,
)

V3_DIR = "defringe_v3"
MAX_EVAL_LIVE = 3
APPLY_SEARCH = 2


@dataclass
class V3Cand:
    source: str
    axis: str
    family: dict
    note: str = ""

    @property
    def q(self) -> float:
        return float(self.family["q"])


@dataclass
class KeptFamily:
    axis: str
    source: str
    family: dict
    q_seed: float
    eval_round: int
    note: str = ""


def v3_out_dir(tif_path: Path) -> Path:
    return Path(tif_path).parent / V3_DIR


def v3_cleaned_path(tif_path: Path) -> Path:
    return v3_out_dir(tif_path) / f"{Path(tif_path).stem}_defringed_v3.tif"


def v3_removed_path(tif_path: Path) -> Path:
    return v3_out_dir(tif_path) / f"{Path(tif_path).stem}_removed_v3.tif"


def _key(axis: str, q: float) -> tuple[str, int]:
    return (str(axis), int(round(float(q))))


def _used(axis: str, q: float, seen: list[tuple[str, int]]) -> bool:
    k = _key(axis, q)
    return any(a == k[0] and abs(a_q - k[1]) < Q_CLUSTER_TOL for a, a_q in seen)


def _eval_search(cand: V3Cand, role: str) -> int:
    """Live frames may walk ±10 (ChanA 160 linescan qx 15→20). Shutter stays tight."""
    if role == "shutter":
        return 2
    return TRACK_SEARCH


def _soften_leftover_votes(evals: list[FrameEval], *, kept: list[KeptFamily], cand: V3Cand) -> None:
    """Leftover shutter often still gates on residual junk after fy was kept.

    That must not veto a live-only linescan (ChanA 160 fx). Weak live notches
    relative to the proposing frame are treated as inactive, not FAIL votes.
    """
    if not evals:
        return
    peak = max((e.removed_rms for e in evals if e.active), default=0.0)
    leftover = bool(kept) or cand.source.startswith("linescan") or cand.source == "fft_detect"
    for e in evals:
        if not e.active:
            continue
        if leftover and e.role == "shutter" and not e.passed:
            e.active = False
            e.skip_reason = "leftover shutter after prior keep"
            continue
        if leftover and e.role != "shutter" and not e.passed and peak > 0 and e.removed_rms < 0.45 * peak:
            e.active = False
            e.skip_reason = "weak leftover vs proposing frame"


def unique_cands(cands: list[V3Cand], used: list[tuple[str, int]]) -> list[V3Cand]:
    out: list[V3Cand] = []
    seen = list(used)
    for c in cands:
        if _used(c.axis, c.q, seen):
            continue
        out.append(c)
        seen.append(_key(c.axis, c.q))
    return out


def pick_eval_frames(n: int, shutter_frames: list[int]) -> tuple[list[int], dict[int, str]]:
    roles: dict[int, str] = {}
    idx: list[int] = []
    for i in shutter_frames:
        if 0 <= i < n:
            idx.append(int(i))
            roles[int(i)] = "shutter"
    for a in EVAL_ANCHOR_FRAMES + (700,):
        if 0 <= int(a) < n and int(a) not in roles:
            idx.append(int(a))
            roles[int(a)] = "live_anchor"
    if len([i for i in idx if roles[i] != "shutter"]) < MAX_EVAL_LIVE and n > 4:
        mid = n // 2
        if mid not in roles:
            idx.append(mid)
            roles[mid] = "live"
    return sorted(set(idx)), roles


def fy_cands_from_frames(frames: list[np.ndarray], source: str) -> list[V3Cand]:
    if not frames:
        return []
    medspec = np.median(np.stack([fft_log_amp(f) for f in frames]), axis=0)
    families, _, _ = detect_families(medspec, **DETECT_LEFTOVER)
    if not families:
        return []
    height = int(medspec.shape[0])
    hydrated = hydrate_families(families, medspec, x_z_thresh=float(DETECT_LEFTOVER["x_z_thresh"]))
    out = []
    for fam in hydrated:
        if _is_nyquist_self_pair(fam, height):
            continue
        fam = dict(fam)
        fam["axis"] = "fy"
        out.append(V3Cand(source=source, axis="fy", family=fam, note="fy-row detect"))
    return out


def linescan_cands(eval_frames: dict[int, np.ndarray]) -> list[V3Cand]:
    out: list[V3Cand] = []
    for _idx, frame in eval_frames.items():
        seed = congruent_seed(frame)
        if not seed.get("ok"):
            continue
        logamp = fft_log_amp(frame)
        if seed.get("winner") in ("fx", "tilted") and seed.get("qx") is not None:
            fam = fx_family_from_q(logamp, float(seed["qx"]))
            fam["axis"] = "fx"
            out.append(
                V3Cand(
                    source="linescan_fx",
                    axis="fx",
                    family=fam,
                    note=f"congruence qx={float(seed['qx']):.1f} score={seed.get('score')}",
                )
            )
        if seed.get("winner") in ("fy", "tilted") and seed.get("qy") is not None:
            fam = fy_family_from_q(logamp, float(seed["qy"]))
            fam["axis"] = "fy"
            out.append(
                V3Cand(
                    source="linescan_fy",
                    axis="fy",
                    family=fam,
                    note=f"congruence qy={float(seed['qy']):.1f} score={seed.get('score')}",
                )
            )
    return out


def propose_v3(
    leftover: list[np.ndarray],
    eval_map: dict[int, np.ndarray],
    *,
    used: list[tuple[str, int]],
    catalog_families: list[dict] | None,
    medspec: np.ndarray | None,
    shutter_families: list[dict],
    include_linescan: bool,
) -> list[V3Cand]:
    cands: list[V3Cand] = []
    if catalog_families and medspec is not None:
        for src in catalog_families:
            if not library_family_supported(src, medspec):
                continue
            hydrated = hydrate_families([src], medspec)
            if not hydrated:
                continue
            fam = dict(hydrated[0])
            fam["axis"] = "fy"
            cands.append(V3Cand(source="catalog", axis="fy", family=fam, note="library A/B/C"))
    for fam in shutter_families:
        f = dict(fam)
        f["axis"] = "fy"
        cands.append(V3Cand(source="shutter_hint", axis="fy", family=f, note="in-stack shutter learn"))
    # Linescan before leftover FFT so ChanA fx (e.g. frame 160) is not starved
    # by a weaker fy leftover row that then fails image-test twice and stops.
    if include_linescan:
        cands.extend(linescan_cands(eval_map))
    cands.extend(fy_cands_from_frames(leftover, "fft_detect"))
    return unique_cands(cands, used)[:8]


def notch_cand(frame: np.ndarray, cand: V3Cand, *, search: int) -> dict:
    if cand.axis == "fx":
        return notch_fx(frame, cand.family, cand.q, search=search)
    return notch_fy(frame, cand.family, cand.q, search=search)


def apply_frame(
    frame: np.ndarray,
    kept: list[KeptFamily],
    q_preds: list[float],
    *,
    search: int = APPLY_SEARCH,
) -> dict:
    """Union apply: one FFT, every kept family on its own axis, then one IFFT."""
    params = dict(PACK_D)
    params.update(y_sigma=1.0, y_radius=2)
    orig_dtype = frame.dtype
    x = np.asarray(frame, dtype=np.float32)
    offset = float(np.median(x))
    x0 = x - offset
    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j * np.angle(F))
    logamp = np.log1p(amp)
    h, w = x.shape
    newamp = amp.copy()
    tracking: list[dict] = []
    xvalid = _xvalid(w)
    yvalid = _yvalid(h)

    for fam, q_pred in zip(kept, q_preds):
        if fam.axis == "fy":
            q, strength = search_q(logamp, float(q_pred), True, xvalid, int(search))
        else:
            q, strength = search_fx(logamp, float(q_pred), True, yvalid, int(search))
        gate = float(np.clip((strength - GATE_LOW) / max(1e-9, GATE_HIGH - GATE_LOW), 0.0, 1.0))
        paired = bool(fam.family.get("paired", True))
        eff_alpha = _effective_max_alpha(
            gate,
            strength,
            paired,
            max_alpha=params["max_alpha"],
            max_alpha_high=params["max_alpha_high"],
            high_gate=params["high_gate"],
            high_strength=params["high_strength"],
            strength_span=params["strength_span"],
        )
        tracking.append(
            {
                "q": float(q),
                "q_pred": float(q_pred),
                "q_seed": float(fam.q_seed),
                "strength": float(strength),
                "gate": gate,
                "eff_max_alpha": float(eff_alpha),
                "axis": fam.axis,
                "source": fam.source,
                "residual_pass": 0,
                "residual_strength": 0.0,
            }
        )
        if gate > 0:
            if fam.axis == "fy":
                _attenuate_family_on_amp(
                    amp,
                    newamp,
                    fam.family,
                    q,
                    gate,
                    max_alpha=eff_alpha,
                    ratio_start=params["ratio_start"],
                    ratio_full=params["ratio_full"],
                    y_sigma=params["y_sigma"],
                    y_radius=int(params["y_radius"]),
                )
            else:
                _attenuate_fx_family_on_amp(
                    amp,
                    newamp,
                    fam.family,
                    q,
                    gate,
                    max_alpha=eff_alpha,
                    ratio_start=params["ratio_start"],
                    ratio_full=params["ratio_full"],
                )

    applied = np.clip((amp - newamp) / (amp + 1e-12), 0.0, 1.0).astype(np.float32)
    predicted = np.real(np.fft.ifft2(np.fft.ifftshift((amp - newamp) * phase))).astype(np.float32)
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(newamp * phase))) + offset
    removed = x - cleaned
    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_w = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_w = cleaned.astype(orig_dtype)
    return {
        "cleaned": cleaned_w,
        "removed": removed.astype(np.float32),
        "predicted": predicted,
        "applied": applied,
        "tracking": tracking,
        "removed_rms": float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2))),
    }


def track_kept(tf: tifffile.TiffFile, kept: list[KeptFamily], *, block_size: int = 50) -> list[np.ndarray]:
    """Per-family q walk. Same-axis families cannot hop onto each other."""
    n, h, w = tf.series[0].shape
    xvalid = _xvalid(w)
    yvalid = _yvalid(h)
    q_states = [float(k.q_seed) for k in kept]
    traj = [np.full(n, q, dtype=float) for q in q_states]
    if not kept:
        return traj
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        inds = np.linspace(start, stop - 1, min(8, stop - start), dtype=int)
        specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]
        block = np.median(np.stack(specs), axis=0)
        new_states = list(q_states)
        for i, spec in enumerate(kept):
            forbidden = [
                q_states[j]
                for j in range(len(kept))
                if j != i and kept[j].axis == spec.axis
            ]
            if spec.axis == "fy":
                q_found, score = search_q(
                    block,
                    q_states[i],
                    True,
                    xvalid,
                    TRACK_SEARCH,
                    forbidden_qs=forbidden,
                    forbidden_radius=FORBIDDEN_Q_RADIUS,
                )
            else:
                q_found, score = search_fx(block, q_states[i], True, yvalid, TRACK_SEARCH)
                if any(abs(q_found - fq) < FORBIDDEN_Q_RADIUS for fq in forbidden):
                    q_found, score = q_states[i], 0.0
            if np.isfinite(score) and score >= 0.08:
                new_states[i] = float(q_found)
        q_states = new_states
        for i in range(len(kept)):
            traj[i][start:stop] = q_states[i]
    return traj


def _snapshot(ev: FrameEval, cand: V3Cand, verdict: str, rnd: int) -> dict:
    return {
        "frame": ev.frame,
        "role": ev.role,
        "round": rnd,
        "source": cand.source,
        "axis": cand.axis,
        "q": ev.q,
        "q_proposed": cand.q,
        "verdict": verdict,
        "passed": ev.passed,
        "gate": ev.gate,
        "removed_rms": ev.removed_rms,
        "raw": np.asarray(ev.raw),
        "cleaned": np.asarray(ev.cleaned),
        "removed": np.asarray(ev.removed),
        "applied": None if ev.applied is None else np.asarray(ev.applied),
    }


def eval_loop(
    eval_map: dict[int, np.ndarray],
    roles: dict[int, str],
    *,
    catalog_families: list[dict] | None,
    medspec: np.ndarray | None,
    shutter_families: list[dict],
) -> tuple[list[KeptFamily], list[dict], list[dict]]:
    order = sorted(eval_map)
    working = {i: np.asarray(eval_map[i]) for i in order}
    used: list[tuple[str, int]] = []
    kept: list[KeptFamily] = []
    discarded: list[dict] = []
    rounds: list[dict] = []
    consecutive_rejects = 0

    for rnd in range(MAX_ROUNDS):
        leftover = [working[i] for i in order]
        leftover_map = {i: working[i] for i in order}
        cands = propose_v3(
            leftover,
            leftover_map,
            used=used,
            catalog_families=catalog_families,
            medspec=medspec,
            shutter_families=shutter_families,
            include_linescan=True,
        )
        if not cands or len(kept) >= MAX_FAMILIES:
            break
        cand = cands[0]
        used.append(_key(cand.axis, cand.q))
        evals: list[FrameEval] = []
        snap = None
        for idx in order:
            search = _eval_search(cand, roles.get(idx, "live"))
            tr = notch_cand(working[idx], cand, search=search)
            gate = float(tr["gate"])
            score = tr["score"] or {
                "passed": False,
                "traits": [],
                "rms": tr["removed_rms"],
                "by_name": {},
            }
            ev = FrameEval(
                frame=int(idx),
                role=roles.get(idx, "live"),
                gate=gate,
                q=float(tr["q"]),
                removed_rms=float(tr["removed_rms"]),
                active=gate > 0,
                score=score,
                passed=bool(tr["passed"]),
                raw=working[idx],
                cleaned=tr["cleaned"],
                removed=tr["removed"],
                skip_reason=None if gate > 0 else "gate=0",
                applied=tr["applied"],
                q_proposed=cand.q,
            )
            evals.append(ev)
            if snap is None and gate > 0:
                snap = ev
        _soften_leftover_votes(evals, kept=kept, cand=cand)
        verdict, reason = _aggregate_verdict(evals)
        rounds.append(
            {
                "round": rnd + 1,
                "source": cand.source,
                "axis": cand.axis,
                "q": cand.q,
                "verdict": verdict,
                "reason": reason,
                "frames": [
                    {
                        "frame": e.frame,
                        "role": e.role,
                        "gate": e.gate,
                        "q": e.q,
                        "passed": e.passed,
                        "active": e.active,
                        "removed_rms": e.removed_rms,
                    }
                    for e in evals
                ],
            }
        )
        if verdict == "accept":
            kept.append(
                KeptFamily(
                    axis=cand.axis,
                    source=cand.source,
                    family=cand.family,
                    q_seed=float(np.median([e.q for e in evals if e.active] or [cand.q])),
                    eval_round=rnd + 1,
                    note=cand.note,
                )
            )
            for e in evals:
                if e.active and e.passed:
                    working[e.frame] = np.asarray(e.cleaned)
            consecutive_rejects = 0
        elif verdict == "inactive":
            continue
        else:
            consecutive_rejects += 1
            if snap is not None:
                discarded.append(_snapshot(snap, cand, verdict, rnd + 1))
            if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                break
    return kept, rounds, discarded[:4]


def _per_frame_fields(n_fam: int) -> list[str]:
    fields = [
        "frame",
        "n_active_families",
        "removed_rms",
        "removed_mean_abs",
        "removed_p99_abs",
        "max_gate",
        "max_eff_max_alpha",
        "n_residual_passes",
    ]
    for i in range(n_fam):
        fields += [
            f"family{i}_q",
            f"family{i}_q_pred",
            f"family{i}_q_seed",
            f"family{i}_strength",
            f"family{i}_gate",
            f"family{i}_eff_max_alpha",
            f"family{i}_axis",
            f"family{i}_source",
            f"family{i}_active",
            f"family{i}_residual_pass",
            f"family{i}_residual_strength",
            f"family{i}_drift",
        ]
    return fields


def _row(frame_i: int, removed: np.ndarray, tracking: list[dict]) -> dict:
    row = tracking_row(frame_i, removed, tracking)
    for i, t in enumerate(tracking):
        row[f"family{i}_q_pred"] = t.get("q_pred")
        row[f"family{i}_q_seed"] = t.get("q_seed")
        row[f"family{i}_axis"] = t.get("axis")
        row[f"family{i}_source"] = t.get("source")
        row[f"family{i}_drift"] = abs(float(t.get("q") or 0) - float(t.get("q_seed") or 0))
    return row


def process_stack_v3(
    tif_path: Path,
    *,
    batch_root: Path,
    computer: str,
    channel: str,
    fingerprint: dict,
    recording_date: str | None = None,
    skip_existing: bool = True,
    force_fresh_seed: bool = False,
) -> ProcessResult:
    tif_path = Path(tif_path)
    out_dir = v3_out_dir(tif_path)
    out_tif = v3_cleaned_path(tif_path)
    removed_tif = v3_removed_path(tif_path)
    if skip_existing and out_tif.is_file():
        return ProcessResult(
            status="skipped",
            message=f"exists: {out_dir.name}/{out_tif.name}",
            out_tif=out_tif,
            out_dir=out_dir,
            removed_tif=removed_tif if removed_tif.is_file() else None,
        )

    lib_hit = None if force_fresh_seed else lookup_prior(
        computer=computer,
        channel=channel,
        fingerprint=fingerprint,
        recording_date=recording_date,
        batch_root=batch_root,
    )
    prior = None if force_fresh_seed else load_prior(batch_root, computer, channel)
    catalog_families = list((lib_hit or {}).get("families") or [])
    if (
        not catalog_families
        and prior
        and prior.get("families")
        and fingerprint_compatible(prior.get("fingerprint"), fingerprint)
    ):
        catalog_families = list(prior["families"])

    with tifffile.TiffFile(tif_path) as tf:
        shape = tf.series[0].shape
        if len(shape) != 3:
            return ProcessResult(status="error", message=f"bad shape {shape}")
        n, h, w = shape
        dtype = tf.pages[0].dtype
        shutter_det = detect_shutter_windows(scan_frame_stats(tf))
        shutter_pub = shutter_public(shutter_det)
        print(f"      shutter auto: {format_shutter_span(shutter_det)}", flush=True)
        eval_idx, roles = pick_eval_frames(n, shutter_pub.get("frames") or [])
        eval_map = {i: np.asarray(tf.pages[i].asarray()) for i in eval_idx}
        print(f"      eval frames: {eval_idx}", flush=True)

        shutter_frames = [eval_map[i] for i in eval_idx if roles.get(i) == "shutter"]
        shutter_families: list[dict] = []
        if shutter_frames:
            shutter_families, _ = learn_shutter_families(shutter_frames)
            print(
                f"      shutter families q={[float(f['q']) for f in shutter_families]}",
                flush=True,
            )

        medspec = np.median(np.stack([fft_log_amp(eval_map[i]) for i in eval_idx]), axis=0)
        kept, rounds, discarded = eval_loop(
            eval_map,
            roles,
            catalog_families=catalog_families or None,
            medspec=medspec,
            shutter_families=shutter_families,
        )
        print(
            f"      kept {len(kept)} families: "
            + ", ".join(f"{k.source}/{k.axis} q={k.q_seed:.1f}" for k in kept),
            flush=True,
        )

        cat = catalog_status(
            lib_hit,
            used=any(k.source == "catalog" for k in kept),
            reseeded=False,
            cache_used=False,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        if not kept:
            print("      no kept families — writing identity cleaned + zero removed", flush=True)
            est = n * h * w * np.dtype(dtype).itemsize
            rem_b = n * h * w * 4
            acc_raw = np.zeros((h, w), dtype=np.float64)
            with (
                tifffile.TiffWriter(out_tif, bigtiff=est > 3_500_000_000) as tw,
                tifffile.TiffWriter(removed_tif, bigtiff=rem_b > 3_500_000_000) as twr,
            ):
                for fi in range(n):
                    frame = tf.pages[fi].asarray()
                    tw.write(frame, contiguous=True)
                    twr.write(np.zeros((h, w), dtype=np.float32), contiguous=True)
                    acc_raw += np.asarray(frame, dtype=np.float64)
                    if (fi + 1) % 200 == 0 or fi == n - 1:
                        print(f"      frames {fi + 1}/{n}", flush=True)
            inv = 1.0 / max(n, 1)
            summary = {
                "n_frames": n,
                "n_families": 0,
                "frac_frames_any_active": 0.0,
                "median_removed_rms": 0.0,
                "max_removed_rms": 0.0,
            }
            payload = {
                "version": "v3",
                "status": "needs_review",
                "source_tif": str(tif_path),
                "computer": computer,
                "channel": channel,
                "shutter": shutter_pub,
                "eval_frames": eval_idx,
                "rounds": rounds,
                "kept": [],
                "catalog": cat,
                "summary": summary,
                "message": "no family passed the image-test leftover loop",
            }
            (out_dir / "families.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
            from .readout import write_mean_tif

            write_mean_tif(out_dir / "mean_raw.tif", acc_raw * inv)
            write_mean_tif(out_dir / "mean_cleaned.tif", acc_raw * inv)
            write_mean_tif(out_dir / "mean_removed.tif", np.zeros((h, w), dtype=np.float64))
            from .v3_report import write_v3_report

            pdf = write_v3_report(
                out_dir / "overview.pdf",
                tif_path=tif_path,
                channel=channel,
                computer=computer,
                status="needs_review",
                shutter=shutter_pub,
                kept=[],
                rounds=rounds,
                discarded=discarded,
                rows=[],
                eval_idx=eval_idx,
                roles=roles,
                cleaned_tif=out_tif,
                removed_tif=removed_tif,
                mean_raw=acc_raw * inv,
                mean_cleaned=acc_raw * inv,
                mean_removed=np.zeros((h, w), dtype=np.float64),
                summary=summary,
            )
            return ProcessResult(
                status="needs_review",
                message="no family passed image-test",
                out_tif=out_tif,
                out_dir=out_dir,
                removed_tif=removed_tif,
                overview_pdf=pdf,
                prior_branch=(lib_hit or {}).get("branch"),
            )

        print("      tracking q …", flush=True)
        traj = track_kept(tf, kept)
        est = n * h * w * np.dtype(dtype).itemsize
        rem_b = n * h * w * 4
        rows: list[dict] = []
        acc_raw = np.zeros((h, w), dtype=np.float64)
        acc_clean = np.zeros((h, w), dtype=np.float64)
        acc_rem = np.zeros((h, w), dtype=np.float64)
        with (
            tifffile.TiffWriter(out_tif, bigtiff=est > 3_500_000_000) as tw,
            tifffile.TiffWriter(removed_tif, bigtiff=rem_b > 3_500_000_000) as twr,
        ):
            for fi in range(n):
                frame = tf.pages[fi].asarray()
                preds = [float(t[fi]) for t in traj]
                rec = apply_frame(frame, kept, preds, search=APPLY_SEARCH)
                tw.write(rec["cleaned"], contiguous=True)
                twr.write(rec["removed"], contiguous=True)
                acc_raw += np.asarray(frame, dtype=np.float64)
                acc_clean += np.asarray(rec["cleaned"], dtype=np.float64)
                acc_rem += np.asarray(rec["removed"], dtype=np.float64)
                rows.append(_row(fi, rec["removed"], rec["tracking"]))
                if (fi + 1) % 200 == 0 or fi == n - 1:
                    print(f"      frames {fi + 1}/{n}", flush=True)

        inv = 1.0 / max(n, 1)
        csv_path = out_dir / "per_frame.csv"
        fields = _per_frame_fields(len(kept))
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            wri = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            wri.writeheader()
            for r in rows:
                wri.writerow(r)

        rms = np.array([r["removed_rms"] for r in rows], dtype=float)
        n_act = np.array([r["n_active_families"] for r in rows], dtype=float)
        summary = {
            "n_frames": n,
            "n_families": len(kept),
            "frac_frames_any_active": float(np.mean(n_act > 0)),
            "median_removed_rms": float(np.median(rms)),
            "max_removed_rms": float(np.max(rms)),
        }
        payload = {
            "version": "v3",
            "config_id": "pack_D_union",
            "status": "ok",
            "source_tif": str(tif_path),
            "computer": computer,
            "channel": channel,
            "fingerprint": _jsonable(fingerprint),
            "shape": [int(n), int(h), int(w)],
            "params": _jsonable(PACK_D),
            "shutter": shutter_pub,
            "eval_frames": eval_idx,
            "roles": {str(k): v for k, v in roles.items()},
            "rounds": rounds,
            "catalog": cat,
            "kept": [
                {
                    "axis": k.axis,
                    "source": k.source,
                    "q_seed": k.q_seed,
                    "eval_round": k.eval_round,
                    "note": k.note,
                    "q": float(k.family["q"]),
                }
                for k in kept
            ],
            "summary": summary,
        }
        (out_dir / "families.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        from .readout import write_mean_tif

        write_mean_tif(out_dir / "mean_raw.tif", acc_raw * inv)
        write_mean_tif(out_dir / "mean_cleaned.tif", acc_clean * inv)
        write_mean_tif(out_dir / "mean_removed.tif", acc_rem * inv)

        from .v3_report import write_v3_report

        pdf = write_v3_report(
            out_dir / "overview.pdf",
            tif_path=tif_path,
            channel=channel,
            computer=computer,
            status="ok",
            shutter=shutter_pub,
            kept=kept,
            rounds=rounds,
            discarded=discarded,
            rows=rows,
            eval_idx=eval_idx,
            roles=roles,
            cleaned_tif=out_tif,
            removed_tif=removed_tif,
            mean_raw=acc_raw * inv,
            mean_cleaned=acc_clean * inv,
            mean_removed=acc_rem * inv,
            summary=summary,
        )
        return ProcessResult(
            status="ok",
            message=f"v3 kept {len(kept)} families; active on {100 * summary['frac_frames_any_active']:.1f}% frames",
            out_tif=out_tif,
            out_dir=out_dir,
            removed_tif=removed_tif,
            overview_pdf=pdf,
            families_q=[k.q_seed for k in kept],
            prior_branch=(lib_hit or {}).get("branch"),
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args(argv)
    from .v3_report import write_schematic_pdf

    schema = _REPO / "v3_pipeline_schematic.pdf"
    write_schematic_pdf(schema)
    print(f"schematic: {schema}", flush=True)

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
    rc = 0
    for job in jobs:
        print(f"\nv3  {job.tif_path}  {job.computer}/{job.channel}", flush=True)
        result = process_stack_v3(
            job.tif_path,
            batch_root=Path(args.root).resolve() if args.root else job.trial_dir,
            computer=job.computer,
            channel=job.channel,
            fingerprint=job.fingerprint,
            recording_date=job.date_utc,
            skip_existing=not args.no_skip_existing,
            force_fresh_seed=job.missing_xml,
        )
        print(f"    {result.status.upper()}: {result.message}", flush=True)
        if result.out_dir:
            print(f"    readout: {result.out_dir}", flush=True)
        if result.status in ("needs_review", "error"):
            rc = 2 if result.status == "needs_review" else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
