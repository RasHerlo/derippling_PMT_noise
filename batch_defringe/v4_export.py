"""Destination-only v4 artifacts: path mirror, E: write-guard, sparse masks, XML copy.

Source trees under ``WRITE_FORBIDDEN_ROOTS`` are never created or overwritten.
Replay of removed / applied-mask stacks uses ``mask_patterns.npz`` plus the
original raw TIFF (and ``mask_recipe.json`` for the line/rung story).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .readout import _jsonable, family_public

V4_DIR = "defringe_v4"
WRITE_FORBIDDEN_ROOTS = (Path(r"E:\Rasmus-Guillermo"),)
SPARSE_THRESH = 1e-6
NPZ_VERSION = "v4_mask_patterns_1"


@dataclass
class SparseMask:
    y: np.ndarray
    x: np.ndarray
    val: np.ndarray

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=np.uint16).reshape(-1)
        self.x = np.asarray(self.x, dtype=np.uint16).reshape(-1)
        self.val = np.asarray(self.val, dtype=np.float32).reshape(-1)
        n = int(self.y.size)
        if self.x.size != n or self.val.size != n:
            raise ValueError("sparse y/x/val length mismatch")


def empty_sparse() -> SparseMask:
    return SparseMask(
        np.zeros(0, dtype=np.uint16),
        np.zeros(0, dtype=np.uint16),
        np.zeros(0, dtype=np.float32),
    )


def sparsify(arr: np.ndarray | None, *, thresh: float = SPARSE_THRESH) -> SparseMask:
    if arr is None:
        return empty_sparse()
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return empty_sparse()
    ys, xs = np.nonzero(np.abs(a) > float(thresh))
    if ys.size == 0:
        return empty_sparse()
    return SparseMask(ys.astype(np.uint16), xs.astype(np.uint16), a[ys, xs])


def densify(mask: SparseMask, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    out = np.zeros((h, w), dtype=np.float32)
    if mask.y.size == 0:
        return out
    out[mask.y.astype(np.int64), mask.x.astype(np.int64)] = mask.val
    return out


def concat_sparse(masks: list[SparseMask]) -> tuple[np.ndarray, SparseMask]:
    ptr = [0]
    if not masks:
        return np.asarray(ptr, dtype=np.int64), empty_sparse()
    ys, xs, vs = [], [], []
    for m in masks:
        ys.append(m.y)
        xs.append(m.x)
        vs.append(m.val)
        ptr.append(ptr[-1] + int(m.y.size))
    return (
        np.asarray(ptr, dtype=np.int64),
        SparseMask(np.concatenate(ys), np.concatenate(xs), np.concatenate(vs)),
    )


def slice_sparse(ptr: np.ndarray, packed: SparseMask, i: int) -> SparseMask:
    lo = int(ptr[i])
    hi = int(ptr[i + 1])
    return SparseMask(packed.y[lo:hi], packed.x[lo:hi], packed.val[lo:hi])


def _norm_root(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def forbidden_root_for(path: Path) -> Path | None:
    """Return the protected root if *path* would be a write under it."""
    resolved = _norm_root(path)
    resolved_s = str(resolved).lower()
    for root in WRITE_FORBIDDEN_ROOTS:
        r = _norm_root(root)
        r_s = str(r).lower().rstrip("\\/")
        try:
            resolved.relative_to(r)
            return r
        except ValueError:
            pass
        if resolved_s == r_s or resolved_s.startswith(r_s + "\\") or resolved_s.startswith(r_s + "/"):
            return r
    return None


def assert_writable(path: Path) -> Path:
    hit = forbidden_root_for(path)
    if hit is not None:
        raise RuntimeError(
            f"Refusing to write {Path(path)} — source tree {hit} is read-only. "
            r"Pass --out (e.g. F:\CollectedData)."
        )
    return Path(path)


def mirror_under(path: Path, source_root: Path, dest_root: Path) -> Path:
    rel = Path(path).resolve().relative_to(Path(source_root).resolve())
    return Path(dest_root).resolve() / rel


def v4_dest_dir(tif_path: Path, source_root: Path, dest_root: Path) -> Path:
    return mirror_under(Path(tif_path).parent, source_root, dest_root) / V4_DIR


def dest_xml_path(xml_path: Path, source_root: Path, dest_root: Path, *, data_dir: Path | None = None) -> Path:
    xml_path = Path(xml_path).resolve()
    root = Path(source_root).resolve()
    try:
        return Path(dest_root).resolve() / xml_path.relative_to(root)
    except ValueError:
        if data_dir is not None:
            dest_data = mirror_under(data_dir, source_root, dest_root)
            return dest_data.parent / xml_path.name
        return Path(dest_root).resolve() / xml_path.name


def copy_experiment_xml(
    xml_path: Path | None,
    source_root: Path,
    dest_root: Path,
    *,
    data_dir: Path | None = None,
) -> Path | None:
    if xml_path is None:
        return None
    src = Path(xml_path)
    if not src.is_file():
        return None
    dest = dest_xml_path(src, source_root, dest_root, data_dir=data_dir)
    assert_writable(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == src.stat().st_size:
        return dest
    shutil.copyfile(src, dest)
    return dest


def line_recipe(line) -> dict[str, Any]:
    """JSON-safe line used as a hint; peak support lives in the npz."""
    fam = getattr(line, "family", None)
    family = None
    if isinstance(fam, dict) and fam.get("q") is not None:
        family = family_public(fam)
    return {
        "axis": line.axis,
        "q": float(line.q),
        "source": line.source,
        "kind": line.kind,
        "tier": getattr(line, "tier", ""),
        "note": getattr(line, "note", "") or "",
        "family": _jsonable(family) if family is not None else None,
        "has_peak_mask": getattr(line, "peak_mask", None) is not None,
    }


def _line_recipe_from_public(d: dict) -> dict[str, Any]:
    fam = d.get("family")
    family = None
    if isinstance(fam, dict) and fam.get("q") is not None:
        family = family_public(fam)
    return {
        "axis": d.get("axis"),
        "q": float(d.get("q") or 0),
        "source": d.get("source"),
        "kind": d.get("kind") or "ridge",
        "tier": d.get("tier") or "",
        "note": d.get("note") or "",
        "family": _jsonable(family) if family is not None else None,
        "has_peak_mask": bool(d.get("has_peak_mask")),
    }


@dataclass
class FrameExport:
    final: SparseMask
    history_masks: list[SparseMask]
    history_meta: list[dict[str, Any]]
    peaks: list[tuple[str, float, SparseMask]]
    recipe: dict[str, Any]
    cleaned: Any
    removed_rms: float
    removed_mean_abs: float
    removed_p99_abs: float
    agree: float
    n_lines: int
    max_alpha: float
    brake: bool
    empty: bool
    core_only: bool
    role: str
    seed: dict[str, Any]
    accepted: list[dict[str, Any]]
    ranked: list[dict[str, Any]]


def _peak_key(axis: str, q: float) -> tuple[str, float]:
    return (str(axis), round(float(q), 3))


def pack_frame_export(fi: int, rec: dict, *, search: int) -> FrameExport:
    rem = np.asarray(rec.get("removed"), dtype=np.float64)
    abs_rem = np.abs(rem) if rem.size else np.zeros(0, dtype=np.float64)
    peaks: list[tuple[str, float, SparseMask]] = []
    peak_index: dict[tuple[str, float], int] = {}

    def remember_peak(line) -> int | None:
        mask = getattr(line, "peak_mask", None)
        if mask is None:
            return None
        key = _peak_key(line.axis, line.q)
        if key in peak_index:
            return peak_index[key]
        idx = len(peaks)
        peaks.append((str(line.axis), float(line.q), sparsify(mask)))
        peak_index[key] = idx
        return idx

    accepted_objs = list(rec.get("accepted_lines") or [])
    accepted_pub = []
    for ln in accepted_objs:
        remember_peak(ln)
        d = line_recipe(ln)
        key = _peak_key(ln.axis, ln.q)
        d["peak_id"] = peak_index.get(key)
        accepted_pub.append(d)
    if not accepted_pub:
        accepted_pub = [_line_recipe_from_public(a) for a in (rec.get("accepted") or [])]

    history_masks: list[SparseMask] = []
    history_meta: list[dict[str, Any]] = []
    history_json: list[dict[str, Any]] = []
    for i, step in enumerate(rec.get("history") or []):
        sp = step.get("applied_sparse")
        if isinstance(sp, SparseMask):
            history_masks.append(sp)
        else:
            history_masks.append(sparsify(step.get("applied")))
        lines_j = []
        for ln in step.get("lines") or []:
            if hasattr(ln, "axis"):
                remember_peak(ln)
                d = line_recipe(ln)
                d["peak_id"] = peak_index.get(_peak_key(ln.axis, ln.q))
            else:
                d = _line_recipe_from_public(ln)
            lines_j.append(d)
        meta = {
            "i": int(i),
            "kept": bool(step.get("kept")),
            "why": step.get("why") or "",
            "reason": step.get("reason") or "",
            "alpha": float(step.get("alpha") or 0),
            "n_lines": int(step.get("n_lines") or 0),
            "n_active": int(step.get("n_active") or 0),
            "removed_rms": float(step.get("removed_rms") or 0),
            "agree": float(step.get("agree") or 0),
            "lines": lines_j,
        }
        history_meta.append(meta)
        history_json.append(meta)

    seed = rec.get("seed") or {}
    recipe = {
        "frame": int(fi),
        "role": rec.get("role") or "",
        "search": int(search),
        "max_alpha": float(rec.get("max_alpha") or 0),
        "n_lines": int(rec.get("n_lines") or 0),
        "brake": bool(rec.get("brake")),
        "empty": bool(rec.get("empty")),
        "core_only": bool(rec.get("core_only")),
        "removed_rms": float(rec.get("removed_rms") or 0),
        "agree": float(rec.get("agree") or 0),
        "seed": _jsonable(seed),
        "ranked": _jsonable(rec.get("ranked") or []),
        "accepted": accepted_pub,
        "history": history_json,
        "n_peaks": len(peaks),
    }
    return FrameExport(
        final=sparsify(rec.get("applied")),
        history_masks=history_masks,
        history_meta=history_meta,
        peaks=peaks,
        recipe=recipe,
        cleaned=rec.get("cleaned"),
        removed_rms=float(rec.get("removed_rms") or 0),
        removed_mean_abs=float(np.mean(abs_rem)) if abs_rem.size else 0.0,
        removed_p99_abs=float(np.percentile(abs_rem, 99.0)) if abs_rem.size else 0.0,
        agree=float(rec.get("agree") or 0),
        n_lines=int(rec.get("n_lines") or 0),
        max_alpha=float(rec.get("max_alpha") or 0),
        brake=bool(rec.get("brake")),
        empty=bool(rec.get("empty")),
        core_only=bool(rec.get("core_only")),
        role=str(rec.get("role") or ""),
        seed=seed,
        accepted=accepted_pub,
        ranked=list(rec.get("ranked") or []),
    )


def save_mask_patterns(
    path: Path,
    *,
    shape_nwh: tuple[int, int, int],
    finals: list[SparseMask],
    history_masks: list[SparseMask],
    history_frame: list[int],
    history_i: list[int],
    history_kept: list[int],
    history_alpha: list[float],
    peaks: list[tuple[int, str, float, SparseMask]],
    meta: dict[str, Any],
) -> Path:
    path = Path(path)
    assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w = (int(x) for x in shape_nwh)
    final_ptr, final_pack = concat_sparse(finals)
    hist_ptr, hist_pack = concat_sparse(history_masks)
    peak_masks = [p[3] for p in peaks]
    peak_ptr, peak_pack = concat_sparse(peak_masks)
    axis_code = np.array([0 if p[1] == "fy" else 1 for p in peaks], dtype=np.uint8)
    np.savez_compressed(
        path,
        version=np.asarray(NPZ_VERSION),
        n_frames=np.int32(n),
        height=np.int32(h),
        width=np.int32(w),
        meta_json=np.asarray(json.dumps(_jsonable(meta), ensure_ascii=False)),
        final_ptr=final_ptr,
        final_y=final_pack.y,
        final_x=final_pack.x,
        final_val=final_pack.val,
        history_ptr=hist_ptr,
        history_y=hist_pack.y,
        history_x=hist_pack.x,
        history_val=hist_pack.val,
        history_frame=np.asarray(history_frame, dtype=np.int32),
        history_i=np.asarray(history_i, dtype=np.int16),
        history_kept=np.asarray(history_kept, dtype=np.uint8),
        history_alpha=np.asarray(history_alpha, dtype=np.float32),
        peak_ptr=peak_ptr,
        peak_y=peak_pack.y,
        peak_x=peak_pack.x,
        peak_val=peak_pack.val,
        peak_frame=np.asarray([p[0] for p in peaks], dtype=np.int32),
        peak_axis=axis_code,
        peak_q=np.asarray([p[2] for p in peaks], dtype=np.float32),
    )
    return path


def load_mask_patterns(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        meta_raw = z["meta_json"]
        meta_s = meta_raw.item() if getattr(meta_raw, "shape", ()) == () else str(meta_raw)
        return {
            "version": str(z["version"].item()) if z["version"].shape == () else str(z["version"]),
            "n_frames": int(z["n_frames"]),
            "height": int(z["height"]),
            "width": int(z["width"]),
            "meta": json.loads(str(meta_s)),
            "final_ptr": np.asarray(z["final_ptr"]),
            "final": SparseMask(z["final_y"], z["final_x"], z["final_val"]),
            "history_ptr": np.asarray(z["history_ptr"]),
            "history": SparseMask(z["history_y"], z["history_x"], z["history_val"]),
            "history_frame": np.asarray(z["history_frame"]),
            "history_i": np.asarray(z["history_i"]),
            "history_kept": np.asarray(z["history_kept"]),
            "history_alpha": np.asarray(z["history_alpha"]),
            "peak_ptr": np.asarray(z["peak_ptr"]),
            "peaks": SparseMask(z["peak_y"], z["peak_x"], z["peak_val"]),
            "peak_frame": np.asarray(z["peak_frame"]),
            "peak_axis": np.asarray(z["peak_axis"]),
            "peak_q": np.asarray(z["peak_q"]),
        }


def artifacts_complete(out_dir: Path, cleaned_tif: Path) -> bool:
    out_dir = Path(out_dir)
    return (
        Path(cleaned_tif).is_file()
        and (out_dir / "per_frame.csv").is_file()
        and (out_dir / "mask_patterns.npz").is_file()
        and (out_dir / "mask_recipe.json").is_file()
        and (out_dir / "families.json").is_file()
    )


KEEPALIVE_SEC = 20.0
IO_RETRY_SLEEP = 20.0
IO_RETRIES = 3
CONSECUTIVE_IO_STOP = 3
DRIVE_WAIT_SEC = 180.0
USB_POWER_HINT = (
    "USB drives sleep under Windows and drop mid-stack. Mitigation in this process: "
    "touch source+dest every 20s, retry I/O, stop if a drive stays gone. Also set "
    "Power Options -> USB selective suspend = Off, and disk timeout = Never "
    "(powercfg /change disk-timeout-ac 0)."
)


def is_drive_io_error(
    exc: BaseException,
    *,
    source_root: Path | None = None,
    dest_root: Path | None = None,
) -> bool:
    """Permission / USB-write-zero / vanished-drive failures, not algorithm bugs.

    A missing TIFF while both roots are still reachable is not treated as a
    drive drop (so one absent file does not abort the whole batch).
    """
    msg = str(exc).lower()
    if "requested and" in msg and "written" in msg:
        return True
    if isinstance(exc, FileNotFoundError):
        src_ok = source_root is None or path_reachable(source_root)
        dst_ok = dest_root is None or path_reachable(dest_root)
        return not (src_ok and dst_ok)
    if isinstance(exc, (OSError, PermissionError, TimeoutError)):
        return True
    return False


def path_reachable(path: Path | None) -> bool:
    if path is None:
        return True
    try:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                next(p.iterdir(), None)
            elif p.is_file():
                with open(p, "rb") as fh:
                    fh.read(1)
            return True
        return False
    except OSError:
        return False


def poke_path(path: Path | None, *, allow_write: bool = False) -> None:
    """Best-effort touch so USB enclosures do not spin down. Never writes unless allowed."""
    if path is None:
        return
    try:
        p = Path(path)
        if p.is_dir():
            next(p.iterdir(), None)
            if allow_write:
                beat_dir = p / ".defringe_v4_runs"
                try:
                    beat_dir.mkdir(parents=True, exist_ok=True)
                    (beat_dir / "keepalive.txt").write_text("ok", encoding="ascii")
                except OSError:
                    pass
        elif p.is_file():
            with open(p, "rb") as fh:
                fh.read(1)
        else:
            p.exists()
    except OSError:
        pass


def wait_until_reachable(path: Path | None, *, timeout: float, interval: float = 10.0) -> bool:
    if path is None:
        return True
    import time

    deadline = time.monotonic() + float(timeout)
    while True:
        if path_reachable(path):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


class DriveKeepAlive:
    """Background listdir/read on source and dest so USB HDDs stay awake.

    Source paths are read-only (listdir / exists). Dest may get a keepalive file
    under ``.defringe_v4_runs/``.
    """

    def __init__(
        self,
        *,
        read_paths: list[Path | None] | None = None,
        write_paths: list[Path | None] | None = None,
        interval: float = KEEPALIVE_SEC,
    ):
        self.read_paths = [Path(p) for p in (read_paths or []) if p is not None]
        self.write_paths = [Path(p) for p in (write_paths or []) if p is not None]
        self.interval = float(interval)
        self._stop = None
        self._thread = None

    def __enter__(self) -> DriveKeepAlive:
        if not self.read_paths and not self.write_paths:
            return self
        import threading

        self._stop = threading.Event()

        def _run() -> None:
            while self._stop is not None and not self._stop.wait(self.interval):
                for p in self.read_paths:
                    poke_path(p, allow_write=False)
                for p in self.write_paths:
                    poke_path(p, allow_write=True)

        self._thread = threading.Thread(target=_run, name="defringe-keepalive", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)


def _fmt_dur(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class BatchProgress:
    """Console + progress.txt / progress.json so a long dest run can be watched."""

    def __init__(
        self,
        *,
        n_jobs: int,
        paths: list[Path | None],
    ):
        import time

        self.n_jobs = int(n_jobs)
        self.paths = [Path(p) for p in paths if p is not None]
        self.t0 = time.monotonic()
        self.job_t0 = self.t0
        self.job_i = 0
        self.job_label = ""
        self.channel = ""
        self.frame_i = 0
        self.n_frames = 0
        self.ok = 0
        self.skipped = 0
        self.error = 0
        self.note = ""
        self.status = "starting"
        self.job_seconds: list[float] = []

    def begin_job(self, i: int, label: str, *, channel: str = "") -> None:
        import time

        self.job_i = int(i)
        self.job_label = str(label)
        self.channel = channel
        self.frame_i = 0
        self.n_frames = 0
        self.note = ""
        self.status = "running"
        self.job_t0 = time.monotonic()
        self.write()

    def set_n_frames(self, n: int) -> None:
        self.n_frames = int(n)
        self.write()

    def tick_frame(self, fi: int, n: int, *, extra: str = "") -> None:
        self.frame_i = int(fi)
        self.n_frames = int(n)
        self.note = extra
        line = (
            f"  frames {fi}/{n}"
            + (f"  {extra}" if extra else "")
            + f"  |  {self._summary_line()}"
        )
        print(line, flush=True)
        self.write()

    def end_job(self, status: str) -> None:
        import time

        self.status = str(status)
        dt = time.monotonic() - self.job_t0
        if status in ("ok", "skipped"):
            self.job_seconds.append(dt)
        if status == "ok":
            self.ok += 1
        elif status == "skipped":
            self.skipped += 1
        else:
            self.error += 1
        self.write()

    def _elapsed(self) -> float:
        import time

        return time.monotonic() - self.t0

    def _eta(self) -> str:
        done = self.ok + self.skipped
        remaining_jobs = max(0, self.n_jobs - self.job_i)
        parts = []
        job_elapsed = 0.0
        import time

        job_elapsed = time.monotonic() - self.job_t0
        if self.n_frames > 0 and self.frame_i > 0 and job_elapsed > 1:
            fps = self.frame_i / job_elapsed
            left_fr = max(0, self.n_frames - self.frame_i)
            parts.append(f"this stack ~{_fmt_dur(left_fr / max(fps, 1e-9))}")
        if self.job_seconds:
            mean_j = sum(self.job_seconds) / len(self.job_seconds)
            left = self.n_jobs - done
            parts.append(f"batch ~{_fmt_dur(mean_j * left)}")
        elif self.n_frames > 0 and self.frame_i > 0 and job_elapsed > 1:
            dur_this = job_elapsed / (self.frame_i / max(self.n_frames, 1))
            left = max(0, self.n_jobs - self.job_i + 1)
            parts.append(f"batch ~{_fmt_dur(dur_this * left)} (assume similar stacks)")
        _ = remaining_jobs
        return "  ".join(parts) if parts else "ETA n/a yet"

    def _summary_line(self) -> str:
        return (
            f"stack {self.job_i}/{self.n_jobs}  "
            f"ok={self.ok} skip={self.skipped} err={self.error}  "
            f"elapsed {_fmt_dur(self._elapsed())}  {self._eta()}"
        )

    def text(self) -> str:
        frac = ""
        if self.n_frames:
            frac = f"{self.frame_i}/{self.n_frames} ({100 * self.frame_i / max(self.n_frames, 1):.1f}%)"
        else:
            frac = "—"
        return "\n".join(
            [
                "v4 dest batch progress",
                f"updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
                f"stack: {self.job_i}/{self.n_jobs}  {self.status}",
                f"ok={self.ok}  skipped={self.skipped}  error={self.error}",
                f"elapsed: {_fmt_dur(self._elapsed())}",
                f"ETA: {self._eta()}",
                f"channel: {self.channel}",
                f"current: {self.job_label}",
                f"frames: {frac}  {self.note}",
                "",
            ]
        )

    def payload(self) -> dict[str, Any]:
        return {
            "job_i": self.job_i,
            "n_jobs": self.n_jobs,
            "status": self.status,
            "ok": self.ok,
            "skipped": self.skipped,
            "error": self.error,
            "elapsed_s": round(self._elapsed(), 1),
            "eta": self._eta(),
            "channel": self.channel,
            "current": self.job_label,
            "frame_i": self.frame_i,
            "n_frames": self.n_frames,
            "note": self.note,
        }

    def write(self) -> None:
        text = self.text()
        blob = json.dumps(self.payload(), indent=2)
        for path in self.paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                path.with_suffix(".json").write_text(blob, encoding="utf-8")
            except OSError:
                continue


