"""Committed fringe-family library (prior branches A / B / C).

Branch A: DarkCurrent, same calendar day, raster match.
Branch B: older DarkCurrent, raster match.
Branch C: successful live cleans, raster match.

Amplitude is never copied — only q / pair / fx geometry, verified on the live stack.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment_xml import (
    finalize_fingerprint,
    fingerprint_compatible,
    sanitize_computer_name,
)

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = _REPO / "fringe_library" / "catalog.json"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def same_calendar_day(a: str | None, b: str | None) -> bool:
    da, db = _parse_date(a), _parse_date(b)
    if da is None or db is None:
        return False
    if da.tzinfo is None:
        da = da.replace(tzinfo=timezone.utc)
    if db.tzinfo is None:
        db = db.replace(tzinfo=timezone.utc)
    return da.astimezone(timezone.utc).date() == db.astimezone(timezone.utc).date()


def catalog_path() -> Path:
    return DEFAULT_CATALOG


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CATALOG
    if not p.is_file():
        return {"version": 1, "records": []}
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("version", 1)
    data.setdefault("records", [])
    return data


def save_catalog(data: dict[str, Any], path: Path | None = None) -> Path:
    p = Path(path) if path is not None else DEFAULT_CATALOG
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": int(data.get("version", 1)), "records": data.get("records", [])}
    if data.get("note"):
        payload["note"] = data["note"]
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def local_library_path(batch_root: Path) -> Path:
    return Path(batch_root) / ".defringe_cache" / "library.jsonl"


def _record_from_jsonl_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) else None


def load_local_records(batch_root: Path) -> list[dict[str, Any]]:
    path = local_library_path(batch_root)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = _record_from_jsonl_line(line)
            if rec:
                out.append(rec)
    return out


def append_local_record(batch_root: Path, record: dict[str, Any]) -> Path:
    path = local_library_path(batch_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return path


def append_catalog_record(record: dict[str, Any], path: Path | None = None) -> Path:
    data = load_catalog(path)
    data["records"].append(record)
    return save_catalog(data, path)


def classify_record(
    rec: dict[str, Any],
    *,
    computer: str,
    channel: str,
    fingerprint: dict[str, Any],
    recording_date: str | None,
) -> str | None:
    """Return 'A', 'B', 'C', or None if this record should not be used."""
    if rec.get("computer") != computer or rec.get("channel") != channel:
        return None
    if not fingerprint_compatible(rec.get("fingerprint"), fingerprint):
        return None
    source = str(rec.get("source") or "")
    if source == "darkcurrent":
        if same_calendar_day(rec.get("date_utc"), recording_date):
            return "A"
        return "B"
    if source == "live_clean":
        return "C"
    return "C"


_BRANCH_RANK = {"A": 0, "B": 1, "C": 2}


def lookup_prior(
    *,
    computer: str,
    channel: str,
    fingerprint: dict[str, Any],
    recording_date: str | None = None,
    batch_root: Path | None = None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Best library hit: A then B then C; newest date wins within a branch."""
    records: list[dict[str, Any]] = []
    cat = catalog if catalog is not None else load_catalog()
    records.extend(cat.get("records") or [])
    if batch_root is not None:
        records.extend(load_local_records(batch_root))

    best: tuple[int, str, str, dict[str, Any]] | None = None
    for rec in records:
        branch = classify_record(
            rec,
            computer=computer,
            channel=channel,
            fingerprint=fingerprint,
            recording_date=recording_date,
        )
        if branch is None:
            continue
        rank = _BRANCH_RANK[branch]
        date = rec.get("date_utc") or ""
        if best is None or rank < best[0] or (rank == best[0] and date > best[1]):
            best = (rank, date, branch, rec)
    if best is None:
        return None
    _rank, _date, branch, winner = best
    return {
        "branch": branch,
        "record": winner,
        "families": winner.get("families") or [],
        "source": winner.get("source"),
        "date_utc": winner.get("date_utc"),
        "origin": winner.get("origin"),
    }


def make_record(
    *,
    source: str,
    computer: str,
    channel: str,
    fingerprint: dict[str, Any],
    families: list[dict[str, Any]],
    date_utc: str | None = None,
    origin: str = "",
    notes: str = "",
) -> dict[str, Any]:
    slim = []
    for fam in families:
        slim.append(
            {
                "q": float(fam["q"]),
                "hi": None if fam.get("hi") is None else float(fam["hi"]),
                "paired": bool(fam.get("paired", True)),
                "row_score": float(fam.get("row_score", 0.0)),
                "pair_score": None
                if fam.get("pair_score") is None
                else float(fam.get("pair_score")),
                "fx_ranges": fam.get("fx_ranges") or [],
            }
        )
    return {
        "source": source,
        "computer": computer,
        "channel": channel,
        "date_utc": date_utc,
        "fingerprint": fingerprint,
        "families": slim,
        "origin": origin,
        "notes": notes,
        "computer_safe": sanitize_computer_name(computer),
    }


def ingest_darkcurrent(
    *,
    measurements_path: Path | None = None,
    registry_path: Path | None = None,
    trusted_run: str = "20260825T160621Z",
) -> list[dict[str, Any]]:
    """Build library records from the trusted characterize run (geometry only)."""
    meas_path = measurements_path or (_REPO / "darkcurrent" / "measurements.json")
    reg_path = registry_path or (_REPO / "darkcurrent" / "registry.json")
    if not meas_path.is_file() or not reg_path.is_file():
        return []

    measurements = json.loads(meas_path.read_text(encoding="utf-8"))
    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in registry.get("recordings") or [] if "label" in r}

    records: list[dict[str, Any]] = []
    for run in measurements.get("runs") or []:
        if run.get("run") != trusted_run:
            continue
        for ch in run.get("channels") or []:
            label = ch.get("label")
            channel = ch.get("channel")
            fams = (ch.get("families") or {}).get("production_families") or []
            if not label or not channel or not fams:
                continue
            reg = by_label.get(label) or {}
            if reg.get("aborted"):
                continue
            scan = reg.get("scan") or {}
            fp = finalize_fingerprint(
                {
                    "frameRate": scan.get("frameRate"),
                    "pixelX": scan.get("pixelX"),
                    "pixelY": scan.get("pixelY"),
                    "fieldSize": scan.get("fieldSize"),
                    "pixelSizeUM": scan.get("pixelSizeUM"),
                    "flybackCycles": scan.get("flybackCycles"),
                    "flybackLines": scan.get("flybackLines"),
                    "dwellTime": scan.get("dwellTime"),
                    "scanMode": scan.get("scanMode"),
                    "twoWayAlignment": scan.get("twoWayAlignment"),
                    "averageMode": scan.get("averageMode"),
                    "averageNum": scan.get("averageNum"),
                    "areaMode": scan.get("areaMode"),
                    "mag": scan.get("mag"),
                    "gainA": (reg.get("pmt") or {}).get("gainA"),
                    "gainB": (reg.get("pmt") or {}).get("gainB"),
                }
            )
            records.append(
                make_record(
                    source="darkcurrent",
                    computer=reg.get("computer") or "UNKNOWN",
                    channel=channel,
                    fingerprint=fp,
                    families=fams,
                    date_utc=reg.get("date_utc"),
                    origin=f"darkcurrent:{trusted_run}:{label}:{channel}",
                    notes="trusted characterize run; geometry only",
                )
            )
    return records
