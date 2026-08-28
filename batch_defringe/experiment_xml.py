"""Parse ThorImage Experiment.xml for microscope / scan fingerprints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _attr_float(elem: ET.Element | None, key: str, default: float | None = None) -> float | None:
    if elem is None or key not in elem.attrib:
        return default
    try:
        return float(elem.attrib[key])
    except ValueError:
        return default


def _attr_int(elem: ET.Element | None, key: str, default: int | None = None) -> int | None:
    if elem is None or key not in elem.attrib:
        return default
    try:
        return int(float(elem.attrib[key]))
    except ValueError:
        return default


def _attr_str(elem: ET.Element | None, key: str) -> str | None:
    if elem is None or key not in elem.attrib:
        return None
    val = elem.attrib.get(key)
    return val.strip() if val else None


def averaging_n(fp: dict[str, Any] | None) -> int:
    """Scanner frames per saved TIFF frame. ``averageMode=0`` (or missing) → 1."""
    if not fp:
        return 1
    mode = fp.get("averageMode")
    n = fp.get("averageNum")
    try:
        if mode is None or int(mode) == 0:
            return 1
    except (TypeError, ValueError):
        return 1
    try:
        n_int = int(n) if n is not None else 1
    except (TypeError, ValueError):
        return 1
    return max(1, n_int)


def effective_frame_rate(fp: dict[str, Any] | None) -> float | None:
    """Saved-stack fps ≈ listed XML rate / averaging N."""
    if not fp or fp.get("frameRate") is None:
        return None
    return float(fp["frameRate"]) / float(averaging_n(fp))


def finalize_fingerprint(fp: dict[str, Any]) -> dict[str, Any]:
    """Fill derived raster fields. Optical mag/µm stay in the dict but are not match keys."""
    out = dict(fp)
    out["averagingN"] = averaging_n(out)
    eff = effective_frame_rate(out)
    out["effectiveFrameRate"] = eff
    return out


def parse_experiment_xml(path: Path) -> dict[str, Any]:
    """Return computer name, LSM/PMT fingerprint, and raw attribute bags."""
    tree = ET.parse(path)
    root = tree.getroot()

    computer_el = root.find(".//Computer")
    lsm_el = root.find(".//LSM")
    pmt_el = root.find(".//PMT")
    mag_el = root.find(".//Magnification")
    date_el = root.find(".//Date")
    streaming_el = root.find(".//Streaming")

    computer = "UNKNOWN"
    if computer_el is not None and computer_el.attrib.get("name"):
        computer = computer_el.attrib["name"].strip() or "UNKNOWN"

    date_utc = None
    if date_el is not None and "uTime" in date_el.attrib:
        try:
            date_utc = datetime.fromtimestamp(
                int(float(date_el.attrib["uTime"])), tz=timezone.utc
            ).isoformat()
        except ValueError:
            date_utc = _attr_str(date_el, "date")
    elif date_el is not None:
        date_utc = _attr_str(date_el, "date")

    fingerprint = finalize_fingerprint(
        {
            "frameRate": _attr_float(lsm_el, "frameRate"),
            "pixelX": _attr_int(lsm_el, "pixelX"),
            "pixelY": _attr_int(lsm_el, "pixelY"),
            "fieldSize": _attr_float(lsm_el, "fieldSize"),
            "pixelSizeUM": _attr_float(lsm_el, "pixelSizeUM"),
            "flybackCycles": _attr_int(lsm_el, "flybackCycles"),
            "flybackLines": _attr_int(streaming_el, "flybackLines"),
            "dwellTime": _attr_float(lsm_el, "dwellTime"),
            "scanMode": _attr_int(lsm_el, "scanMode"),
            "twoWayAlignment": _attr_int(lsm_el, "twoWayAlignment"),
            "averageMode": _attr_int(lsm_el, "averageMode"),
            "averageNum": _attr_int(lsm_el, "averageNum"),
            "areaMode": _attr_int(lsm_el, "areaMode"),
            "mag": _attr_float(mag_el, "mag"),
            "gainA": _attr_float(pmt_el, "gainA"),
            "gainB": _attr_float(pmt_el, "gainB"),
        }
    )

    return {
        "computer": computer,
        "fingerprint": fingerprint,
        "date_utc": date_utc,
        "xml_path": str(path),
    }


def fingerprint_compatible(
    prior_fp: dict[str, Any] | None,
    current_fp: dict[str, Any],
    *,
    frame_rate_tol: float = 0.5,
    field_size_tol: float = 5.0,
    pixel_size_tol: float = 0.05,
    alignment_tol: float = 3.0,
) -> bool:
    """True if *raster* geometry is close enough that a pixel-q prior remains useful.

    Match keys: pixelX/Y, fieldSize, listed frameRate, scanMode, averaging,
    flybackCycles, twoWayAlignment (when scanMode looks two-way / faster).

    ``mag`` / ``pixelSizeUM`` are stored but ignored (PMT fringe does not follow
    the objective). ``pixel_size_tol`` is unused; kept so old callers do not break.
    """
    del pixel_size_tol
    if not prior_fp:
        return False

    def close(a, b, tol) -> bool:
        if a is None or b is None:
            return True
        return abs(float(a) - float(b)) <= tol

    if prior_fp.get("pixelX") not in (None, current_fp.get("pixelX")):
        if prior_fp.get("pixelX") != current_fp.get("pixelX"):
            return False
    if prior_fp.get("pixelY") not in (None, current_fp.get("pixelY")):
        if prior_fp.get("pixelY") != current_fp.get("pixelY"):
            return False

    prior_scan = prior_fp.get("scanMode")
    cur_scan = current_fp.get("scanMode")
    if prior_scan is not None and cur_scan is not None:
        if int(prior_scan) != int(cur_scan):
            return False

    prior_n = averaging_n(prior_fp)
    cur_n = averaging_n(current_fp)
    if prior_fp.get("averageMode") is not None and current_fp.get("averageMode") is not None:
        if prior_n != cur_n:
            return False

    if prior_fp.get("flybackCycles") is not None and current_fp.get("flybackCycles") is not None:
        if int(prior_fp["flybackCycles"]) != int(current_fp["flybackCycles"]):
            return False

    # twoWayAlignment matters for two-way packing. On this rig scanMode 0 is the
    # faster (~30 Hz) two-way setting; scanMode 1 is the slower (~15 Hz) one-way.
    two_way = False
    if prior_scan is not None:
        two_way = int(prior_scan) == 0
    elif cur_scan is not None:
        two_way = int(cur_scan) == 0
    if two_way:
        if not close(
            prior_fp.get("twoWayAlignment"),
            current_fp.get("twoWayAlignment"),
            alignment_tol,
        ):
            return False

    return close(prior_fp.get("frameRate"), current_fp.get("frameRate"), frame_rate_tol) and close(
        prior_fp.get("fieldSize"), current_fp.get("fieldSize"), field_size_tol
    )


def sanitize_computer_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    return safe or "UNKNOWN"
