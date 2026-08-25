"""Parse ThorImage Experiment.xml for microscope / scan fingerprints."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


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


def parse_experiment_xml(path: Path) -> dict[str, Any]:
    """Return computer name, LSM/PMT fingerprint, and raw attribute bags."""
    tree = ET.parse(path)
    root = tree.getroot()

    computer_el = root.find(".//Computer")
    lsm_el = root.find(".//LSM")
    pmt_el = root.find(".//PMT")
    mag_el = root.find(".//Magnification")

    computer = "UNKNOWN"
    if computer_el is not None and computer_el.attrib.get("name"):
        computer = computer_el.attrib["name"].strip() or "UNKNOWN"

    fingerprint = {
        "frameRate": _attr_float(lsm_el, "frameRate"),
        "pixelX": _attr_int(lsm_el, "pixelX"),
        "pixelY": _attr_int(lsm_el, "pixelY"),
        "fieldSize": _attr_float(lsm_el, "fieldSize"),
        "pixelSizeUM": _attr_float(lsm_el, "pixelSizeUM"),
        "flybackCycles": _attr_int(lsm_el, "flybackCycles"),
        "dwellTime": _attr_float(lsm_el, "dwellTime"),
        "mag": _attr_float(mag_el, "mag"),
        "gainA": _attr_float(pmt_el, "gainA"),
        "gainB": _attr_float(pmt_el, "gainB"),
    }

    return {
        "computer": computer,
        "fingerprint": fingerprint,
        "xml_path": str(path),
    }


def fingerprint_compatible(
    prior_fp: dict[str, Any] | None,
    current_fp: dict[str, Any],
    *,
    frame_rate_tol: float = 0.5,
    field_size_tol: float = 5.0,
    pixel_size_tol: float = 0.05,
) -> bool:
    """True if scan geometry is close enough that a pixel-q prior remains useful."""
    if not prior_fp:
        return False

    def close(a, b, tol) -> bool:
        if a is None or b is None:
            return True  # missing → don't block
        return abs(float(a) - float(b)) <= tol

    if prior_fp.get("pixelX") not in (None, current_fp.get("pixelX")):
        if prior_fp.get("pixelX") != current_fp.get("pixelX"):
            return False
    if prior_fp.get("pixelY") not in (None, current_fp.get("pixelY")):
        if prior_fp.get("pixelY") != current_fp.get("pixelY"):
            return False

    return (
        close(prior_fp.get("frameRate"), current_fp.get("frameRate"), frame_rate_tol)
        and close(prior_fp.get("fieldSize"), current_fp.get("fieldSize"), field_size_tol)
        and close(prior_fp.get("pixelSizeUM"), current_fp.get("pixelSizeUM"), pixel_size_tol)
    )


def sanitize_computer_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    return safe or "UNKNOWN"
