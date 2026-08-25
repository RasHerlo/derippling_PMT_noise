"""Read ThorImage Experiment.xml for dark-current control recordings.

Wider than ``batch_defringe.experiment_xml``: dark-current controls need the
acquisition levers that could plausibly change the fringe (scan direction,
two-way alignment, averaging, Pockels level, PMT gain), not just the pixel-q
fingerprint used for prior compatibility.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STACK_NAMES = {"ChanA_stk.tif": "ChanA", "ChanB_stk.tif": "ChanB"}

# ThorImage records only the acquisition computer, not the lab name for the rig.
SCOPE_ALIASES = {"THORLABS_30_016": "Shinano"}

# Trial folder names carry the real Pockels setting; the XML <Pockels start="...">
# does not track it (confirmed 2026-08-25: PC050/PC150/PC250 all record 20.5 or 0).
_PC_LABEL = re.compile(r"^PC(\d+)", re.IGNORECASE)


def scope_alias(computer: str) -> str:
    return SCOPE_ALIASES.get(computer, computer)


def pockels_from_label(label: str) -> int | None:
    """Nominal Pockels setting taken from the trial folder name."""
    m = _PC_LABEL.match(label.strip())
    return int(m.group(1)) if m else None


def _num(elem: ET.Element | None, key: str) -> float | None:
    if elem is None or key not in elem.attrib:
        return None
    try:
        return float(elem.attrib[key])
    except ValueError:
        return None


def _int(elem: ET.Element | None, key: str) -> int | None:
    v = _num(elem, key)
    return None if v is None else int(v)


def _str(elem: ET.Element | None, key: str) -> str | None:
    if elem is None:
        return None
    val = elem.attrib.get(key)
    return val.strip() if val else None


@dataclass
class DarkRecording:
    """One dark-current control recording."""

    xml_path: Path
    trial_dir: Path
    data_dir: Path | None
    stacks: dict[str, Path] = field(default_factory=dict)

    name: str | None = None
    date_utc: str | None = None
    computer: str = "UNKNOWN"

    # Acquisition levers that could change the fringe.
    scan: dict[str, Any] = field(default_factory=dict)
    pmt: dict[str, Any] = field(default_factory=dict)
    pockels: dict[str, Any] = field(default_factory=dict)
    status: str | None = None

    @property
    def label(self) -> str:
        return self.trial_dir.name

    @property
    def scope(self) -> str:
        """Familiar rig name where known, else the raw computer tag."""
        return scope_alias(self.computer)

    @property
    def pockels_setting(self) -> int | None:
        """Nominal Pockels level from the folder name (authoritative over XML)."""
        return pockels_from_label(self.label)

    @property
    def aborted(self) -> bool:
        """No assembled stacks means the acquisition did not complete."""
        return not self.stacks

    @property
    def config_key(self) -> str:
        """Identity of the optical/scan configuration a calibration is valid for.

        Deliberately excludes PMT gain and Pockels level: those are expected to
        change fringe amplitude, not its spatial frequency.
        """
        s = self.scan
        return (
            f"{self.computer}"
            f"|{s.get('pixelX')}x{s.get('pixelY')}"
            f"|fr{s.get('frameRate')}"
            f"|fs{s.get('fieldSize')}"
            f"|px{s.get('pixelSizeUM')}"
            f"|scanMode{s.get('scanMode')}"
            f"|avg{s.get('averageMode')}x{s.get('averageNum')}"
        )


def parse_dark_xml(xml_path: Path) -> DarkRecording:
    root = ET.parse(xml_path).getroot()

    name_el = root.find(".//Name")
    date_el = root.find(".//Date")
    lsm = root.find(".//LSM")
    pmt = root.find(".//PMT")
    mag = root.find(".//Magnification")
    timelapse = root.find(".//Timelapse")
    streaming = root.find(".//Streaming")
    lightpath = root.find(".//LightPath")
    status_el = root.find(".//ExperimentStatus")

    # First Pockels block with type != 0 is the active modulator.
    pockels_el = None
    for cand in root.findall(".//Pockels"):
        if _int(cand, "type"):
            pockels_el = cand
            break

    date_utc = None
    if date_el is not None and "uTime" in date_el.attrib:
        try:
            date_utc = datetime.fromtimestamp(
                int(float(date_el.attrib["uTime"])), tz=timezone.utc
            ).isoformat()
        except ValueError:
            date_utc = _str(date_el, "date")
    elif date_el is not None:
        date_utc = _str(date_el, "date")

    rec = DarkRecording(
        xml_path=xml_path,
        trial_dir=xml_path.parent,
        data_dir=None,
        name=_str(name_el, "name"),
        date_utc=date_utc,
        computer=_str(root.find(".//Computer"), "name") or "UNKNOWN",
        status=_str(status_el, "value"),
    )

    rec.scan = {
        "scanMode": _int(lsm, "scanMode"),
        "areaMode": _int(lsm, "areaMode"),
        "pixelX": _int(lsm, "pixelX"),
        "pixelY": _int(lsm, "pixelY"),
        "fieldSize": _num(lsm, "fieldSize"),
        "pixelSizeUM": _num(lsm, "pixelSizeUM"),
        "frameRate": _num(lsm, "frameRate"),
        "dwellTime": _num(lsm, "dwellTime"),
        "flybackCycles": _int(lsm, "flybackCycles"),
        "flybackLines": _int(streaming, "flybackLines"),
        "twoWayAlignment": _int(lsm, "twoWayAlignment"),
        "averageMode": _int(lsm, "averageMode"),
        "averageNum": _int(lsm, "averageNum"),
        "inputRange1": _int(lsm, "inputRange1"),
        "inputRange2": _int(lsm, "inputRange2"),
        "mag": _num(mag, "mag"),
        "magName": _str(mag, "name"),
        "timepoints": _int(timelapse, "timepoints"),
        "frames": _int(streaming, "frames"),
        "galvoGalvo": _int(lightpath, "GalvoGalvo"),
        "galvoResonance": _int(lightpath, "GalvoResonance"),
    }
    rec.pmt = {
        "enableA": _int(pmt, "enableA"),
        "gainA": _num(pmt, "gainA"),
        "enableB": _int(pmt, "enableB"),
        "gainB": _num(pmt, "gainB"),
    }
    rec.pockels = {
        "type": _int(pockels_el, "type"),
        "start": _num(pockels_el, "start"),
        "stop": _num(pockels_el, "stop"),
        "minV": _num(pockels_el, "pockelsMinV"),
        "maxV": _num(pockels_el, "pockelsMaxV"),
        "blankPercentage": _num(pockels_el, "pockelsBlankPercentage"),
        "rampPath": _str(pockels_el, "path"),
    }
    return rec


def _attach_stacks(rec: DarkRecording) -> DarkRecording:
    """Find assembled Chan*_stk.tif under the trial's DATA folder, if built."""
    for data_dir in sorted(rec.trial_dir.rglob("DATA")):
        if not data_dir.is_dir():
            continue
        found = {}
        for tif in sorted(data_dir.rglob("*.tif")):
            chan = STACK_NAMES.get(tif.name)
            if chan and chan not in found:
                found[chan] = tif
        if found:
            rec.data_dir = data_dir
            rec.stacks = found
            break
    return rec


def discover_dark_recordings(root: Path) -> list[DarkRecording]:
    """Find every dark-current trial (Experiment.xml) under ``root``."""
    root = Path(root).resolve()
    recs: list[DarkRecording] = []
    for xml_path in sorted(root.rglob("Experiment.xml")):
        try:
            rec = parse_dark_xml(xml_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] bad Experiment.xml ({xml_path}): {exc}")
            continue
        recs.append(_attach_stacks(rec))
    return recs


def recording_to_json(rec: DarkRecording) -> dict[str, Any]:
    return {
        "label": rec.label,
        "name": rec.name,
        "date_utc": rec.date_utc,
        "computer": rec.computer,
        "scope": rec.scope,
        "status": rec.status,
        "aborted": rec.aborted,
        "config_key": rec.config_key,
        "pockels_setting": rec.pockels_setting,
        "scan": rec.scan,
        "pmt": rec.pmt,
        "pockels_xml": rec.pockels,
        "xml_path": str(rec.xml_path),
        "stacks": {k: str(v) for k, v in rec.stacks.items()},
    }
