"""Discover ChanA/B_stk.tif under DATA/ folders (optional Experiment.xml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment_xml import parse_experiment_xml


STACK_BASENAMES = {
    "ChanA_stk.tif": "ChanA",
    "ChanB_stk.tif": "ChanB",
}

NO_XML_COMPUTER = "UNKNOWN_NO_XML"


@dataclass
class StackJob:
    """One raw stack to defringe."""

    tif_path: Path
    channel: str
    data_dir: Path
    trial_dir: Path
    xml_path: Path | None
    computer: str
    fingerprint: dict
    missing_xml: bool
    date_utc: str | None = None

    @property
    def rel_id(self) -> str:
        return str(self.tif_path)


def find_experiment_xml(start: Path, stop: Path | None = None, *, max_up: int = 8) -> Path | None:
    """Walk parents from start looking for Experiment.xml."""
    folder = Path(start).resolve()
    stop_res = Path(stop).resolve() if stop is not None else None
    for _ in range(max_up):
        for name in ("Experiment.xml", "experiment.xml"):
            cand = folder / name
            if cand.is_file():
                return cand
        if stop_res is not None and folder == stop_res:
            break
        parent = folder.parent
        if parent == folder:
            break
        folder = parent
    return None


def _find_experiment_xml(data_dir: Path, root: Path) -> Path | None:
    """Prefer Experiment.xml in or beside DATA; else walk parents up to root."""
    return find_experiment_xml(data_dir, stop=root)


def _iter_data_dirs(root: Path):
    """Yield every directory named DATA under root (case-sensitive on Windows usually OK)."""
    for path in sorted(root.rglob("DATA")):
        if path.is_dir():
            yield path


def discover_stacks(root: Path) -> list[StackJob]:
    """
    Find ChanA_stk.tif / ChanB_stk.tif under any DATA/ tree.

    Microscope priors use Experiment.xml <Computer> when present.
    Missing XML → computer=UNKNOWN_NO_XML, missing_xml=True (caller uses fresh seed).
    """
    root = root.resolve()
    jobs: list[StackJob] = []
    seen: set[Path] = set()

    for data_dir in _iter_data_dirs(root):
        xml_path = _find_experiment_xml(data_dir, root)
        computer = NO_XML_COMPUTER
        fingerprint: dict = {}
        missing_xml = xml_path is None
        date_utc = None

        if xml_path is not None:
            try:
                meta = parse_experiment_xml(xml_path)
                computer = meta["computer"]
                fingerprint = meta["fingerprint"]
                date_utc = meta.get("date_utc")
                missing_xml = False
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] bad Experiment.xml ({xml_path}): {exc}")
                missing_xml = True
                computer = NO_XML_COMPUTER
                fingerprint = {}
                xml_path = None

        trial_dir = data_dir.parent
        for tif in sorted(data_dir.rglob("*.tif")):
            channel = STACK_BASENAMES.get(tif.name)
            if channel is None:
                continue
            resolved = tif.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            jobs.append(
                StackJob(
                    tif_path=tif,
                    channel=channel,
                    data_dir=data_dir,
                    trial_dir=trial_dir,
                    xml_path=xml_path,
                    computer=computer,
                    fingerprint=fingerprint,
                    missing_xml=missing_xml,
                    date_utc=date_utc,
                )
            )

    return jobs


def job_for_stack(tif_path: Path, root: Path | None = None) -> StackJob:
    """Build a StackJob for one TIFF (single-stack CLI). Walks up for Experiment.xml."""
    tif_path = Path(tif_path).resolve()
    channel = STACK_BASENAMES.get(tif_path.name)
    if channel is None:
        parent_name = tif_path.parent.name
        channel = parent_name if parent_name in ("ChanA", "ChanB") else "ChanA"
    if tif_path.parent.name in ("ChanA", "ChanB"):
        data_dir = tif_path.parent.parent
    else:
        data_dir = tif_path.parent
    trial_dir = data_dir.parent if data_dir.name.upper() == "DATA" else data_dir
    xml_path = find_experiment_xml(tif_path.parent, stop=root)
    computer = NO_XML_COMPUTER
    fingerprint: dict = {}
    missing_xml = xml_path is None
    date_utc = None
    if xml_path is not None:
        try:
            meta = parse_experiment_xml(xml_path)
            computer = meta["computer"]
            fingerprint = meta["fingerprint"]
            date_utc = meta.get("date_utc")
            missing_xml = False
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] bad Experiment.xml ({xml_path}): {exc}")
            xml_path = None
            missing_xml = True
            computer = NO_XML_COMPUTER
    return StackJob(
        tif_path=tif_path,
        channel=channel,
        data_dir=data_dir,
        trial_dir=trial_dir,
        xml_path=xml_path,
        computer=computer,
        fingerprint=fingerprint,
        missing_xml=missing_xml,
        date_utc=date_utc,
    )


# Back-compat alias used by older call sites / docs.
def discover_trials(root: Path) -> list[StackJob]:
    return discover_stacks(root)
