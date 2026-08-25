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

    @property
    def rel_id(self) -> str:
        return str(self.tif_path)


def _find_experiment_xml(data_dir: Path, root: Path) -> Path | None:
    """Prefer Experiment.xml in or beside DATA; else walk parents up to root."""
    root = root.resolve()
    folder = data_dir.resolve()
    while True:
        for name in ("Experiment.xml", "experiment.xml"):
            cand = folder / name
            if cand.is_file():
                return cand
        if folder == root:
            break
        parent = folder.parent
        if parent == folder:
            break
        try:
            parent.relative_to(root)
        except ValueError:
            break
        folder = parent
    return None


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

        if xml_path is not None:
            try:
                meta = parse_experiment_xml(xml_path)
                computer = meta["computer"]
                fingerprint = meta["fingerprint"]
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
                )
            )

    return jobs


# Back-compat alias used by older call sites / docs.
def discover_trials(root: Path) -> list[StackJob]:
    return discover_stacks(root)
