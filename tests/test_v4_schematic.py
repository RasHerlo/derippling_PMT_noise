"""v4 schematic PDF writes."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from batch_defringe.v4_schematic import write_v4_schematic_pdf  # noqa: E402


def test_v4_schematic_pdf(tmp_path=None):
    dest = Path(tmp_path) / "v4_pipeline_schematic.pdf" if tmp_path is not None else _REPO / "v4_pipeline_schematic.pdf"
    out = write_v4_schematic_pdf(dest)
    assert out.is_file()
    assert out.stat().st_size > 1000
    return out


if __name__ == "__main__":
    test_v4_schematic_pdf()
    print("ok")
