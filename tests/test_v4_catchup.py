"""Catch-up list: permission-denied dests stay open until artifacts are complete."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from batch_defringe.v4_catchup import build_catchup, render_md  # noqa: E402
from batch_defringe.v4_export import V4_DIR  # noqa: E402


def test_outstanding_until_complete(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    tif = src / "trial" / "DATA" / "ChanA" / "ChanA_stk.tif"
    tif.parent.mkdir(parents=True)
    tif.write_bytes(b"x")
    out = dst / "trial" / "DATA" / "ChanA" / V4_DIR
    out.mkdir(parents=True)
    run = dst / ".defringe_v4_runs" / "20260926T000000Z"
    run.mkdir(parents=True)
    (run / "summary.jsonl").write_text(
        json.dumps(
            {
                "i": 12,
                "tif": str(tif),
                "channel": "ChanA",
                "status": "error",
                "message": "[Errno 13] Permission denied",
                "out_dir": str(out),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = build_catchup(source_root=src, dest_root=dst)
    assert len(payload["outstanding_errors"]) == 1
    assert payload["outstanding_errors"][0]["transient_io"] is True
    assert payload["later_succeeded"] == []
    text = render_md(payload)
    assert "Still open (retry these)" in text
    assert "Permission denied" in text

    (out / "ChanA_stk_defringed_v4.tif").write_bytes(b"ok")
    (out / "per_frame.csv").write_text("frame\n", encoding="utf-8")
    (out / "mask_patterns.npz").write_bytes(b"z")
    (out / "mask_recipe.json").write_text("{}", encoding="utf-8")
    (out / "families.json").write_text("[]", encoding="utf-8")
    payload2 = build_catchup(source_root=src, dest_root=dst)
    assert payload2["outstanding_errors"] == []
    assert len(payload2["later_succeeded"]) == 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_outstanding_until_complete(Path(td))
    print("ok")
