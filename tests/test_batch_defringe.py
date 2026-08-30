"""Unit tests for raster fingerprint, library branches, recurrent seed, q lock."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "reference" / "gpt"))

from batch_defringe.experiment_xml import (  # noqa: E402
    averaging_n,
    effective_frame_rate,
    finalize_fingerprint,
    fingerprint_compatible,
)
from batch_defringe.library import (  # noqa: E402
    catalog_status,
    classify_record,
    format_catalog_line,
    lookup_prior,
    make_record,
    same_calendar_day,
)
from batch_defringe.seed import cluster_recurrent_families  # noqa: E402
from pmt_fringe_raw_adaptive import search_q  # noqa: E402


def _fp(**kwargs):
    base = dict(
        frameRate=15.136,
        pixelX=512,
        pixelY=512,
        fieldSize=178.0,
        pixelSizeUM=1.623,
        flybackCycles=12,
        scanMode=1,
        twoWayAlignment=-12,
        averageMode=1,
        averageNum=6,
        mag=16.0,
    )
    base.update(kwargs)
    return finalize_fingerprint(base)


def test_averaging_and_effective_rate():
    assert averaging_n({"averageMode": 0, "averageNum": 2}) == 1
    assert averaging_n({"averageMode": 1, "averageNum": 6}) == 6
    fp = _fp()
    assert abs(fp["effectiveFrameRate"] - 15.136 / 6) < 1e-9
    assert abs(effective_frame_rate(fp) - 15.136 / 6) < 1e-9


def test_mag_and_pixel_size_ignored():
    a = _fp(mag=16.0, pixelSizeUM=1.623)
    b = _fp(mag=27.77, pixelSizeUM=0.935)
    assert fingerprint_compatible(a, b)


def test_scan_mode_mismatch():
    a = _fp(scanMode=1)
    b = _fp(scanMode=0, frameRate=29.595, averageMode=0, averageNum=2, twoWayAlignment=0)
    assert not fingerprint_compatible(a, b)


def test_two_way_alignment_matters_for_scanmode_0():
    a = _fp(scanMode=0, frameRate=29.595, averageMode=0, averageNum=2, twoWayAlignment=0)
    b = _fp(scanMode=0, frameRate=29.595, averageMode=0, averageNum=2, twoWayAlignment=-17)
    assert not fingerprint_compatible(a, b)
    c = _fp(scanMode=0, frameRate=29.595, averageMode=0, averageNum=2, twoWayAlignment=-1)
    assert fingerprint_compatible(a, c)


def test_averaging_mismatch():
    a = _fp(averageNum=6)
    b = _fp(averageNum=1)
    assert not fingerprint_compatible(a, b)


def test_same_day_and_branches():
    assert same_calendar_day("2024-09-16T17:30:16+00:00", "2024-09-16T08:00:00+00:00")
    assert not same_calendar_day("2024-09-16T17:30:16+00:00", "2024-09-17T17:30:16+00:00")

    fp = _fp()
    rec_dc = make_record(
        source="darkcurrent",
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        families=[{"q": 6, "hi": 250, "paired": True, "row_score": 10}],
        date_utc="2024-09-16T12:00:00+00:00",
        origin="dc",
    )
    rec_live = make_record(
        source="live_clean",
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        families=[{"q": 14, "hi": 242, "paired": True, "row_score": 12}],
        date_utc="2024-09-10T12:00:00+00:00",
        origin="live",
    )
    assert (
        classify_record(
            rec_dc,
            computer="THORLABS_30_016",
            channel="ChanA",
            fingerprint=fp,
            recording_date="2024-09-16T19:00:00+00:00",
        )
        == "A"
    )
    assert (
        classify_record(
            rec_dc,
            computer="THORLABS_30_016",
            channel="ChanA",
            fingerprint=fp,
            recording_date="2024-09-20T19:00:00+00:00",
        )
        == "B"
    )
    assert (
        classify_record(
            rec_live,
            computer="THORLABS_30_016",
            channel="ChanA",
            fingerprint=fp,
            recording_date="2024-09-16T19:00:00+00:00",
        )
        == "C"
    )

    hit = lookup_prior(
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        recording_date="2024-09-16T19:00:00+00:00",
        catalog={"version": 1, "records": [rec_dc, rec_live]},
    )
    assert hit is not None
    assert hit["branch"] == "A"
    assert hit["families"][0]["q"] == 6
    assert hit["complete"] is True


def test_shutter_is_incomplete_a_and_loses_to_dc():
    fp = _fp()
    rec_dc = make_record(
        source="darkcurrent",
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        families=[{"q": 6, "hi": 250, "paired": True, "row_score": 10}],
        date_utc="2024-09-16T12:00:00+00:00",
        origin="dc",
    )
    rec_shutter = make_record(
        source="in_stack_shutter",
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        families=[{"q": 10, "hi": 246, "paired": True, "row_score": 8}],
        date_utc="2024-09-16T17:00:00+00:00",
        origin="stack",
        frames=[756, 757, 758, 759, 760],
    )
    rec_live = make_record(
        source="live_clean",
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        families=[{"q": 14, "hi": 242, "paired": True, "row_score": 12}],
        date_utc="2024-09-16T18:00:00+00:00",
        origin="live",
    )
    assert rec_shutter["complete"] is False
    assert rec_shutter["frames"] == [756, 757, 758, 759, 760]
    assert (
        classify_record(
            rec_shutter,
            computer="THORLABS_30_016",
            channel="ChanA",
            fingerprint=fp,
            recording_date="2024-09-16T19:00:00+00:00",
        )
        == "A"
    )
    assert (
        classify_record(
            rec_shutter,
            computer="THORLABS_30_016",
            channel="ChanA",
            fingerprint=fp,
            recording_date="2024-09-20T19:00:00+00:00",
        )
        == "B"
    )

    vs_dc = lookup_prior(
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        recording_date="2024-09-16T19:00:00+00:00",
        catalog={"version": 1, "records": [rec_shutter, rec_dc]},
    )
    assert vs_dc is not None
    assert vs_dc["source"] == "darkcurrent"
    assert vs_dc["complete"] is True
    assert vs_dc["families"][0]["q"] == 6

    vs_live = lookup_prior(
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=fp,
        recording_date="2024-09-16T19:00:00+00:00",
        catalog={"version": 1, "records": [rec_shutter, rec_live]},
    )
    assert vs_live is not None
    assert vs_live["source"] == "in_stack_shutter"
    assert vs_live["complete"] is False
    assert vs_live["frames"] == [756, 757, 758, 759, 760]
    assert vs_live["families"][0]["q"] == 10


def test_catalog_line_for_overview():
    assert "none" in format_catalog_line(None)
    assert "none" in format_catalog_line(catalog_status(None))
    rec = make_record(
        source="in_stack_shutter",
        computer="THORLABS_30_016",
        channel="ChanA",
        fingerprint=_fp(),
        families=[{"q": 10, "hi": 246, "paired": True, "row_score": 8}],
        date_utc="2024-09-16T17:00:00+00:00",
        frames=[756, 760],
    )
    hit = {
        "branch": "A",
        "source": "in_stack_shutter",
        "complete": False,
        "frames": [756, 760],
        "families": rec["families"],
        "origin": "stack",
        "date_utc": rec["date_utc"],
    }
    used = format_catalog_line(catalog_status(hit, used=True))
    assert "A in_stack_shutter" in used
    assert "incomplete" in used
    assert "756–760" in used
    assert "used as seed" in used
    considered = format_catalog_line(
        catalog_status(hit, used=False, rejected_qs=[10], supported_qs=[])
    )
    assert "no fx support" in considered


def test_recurrent_cluster_prefers_repeat():
    spec = np.zeros((16, 16))
    fam6 = {"q": 6.0, "hi": 10.0, "paired": True, "row_score": 8.0}
    fam14 = {"q": 14.0, "hi": 2.0, "paired": True, "row_score": 20.0}
    hits = [
        {"start": 0, "stop": 50, "medspec": spec, "families": [fam6]},
        {"start": 50, "stop": 100, "medspec": spec, "families": [fam6]},
        {"start": 100, "stop": 150, "medspec": spec, "families": [fam14]},
    ]
    families, _, info = cluster_recurrent_families(hits, max_families=1)
    assert len(families) == 1
    assert families[0]["q"] == 6.0
    assert families[0]["n_blocks"] == 2
    assert info["chosen"][0]["n_blocks"] == 2


def test_search_q_identity_lock():
    rng = np.random.default_rng(0)
    logamp = rng.normal(size=(64, 64))
    logamp[32 + 14, :] += 5.0
    xvalid = np.ones(64, dtype=bool)
    xvalid[:5] = False
    q, score = search_q(logamp, 6, False, xvalid, 10, forbidden_qs=[14], forbidden_radius=3)
    assert abs(q - 14) >= 3
    assert np.isfinite(score) or q == 6.0


def test_choose_inspection_frames():
    from batch_defringe.readout import choose_inspection_frames

    rows = [
        {"frame": i, "removed_rms": 10.0, "max_gate": 0.1, "family0_q": 81.0}
        for i in range(1200)
    ]
    rows[50]["removed_rms"] = 100.0
    rows[51]["removed_rms"] = 90.0
    rows[3]["removed_rms"] = 0.01
    rows[4]["removed_rms"] = 0.02
    chosen = choose_inspection_frames(rows, n_frames=1200)
    by_role: dict[str, list[int]] = {}
    for c in chosen:
        by_role.setdefault(c["role"], []).append(c["frame"])
    assert by_role["anchor"] == [160, 1061]
    assert by_role["strong"] == [50, 51]
    assert by_role["weak"] == [3, 4]
    assert len(chosen) == 6


if __name__ == "__main__":
    test_averaging_and_effective_rate()
    test_mag_and_pixel_size_ignored()
    test_scan_mode_mismatch()
    test_two_way_alignment_matters_for_scanmode_0()
    test_averaging_mismatch()
    test_same_day_and_branches()
    test_shutter_is_incomplete_a_and_loses_to_dc()
    test_catalog_line_for_overview()
    test_recurrent_cluster_prefers_repeat()
    test_search_q_identity_lock()
    test_choose_inspection_frames()
    print("ok")
