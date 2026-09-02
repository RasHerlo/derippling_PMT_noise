"""Shutter quiet-window detect from FOV contrast collapse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from batch_defringe.shutter_detect import detect_shutter_windows, format_shutter_span, low_std_runs  # noqa: E402


def _stats_from_std(std: list[float], mean: float = 900.0) -> list[dict]:
    return [{"frame": i, "mean": mean, "std": float(s)} for i, s in enumerate(std)]


def test_haj_like_plummet_is_756_760():
    rng = np.random.default_rng(0)
    std = list(200 + rng.normal(0, 8, 755))
    std += [30, 29, 28, 27, 26]
    std += list(200 + rng.normal(0, 8, 20))
    det = detect_shutter_windows(_stats_from_std(std, mean=960.0))
    assert det["frames"] == [755, 756, 757, 758, 759]
    assert format_shutter_span(det) == "755–759 (5 frames)"


def test_mean_offset_does_not_fire_a_mean_fraction_cut():
    """Haj Grant: mean stays ~850–980; a 0.55×median(mean) cut would miss."""
    mean_live, mean_shut = 970.0, 851.0
    stats = [{"frame": i, "mean": mean_live, "std": 210.0} for i in range(10)]
    stats += [{"frame": i, "mean": mean_shut, "std": 28.0} for i in range(10, 15)]
    stats += [{"frame": i, "mean": mean_live, "std": 210.0} for i in range(15, 25)]
    med_mean = float(np.median([s["mean"] for s in stats]))
    assert all(s["mean"] > 0.55 * med_mean for s in stats)
    det = detect_shutter_windows(stats)
    assert det["frames"] == [10, 11, 12, 13, 14]


def test_no_shutter_returns_empty():
    stats = [{"frame": i, "mean": 900.0, "std": 180.0 + (i % 5)} for i in range(40)]
    det = detect_shutter_windows(stats)
    assert det["frames"] == []
    assert format_shutter_span(det) == "none"


def test_gradual_dip_without_cliff_is_ignored():
    std = [200.0] * 10 + [160, 120, 90, 70, 55, 50, 50, 50] + [200.0] * 10
    det = detect_shutter_windows(_stats_from_std(std))
    assert det["frames"] == []


def test_low_std_runs_wrapper_matches():
    std = [200.0] * 8 + [25, 24, 23, 22] + [200.0] * 8
    stats = _stats_from_std(std)
    runs = low_std_runs(stats)
    det = detect_shutter_windows(stats)
    assert runs == [det["frames"]]


if __name__ == "__main__":
    test_haj_like_plummet_is_756_760()
    test_mean_offset_does_not_fire_a_mean_fraction_cut()
    test_no_shutter_returns_empty()
    test_gradual_dip_without_cliff_is_ignored()
    test_low_std_runs_wrapper_matches()
    print("ok")
