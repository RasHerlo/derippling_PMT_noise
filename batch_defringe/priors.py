"""Microscope × channel prior cache (JSON on disk under the batch root)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .experiment_xml import sanitize_computer_name


def cache_root(batch_root: Path) -> Path:
    return batch_root / ".defringe_cache"


def prior_path(batch_root: Path, computer: str, channel: str) -> Path:
    return (
        cache_root(batch_root)
        / "priors"
        / sanitize_computer_name(computer)
        / f"{channel}.json"
    )


def _family_to_json(fam: dict) -> dict[str, Any]:
    out = {
        "q": float(fam["q"]),
        "hi": None if fam.get("hi") is None else float(fam["hi"]),
        "paired": bool(fam.get("paired", True)),
        "row_score": float(fam.get("row_score", 0.0)),
        "pair_score": None
        if fam.get("pair_score") is None
        else float(fam["pair_score"]),
        "fx_ranges": fam.get("fx_ranges", []),
    }
    if "x_weight" in fam and fam["x_weight"] is not None:
        out["x_weight"] = np.asarray(fam["x_weight"], dtype=float).tolist()
    return out


def _family_from_json(obj: dict) -> dict:
    fam = {
        "q": float(obj["q"]),
        "hi": None if obj.get("hi") is None else float(obj["hi"]),
        "paired": bool(obj.get("paired", True)),
        "row_score": float(obj.get("row_score", 0.0)),
        "pair_score": None
        if obj.get("pair_score") is None
        else float(obj["pair_score"]),
        "fx_ranges": obj.get("fx_ranges", []),
    }
    if "x_weight" in obj and obj["x_weight"] is not None:
        fam["x_weight"] = np.asarray(obj["x_weight"], dtype=float)
    return fam


def load_prior(batch_root: Path, computer: str, channel: str) -> dict | None:
    path = prior_path(batch_root, computer, channel)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["families"] = [_family_from_json(f) for f in data.get("families", [])]
    return data


def save_prior(
    batch_root: Path,
    computer: str,
    channel: str,
    *,
    families: list[dict],
    fingerprint: dict,
    source_tif: Path | str,
    notes: str = "",
) -> Path:
    path = prior_path(batch_root, computer, channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v2.2_pack_D",
        "computer": computer,
        "channel": channel,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source_tif": str(source_tif),
        "fingerprint": fingerprint,
        "notes": notes,
        "families": [_family_to_json(f) for f in families],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
