"""Reconcile dest-batch errors vs complete folders for later retry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .v4_export import artifacts_complete, v4_dest_dir

DEFAULT_SRC = Path(r"E:\Rasmus-Guillermo\ECF1")
DEFAULT_DST = Path(r"F:\CollectedData")


def _iter_summary_rows(runs: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sumf in sorted(runs.glob("20*/summary.jsonl")):
        run = sumf.parent.name
        for line in sumf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("tif") or not rec.get("status"):
                continue
            rec["run"] = run
            rows.append(rec)
    return rows


def _latest_per_tif(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for rec in rows:
        latest[str(rec["tif"])] = rec
    return latest


def _error_kind(message: str) -> str:
    m = str(message).lower()
    if "permission denied" in m:
        return "permission_denied"
    if "no such file" in m or "filenotfound" in m:
        return "missing_source"
    if "written" in m:
        return "usb_write_zero"
    return "other"


def dest_complete(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    tifs = list(out_dir.glob("*_defringed_v4.tif"))
    return bool(tifs and artifacts_complete(out_dir, tifs[0]))


def build_catchup(
    *,
    source_root: Path = DEFAULT_SRC,
    dest_root: Path = DEFAULT_DST,
) -> dict[str, Any]:
    runs = dest_root / ".defringe_v4_runs"
    rows = _iter_summary_rows(runs)
    latest = _latest_per_tif(rows)
    ever_error = {str(r["tif"]) for r in rows if r.get("status") == "error"}
    current_run = max((r["run"] for r in rows), default="")
    current_max_i = max(
        (int(r.get("i") or 0) for r in rows if r.get("run") == current_run),
        default=0,
    )

    outstanding: list[dict[str, Any]] = []
    caught: list[dict[str, Any]] = []
    pending_ahead: list[dict[str, Any]] = []
    for tif, e in sorted(latest.items(), key=lambda kv: int(kv[1].get("i") or 0)):
        out = Path(e.get("out_dir") or v4_dest_dir(Path(tif), source_root, dest_root))
        complete = dest_complete(out)
        kind = _error_kind(str(e.get("message", "")))
        rec = {
            "i": e.get("i"),
            "tif": e["tif"],
            "channel": e.get("channel"),
            "status": e.get("status"),
            "message": e.get("message"),
            "run": e.get("run"),
            "out_dir": str(out),
            "complete_now": complete,
            "kind": kind,
            "transient_io": kind in ("permission_denied", "usb_write_zero"),
        }
        if e.get("status") != "error":
            if complete and tif in ever_error:
                caught.append(rec)
            continue
        if complete:
            caught.append(rec)
            continue
        # Mass ENOENT from an old drive-drop run, for jobs the current run
        # has not reached yet — do not treat as catch-up failures.
        i = int(e.get("i") or 0)
        if (
            kind == "missing_source"
            and e.get("run") != current_run
            and i > current_max_i
        ):
            pending_ahead.append(rec)
            continue
        outstanding.append(rec)

    extra: list[dict[str, Any]] = []
    known = {Path(r["out_dir"]) for r in outstanding}
    if dest_root.exists():
        for d in dest_root.rglob("defringe_v4"):
            if dest_complete(d) or d in known:
                continue
            files = sorted(p.name for p in d.iterdir() if p.is_file())
            # Live stack writes *.partial.tif first; not a catch-up target.
            if files and all(n.endswith(".partial.tif") for n in files):
                continue
            extra.append({"out_dir": str(d), "files": files})

    return {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "note": (
            "Permission denied is usually a transient USB/dest lock. "
            "The same process_v4 --out command with skip-existing retries any dest "
            "that is not artifacts_complete."
        ),
        "outstanding_errors": outstanding,
        "later_succeeded": caught,
        "pending_ahead": pending_ahead,
        "incomplete_dest_not_in_error_log": extra,
        "current_run": current_run,
        "current_max_i": current_max_i,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# v4 dest batch — catch-up list",
        "",
        f"Updated: {payload['updated']}",
        "",
        payload["note"],
        "",
        "Retry (after the main run finishes, or on any resume):",
        "",
        "```",
        'python -m batch_defringe.process_v4 --root "E:\\Rasmus-Guillermo\\ECF1" --out "F:\\CollectedData"',
        "```",
        "",
        "Complete dest folders are skipped. Incomplete / permission-denied dests are processed again.",
        "",
        "## Still open (retry these)",
        "",
    ]
    open_rows = payload["outstanding_errors"]
    if not open_rows:
        lines.append("None.")
    else:
        for r in open_rows:
            kind = r.get("kind") or ("transient I/O" if r.get("transient_io") else "error")
            lines.append(
                f"- job {r.get('i')} {r.get('channel')} ({kind}): `{r.get('message')}`"
            )
            lines.append(f"  - `{r['tif']}`")
    pending = payload.get("pending_ahead") or []
    if pending:
        lines.extend(
            [
                "",
                f"## Not yet reached by current run `{payload.get('current_run')}` "
                f"(through job {payload.get('current_max_i')})",
                "",
                "Old missing-file logs from a drive drop. The live batch will try them "
                "when it gets there; only add to catch-up if they still fail.",
                "",
                f"{len(pending)} stacks in this bucket (not listed).",
            ]
        )
    extra = payload["incomplete_dest_not_in_error_log"]
    if extra:
        lines.extend(["", "## Incomplete dest (not in an error log)", ""])
        for r in extra:
            lines.append(f"- `{r['out_dir']}`")
            if r.get("files"):
                lines.append(f"  - files: {', '.join(r['files'])}")
    lines.extend(["", "## Logged as error, later complete", ""])
    caught = payload["later_succeeded"]
    if not caught:
        lines.append("None yet.")
    else:
        for r in caught:
            lines.append(f"- job {r.get('i')} {r.get('channel')}: `{r['tif']}`")
    lines.append("")
    return "\n".join(lines)


def write_catchup(
    *,
    source_root: Path = DEFAULT_SRC,
    dest_root: Path = DEFAULT_DST,
    notes_path: Path | None = None,
) -> dict[str, Path]:
    payload = build_catchup(source_root=source_root, dest_root=dest_root)
    runs = dest_root / ".defringe_v4_runs"
    runs.mkdir(parents=True, exist_ok=True)
    json_path = runs / "catchup.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = render_md(payload)
    dest_md = runs / "catchup.md"
    dest_md.write_text(md, encoding="utf-8")
    written = {"json": json_path, "dest_md": dest_md}
    if notes_path is not None:
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(md, encoding="utf-8")
        written["notes"] = notes_path
    return written


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    paths = write_catchup(notes_path=repo / "notes" / "V4_DEST_CATCHUP.md")
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
