"""Interactive / CLI entry for batch v2.2 defringe."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .discover import NO_XML_COMPUTER, discover_stacks
from .priors import cache_root
from .process import process_stack

WARNING_FILENAME = "DEFRINGE_WARNING_NO_EXPERIMENT_XML.txt"


def pick_root_directory(initial: str | None = None) -> Path | None:
    """Open a folder dialog; fall back to typed path if GUI unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(
            title="Select experiment root for batch defringe (v2.2)",
            initialdir=initial or str(Path.home()),
            mustexist=True,
        )
        root.destroy()
        if path:
            return Path(path)
    except Exception as exc:  # noqa: BLE001
        print(f"(folder dialog unavailable: {exc})")

    typed = input("Enter root directory path: ").strip().strip('"')
    if not typed:
        return None
    return Path(typed)


def _write_missing_xml_warning(tif_path: Path, root: Path) -> Path:
    """Leave a warning text file next to the stack when Experiment.xml is missing."""
    warning_path = tif_path.parent / WARNING_FILENAME
    try:
        rel = tif_path.relative_to(root)
    except ValueError:
        rel = tif_path
    text = (
        "WARNING: No Experiment.xml was found for this stack.\n"
        "\n"
        "batch_defringe could not identify the recording microscope, so it used a\n"
        "fresh fringe seed (no microscope x channel prior).\n"
        "\n"
        f"Stack: {rel}\n"
        f"Computer tag used: {NO_XML_COMPUTER}\n"
        "\n"
        "Add/restore Experiment.xml beside the trial DATA folder and re-run if you\n"
        "want microscope-specific soft priors.\n"
    )
    warning_path.write_text(text, encoding="utf-8")
    return warning_path


def run_batch(
    root: Path,
    *,
    dry_run: bool = False,
    skip_existing: bool = True,
    write_diagnostics: bool = True,
    assume_yes: bool = True,
) -> int:
    root = root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    print(f"\nScanning: {root}")
    jobs = discover_stacks(root)
    if not jobs:
        print(
            "No stacks found under DATA/ "
            "(looking for ChanA_stk.tif / ChanB_stk.tif)."
        )
        return 1

    computers = sorted({j.computer for j in jobs})
    n_missing_xml = sum(1 for j in jobs if j.missing_xml)
    print(f"Found {len(jobs)} channel stacks.")
    print(f"Microscopes (Computer): {', '.join(computers)}")
    if n_missing_xml:
        print(
            f"WARNING: {n_missing_xml} stack(s) have no Experiment.xml "
            f"— will use fresh seed and write {WARNING_FILENAME}."
        )
    for j in jobs[:12]:
        try:
            rel = j.tif_path.relative_to(root)
        except ValueError:
            rel = j.tif_path
        tag = j.computer
        if j.missing_xml:
            tag += ", NO XML"
        print(f"  - {rel}  [{tag}]")
    if len(jobs) > 12:
        print(f"  ... and {len(jobs) - 12} more")

    if dry_run:
        print("\nDry-run only - no defringe performed.")
        return 0

    print("\nStarting defringe with v2.2 pack_D.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = cache_root(root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.jsonl"
    print(f"Run log: {run_dir}")

    counts = {"ok": 0, "skipped": 0, "needs_review": 0, "error": 0}
    with open(summary_path, "w", encoding="utf-8") as logfh:
        for si, job in enumerate(jobs, 1):
            try:
                rel = job.tif_path.relative_to(root)
            except ValueError:
                rel = job.tif_path

            print(f"\n[{si}/{len(jobs)}] {rel}  ({job.computer})")
            if job.missing_xml:
                warn_path = _write_missing_xml_warning(job.tif_path, root)
                print(
                    f"  WARNING: no Experiment.xml — fresh seed; "
                    f"wrote {warn_path.name}"
                )

            diag = None
            if write_diagnostics:
                diag = run_dir / "diagnostics" / f"{si:04d}_{job.channel}"

            try:
                result = process_stack(
                    job.tif_path,
                    batch_root=root,
                    computer=job.computer,
                    channel=job.channel,
                    fingerprint=job.fingerprint,
                    skip_existing=skip_existing,
                    diag_dir=diag,
                    force_fresh_seed=job.missing_xml,
                    update_prior_on_success=not job.missing_xml,
                    recording_date=job.date_utc,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ERROR: {exc}")
                counts["error"] += 1
                logfh.write(
                    json.dumps(
                        {
                            "stack": str(rel),
                            "channel": job.channel,
                            "computer": job.computer,
                            "missing_xml": job.missing_xml,
                            "status": "error",
                            "message": str(exc),
                            "tif": str(job.tif_path),
                        }
                    )
                    + "\n"
                )
                logfh.flush()
                continue

            counts[result.status] = counts.get(result.status, 0) + 1
            tag = result.status.upper()
            extra = []
            if result.used_prior:
                extra.append("prior")
                if result.prior_branch:
                    extra.append(f"branch={result.prior_branch}")
            if result.reseeded:
                extra.append("reseed")
            if job.missing_xml:
                extra.append("no_xml_fresh")
            if result.families_q:
                extra.append("q=" + ",".join(f"{q:.1f}" for q in result.families_q))
            print(
                f"    {tag}: {result.message}"
                + (f" ({'; '.join(extra)})" if extra else "")
            )
            if result.status in ("ok", "needs_review") and result.out_dir is not None:
                print(f"    readout: {result.out_dir}")

            logfh.write(
                json.dumps(
                    {
                        "stack": str(rel),
                        "channel": job.channel,
                        "computer": job.computer,
                        "missing_xml": job.missing_xml,
                        "status": result.status,
                        "message": result.message,
                        "tif": str(job.tif_path),
                        "out": str(result.out_tif) if result.out_tif else None,
                        "out_dir": str(result.out_dir) if result.out_dir else None,
                        "removed": str(result.removed_tif) if result.removed_tif else None,
                        "overview_pdf": str(result.overview_pdf)
                        if result.overview_pdf
                        else None,
                        "used_prior": result.used_prior,
                        "reseeded": result.reseeded,
                        "prior_branch": result.prior_branch,
                        "families_q": result.families_q,
                    }
                )
                + "\n"
            )
            logfh.flush()

    print("\nDone.")
    print(
        f"  ok={counts['ok']}  skipped={counts['skipped']}  "
        f"needs_review={counts['needs_review']}  error={counts['error']}"
    )
    print(f"  priors/logs: {cache_root(root)}")
    if counts["needs_review"] or counts["error"]:
        print(f"  see {summary_path}")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Batch v2.2 PMT defringe. Default: pick a root folder, find "
            "DATA/**/ChanA_stk.tif and ChanB_stk.tif, defringe with microscope x "
            "channel soft priors from Experiment.xml."
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        help="Experiment root (if omitted, a folder picker is shown)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover stacks; do not defringe",
    )
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-defringe even if defringe_v22/*_defringed_v22.tif already exists",
    )
    ap.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Do not write per-stack diagnostics under .defringe_cache/runs/",
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Ask Y/n before starting (default: start after root is chosen)",
    )
    args = ap.parse_args(argv)

    root = args.root
    picked_via_gui = False
    if root is None:
        print("Select the experiment root directory…")
        root = pick_root_directory()
        picked_via_gui = True
        if root is None:
            print("No directory selected.")
            return 1

    assume_yes = True
    if args.confirm and not args.dry_run:
        ans = input("Start defringe with v2.2 pack_D? [Y/n] ").strip().lower()
        if ans in ("n", "no"):
            print("Aborted.")
            return 0

    return run_batch(
        root,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip_existing,
        write_diagnostics=not args.no_diagnostics,
        assume_yes=assume_yes or picked_via_gui,
    )


if __name__ == "__main__":
    sys.exit(main())
