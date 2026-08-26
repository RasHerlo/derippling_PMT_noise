"""CLI for dark-current control recordings: python -m darkcurrent"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .characterize import analyze_channel
from .confirm import confirm_channel
from .figures import plot_channel, plot_condition_compare
from .metadata import discover_dark_recordings, recording_to_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO / "darkcurrent" / "registry.json"
DEFAULT_MEASUREMENTS = REPO / "darkcurrent" / "measurements.json"


def cmd_census(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    recs = discover_dark_recordings(root)
    if not recs:
        print(f"No Experiment.xml found under {root}")
        return 1

    print(f"\nDark-current recordings under: {root}")
    print(f"Found {len(recs)} trial(s).\n")

    header = (
        f"{'trial':<12} {'date (UTC)':<20} {'scope':<10} {'chan':<11} "
        f"{'frames':>7} {'gainA/B':>9} {'PC set':>7} {'PC xml':>7} "
        f"{'scan':>5} {'2way':>5} {'avg':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in recs:
        chans = ",".join(sorted(r.stacks)) or "ABORTED"
        gains = f"{r.pmt.get('gainA')}/{r.pmt.get('gainB')}"
        print(
            f"{r.label:<12} {(r.date_utc or '?')[:19]:<20} {r.scope:<10} "
            f"{chans:<11} {str(r.scan.get('frames')):>7} {gains:>9} "
            f"{str(r.pockels_setting):>7} {str(r.pockels.get('start')):>7} "
            f"{str(r.scan.get('scanMode')):>5} "
            f"{str(r.scan.get('twoWayAlignment')):>5} "
            f"{str(r.scan.get('averageMode')) + 'x' + str(r.scan.get('averageNum')):>7}"
        )

    if any(r.pockels_setting is not None for r in recs):
        print(
            "\nNote: 'PC set' comes from the trial folder name and is authoritative; "
            "'PC xml' is what ThorImage stored and does not track it."
        )

    groups: dict[str, list[str]] = {}
    for r in recs:
        groups.setdefault(r.config_key, []).append(r.label)
    print(f"\nDistinct scan configurations: {len(groups)}")
    for key, labels in groups.items():
        print(f"  [{', '.join(labels)}]")
        print(f"    {key}")

    aborted = [r.label for r in recs if r.aborted]
    if aborted:
        print(f"\nAborted (no assembled DATA stacks, excluded): {', '.join(aborted)}")

    if args.registry:
        reg_path = Path(args.registry)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(root),
            "recordings": [recording_to_json(r) for r in recs],
        }
        reg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nRegistry written: {reg_path}")

    return 0


def cmd_characterize(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    recs = [r for r in discover_dark_recordings(root) if r.stacks]
    if not recs:
        print(f"No dark recordings with assembled DATA stacks under {root}")
        return 1

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        recs = [r for r in recs if r.label in wanted]
        if not recs:
            print(f"No recordings matched --only {args.only}")
            return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else root / ".darkcurrent_analysis" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nArtifacts: {out_dir}")
    print(f"Sampling: sample_n={args.sample_n} tiles={args.tiles} "
          f"temporal_every={args.temporal_every}")

    results = []
    for rec in recs:
        for channel, tif in sorted(rec.stacks.items()):
            print(f"\n[{rec.label} / {channel}] {tif.name}", flush=True)
            res = analyze_channel(
                tif,
                rec.label,
                channel,
                sample_n=args.sample_n,
                tiles=args.tiles,
                temporal_every=args.temporal_every,
                phase_count=args.phase_count,
                do_q_track=not args.no_q_track,
            )
            payload = res.to_json()
            payload["config_key"] = rec.config_key
            payload["scope"] = rec.scope
            payload["pockels_setting"] = rec.pockels_setting
            payload["pockels_start_xml"] = rec.pockels.get("start")
            payload["gain"] = rec.pmt.get(f"gain{channel[-1]}")
            payload["date_utc"] = rec.date_utc
            results.append(payload)

            fam = res.families["production_families"]
            relaxed = res.families["relaxed_families"]
            print(
                f"    intensity mean={res.intensity['mean']:.1f} "
                f"p99={res.intensity['p99']:.1f} max={res.intensity['max']:.0f}"
            )
            print(
                f"    production families={len(fam)} relaxed={len(relaxed)} "
                f"row_z_max={res.families['row_z_max']:.2f}"
            )
            for f in fam or relaxed:
                print(
                    f"      q={f['q']:.0f} hi={f['hi']} paired={f['paired']} "
                    f"row_z={f['row_score']:.2f} period={f['period_px']:.1f}px"
                )
            if not fam and not relaxed:
                top = res.families["top_row_peaks"][0]
                print(f"      no family; strongest row q={top['q']} z={top['row_z']:.2f}")
            print(
                f"    fx support(z>3.5)={res.fx['support_by_threshold']['3.5']['n_bins']} bins "
                f"|fx|={res.fx['support_by_threshold']['3.5']['abs_fx_min']}-"
                f"{res.fx['support_by_threshold']['3.5']['abs_fx_max']}"
            )
            print("    candidate evidence (excess over matched control rows):")
            for s in res.family_snr:
                print(
                    f"      q={s['q']:.0f} period={s['period_px']:.1f}px "
                    f"bins={s['support_bins']:>3} excess={s['excess']:.1f} "
                    f"noise={s['noise']:.1f} SNR={s['snr']:.2f}"
                )
            print(f"    analysis q={res.q_used:.0f} (strongest evidence)")
            print(
                f"    FOV excess min={res.fov['excess_min']:.1f} "
                f"max={res.fov['excess_max']:.1f} "
                f"ratio={res.fov['excess_ratio_max_min']:.2f} "
                f"tileSNR={res.fov['snr_min']:.1f}-{res.fov['snr_max']:.1f}"
            )
            print(
                f"      tile row means={[round(v, 1) for v in res.fov['row_means']]}"
            )
            print(
                f"      tile col means={[round(v, 1) for v in res.fov['col_means']]}"
            )
            print(
                f"    x-profile edge/middle={res.x_prof['edge_over_middle']:.2f} "
                f"max/min={res.x_prof['max_over_min']:.2f} "
                f"peak_at_x={res.x_prof['argmax_px']:.0f}px"
            )
            print(
                f"      excess vs x={[round(v) for v in res.x_prof['excess']]}"
            )
            print(
                f"    tile dominant q spread={res.tile_census['q_spread']:.0f} "
                f"distinct={res.tile_census['q_unique_count']} "
                f"z={res.tile_census['z_min']:.1f}-{res.tile_census['z_max']:.1f}"
            )
            print(
                f"    temporal excess mean={res.temporal['mean']:.1f} "
                f"CV={res.temporal['cv']:.2f} "
                f"SNR_vs_control={res.temporal['snr_mean']:.1f} "
                f"frames_below_control={res.temporal['frac_below_control']*100:.0f}%"
            )
            print(
                f"    phase (stride 1): |step|med="
                f"{res.phase['step_abs_median_rad']:.3f}rad "
                f"coherence={res.phase['step_coherence']:.3f} "
                f"mag_lag1={res.phase['magnitude_lag1_autocorr']:.3f}"
            )
            if res.q_track:
                print(
                    f"    q track median={res.q_track['q_median']:.1f} "
                    f"range={res.q_track['q_min']}-{res.q_track['q_max']} "
                    f"span={res.q_track['q_span']}"
                )
            plot_channel(payload, out_dir)

    plot_condition_compare(results, out_dir)

    (out_dir / "measurements.json").write_text(
        json.dumps({"root": str(root), "run": run_id, "channels": results}, indent=2),
        encoding="utf-8",
    )
    if args.metrics:
        mpath = Path(args.metrics)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if mpath.is_file():
            try:
                history = json.loads(mpath.read_text(encoding="utf-8")).get("runs", [])
            except Exception:  # noqa: BLE001
                history = []
        history.append(
            {
                "run": run_id,
                "root": str(root),
                "artifacts": str(out_dir),
                "channels": results,
            }
        )
        mpath.write_text(json.dumps({"runs": history}, indent=2), encoding="utf-8")
        print(f"\nMeasurement history updated: {mpath}")

    print(f"\nDone. Artifacts in {out_dir}")
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    recs = [r for r in discover_dark_recordings(root) if r.stacks]
    if not recs:
        print(f"No dark recordings with assembled DATA stacks under {root}")
        return 1

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        recs = [r for r in recs if r.label in wanted]
        if not recs:
            print(f"No recordings matched --only {args.only}")
            return 1

    channels = None
    if args.channel:
        channels = {s.strip() for s in args.channel.split(",")}

    forced = [float(s) for s in args.q.split(",")] if args.q else None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else root / ".darkcurrent_analysis" / f"confirm_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nArtifacts: {out_dir}")
    print(
        f"Settings: sample_n={args.sample_n} dc_guard={args.dc_guard} "
        f"fx_medfilt={args.fx_medfilt} hp_kernel={args.hp_kernel} "
        f"temporal_frames={args.temporal_frames}"
    )
    print(
        "Tests: background-policy probe | skirt-suppressed amplitude (gain "
        "corrected, in-row fx background) | temporal character"
    )

    results = []
    for rec in recs:
        for channel, tif in sorted(rec.stacks.items()):
            if channels and channel not in channels:
                continue
            print(f"\n[{rec.label} / {channel}] {tif.name}", flush=True)
            payload = confirm_channel(
                tif,
                rec.label,
                channel,
                sample_n=args.sample_n,
                temporal_frames=args.temporal_frames,
                dc_guard=args.dc_guard,
                medfilt=args.fx_medfilt,
                hp_kernel=args.hp_kernel,
                frame_rate=rec.scan.get("frameRate"),
                forced_qs=forced,
            )
            payload["config_key"] = rec.config_key
            payload["scope"] = rec.scope
            payload["pockels_setting"] = rec.pockels_setting
            payload["gain"] = rec.pmt.get(f"gain{channel[-1]}")
            results.append(payload)

            pk = payload["picked"]
            peaks = ", ".join(
                f"q={p['q']}(z={p['row_z']:.1f})" for p in pk["row_z_peaks"][:5]
            )
            print(f"    row-z peaks: {peaks}")
            ns = payload.get("null_summary") or {}
            an = ns.get("amplitude") or {}
            print(
                f"    positive control q={pk.get('positive_control')} | "
                f"residual noise scale="
                f"{payload['settings']['residual_noise_scale']:.2f}"
            )
            if an.get("n_rows"):
                print(
                    f"    amplitude null over {an['n_rows']} rows: "
                    f"SNR med={an['snr_median']:.2f} sigma={an['snr_sigma']:.2f} "
                    f"core_max={an['snr_core_max']:.2f} -> threshold "
                    f"{ns.get('amplitude_threshold', 0):.2f}; "
                    f"{an['n_structured_rows']} rows carry structure "
                    f"(q={an['structured_q']})"
                )
            if ns.get("n_rows"):
                print(
                    f"    temporal null over {ns['n_rows']} empty rows: "
                    f"prominence med={ns['prominence_median']:.1f} "
                    f"core_max={ns.get('prominence_core_max', 0):.1f} "
                    f"max={ns['prominence_max']:.1f} "
                    f"({ns.get('prominence_outliers', 0)} outliers) -> threshold "
                    f"{ns.get('prominence_threshold', 0):.1f}"
                )
                print(
                    f"    false positives: {ns['n_false_positive']}/{ns['n_rows']} "
                    f"({100 * ns['false_positive_rate']:.0f}%)"
                    + (f" at q={ns['false_positive_q']}" if ns["n_false_positive"] else "")
                )
            for c in payload["candidates"]:
                amp = c["amplitude"]
                sup = amp.get("support") or {}
                tmp = c["temporal"]
                print(
                    f"    q={c['q']:.0f} [{c['role']}] period={c['period_px']:.1f}px "
                    f"-> {c['verdict'].upper()}"
                )
                bp = c.get("background_probe") or {}
                if bp:
                    lg, ds = bp["legacy"], bp["dc_safe"]
                    print(
                        f"      bg policy: legacy nearest_dy={lg['nearest_bg_dy']} "
                        f"uses_DC_row={lg['uses_dc_row']} | "
                        f"dc_safe nearest_dy={ds['nearest_bg_dy']}"
                    )
                if "excess" in amp:
                    corr = amp.get("excess_gain_corrected")
                    corr_txt = "n/a" if corr is None else f"{corr:.1f}"
                    print(
                        f"      amplitude: excess={amp['excess']:.1f} "
                        f"ctrl_row={amp['control_row_excess']:.1f} "
                        f"SNR={amp['snr']:.2f} "
                        f"gain={amp['filter_gain']:.3f} corrected={corr_txt}"
                    )
                    print(
                        f"      support: z_max={sup.get('z_max', 0.0):.2f} "
                        f"{sup.get('support_bins')} bins "
                        f"({100 * sup.get('support_frac_of_valid', 0):.1f}% of valid) "
                        f"|fx|={sup.get('abs_fx_min')}-{sup.get('abs_fx_max')}"
                    )
                hz = tmp.get("f_peak_hz")
                hz_txt = "" if hz is None else f" ({hz:+.2f} Hz)"
                print(
                    f"      temporal: peak={tmp.get('f_peak_cycles_per_frame', 0):+.4f}"
                    f" cyc/frame{hz_txt} "
                    f"prominence={tmp.get('prominence', 0):.1f} "
                    f"static_frac={tmp.get('static_power_fraction', 0):.3f} "
                    f"bins={tmp.get('n_bins')}"
                )
                if tmp.get("peak_is_static", True) and tmp.get(
                    "f_nonstatic_peak_cycles_per_frame"
                ) is not None:
                    print(
                        f"        strongest moving peak="
                        f"{tmp['f_nonstatic_peak_cycles_per_frame']:+.4f} cyc/frame "
                        f"prominence={tmp.get('nonstatic_prominence', 0):.1f}"
                    )
                failed = [k for k, v in c["checks"].items() if not v]
                if failed:
                    print(f"      failed checks: {', '.join(failed)}")

    payload_path = out_dir / "confirm.json"
    payload_path.write_text(
        json.dumps({"root": str(root), "run": run_id, "channels": results}, indent=2),
        encoding="utf-8",
    )

    print("\nSummary")
    print(
        f"{'trial':<12} {'chan':<6} {'q':>5} {'role':<20} {'verdict':<22} {'null FP':>8}"
    )
    print("-" * 78)
    for payload in results:
        ns = payload.get("null_summary") or {}
        fp = (
            f"{ns['n_false_positive']}/{ns['n_rows']}" if ns.get("n_rows") else "-"
        )
        for c in payload["candidates"]:
            print(
                f"{payload['label']:<12} {payload['channel']:<6} {c['q']:>5.0f} "
                f"{c['role']:<20} {c['verdict']:<22} {fp:>8}"
            )

    print(f"\nDone. {payload_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m darkcurrent",
        description="Inspect and characterise dark-current PMT fringe controls.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    ap_census = sub.add_parser(
        "census", help="List dark-current recordings and their acquisition settings"
    )
    ap_census.add_argument("--root", required=True, help="Folder containing dark recordings")
    ap_census.add_argument(
        "--registry",
        nargs="?",
        const=str(DEFAULT_REGISTRY),
        default=None,
        help=f"Write machine-readable registry (default path: {DEFAULT_REGISTRY})",
    )
    ap_census.set_defaults(func=cmd_census)

    ap_ch = sub.add_parser(
        "characterize", help="Measure the fringe layer (families, FOV, dynamics)"
    )
    ap_ch.add_argument("--root", required=True, help="Folder containing dark recordings")
    ap_ch.add_argument("--only", help="Comma-separated trial labels to restrict to")
    ap_ch.add_argument("--out", help="Artifact directory (default: <root>/.darkcurrent_analysis/<run>)")
    ap_ch.add_argument("--sample-n", type=int, default=160, help="Frames sampled for spectra")
    ap_ch.add_argument("--tiles", type=int, default=4, help="FOV tile grid per axis")
    ap_ch.add_argument("--temporal-every", type=int, default=4, help="Frame stride for time trace")
    ap_ch.add_argument(
        "--phase-count",
        type=int,
        default=300,
        help="Consecutive frames used for phase continuity (stride 1)",
    )
    ap_ch.add_argument("--no-q-track", action="store_true", help="Skip per-block q tracking")
    ap_ch.add_argument(
        "--metrics",
        nargs="?",
        const=str(DEFAULT_MEASUREMENTS),
        default=None,
        help=f"Append run to measurement history (default: {DEFAULT_MEASUREMENTS})",
    )
    ap_ch.set_defaults(func=cmd_characterize)

    ap_cf = sub.add_parser(
        "confirm",
        help="Independently re-test ridge candidates without DC-adjacent backgrounds",
    )
    ap_cf.add_argument("--root", required=True, help="Folder containing dark recordings")
    ap_cf.add_argument("--only", help="Comma-separated trial labels to restrict to")
    ap_cf.add_argument("--channel", help="Comma-separated channels (e.g. ChanA)")
    ap_cf.add_argument(
        "--q",
        help="Comma-separated q values to test instead of auto-picked candidates",
    )
    ap_cf.add_argument("--out", help="Artifact directory")
    ap_cf.add_argument("--sample-n", type=int, default=128, help="Frames sampled for spectra")
    ap_cf.add_argument(
        "--dc-guard",
        type=int,
        default=12,
        help="Rows around DC that may never serve as background",
    )
    ap_cf.add_argument(
        "--fx-medfilt",
        type=int,
        default=81,
        help="Width of the in-row median filter along fx",
    )
    ap_cf.add_argument(
        "--hp-kernel",
        type=int,
        default=161,
        help="Per-column boxcar width removed along y to suppress the DC skirt",
    )
    ap_cf.add_argument(
        "--temporal-frames",
        type=int,
        default=512,
        help="Consecutive frames used for the temporal character test",
    )
    ap_cf.set_defaults(func=cmd_confirm)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
