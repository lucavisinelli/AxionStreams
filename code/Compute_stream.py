#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

import stream


IGNORED_SUFFIXES = (
    "_stream.npz",
    "_summary.npz",
    "_aggregate.npz",
)

SLOPE_CG_KEY = "cg_slope_window"
SLOPE_LOCAL_KEY = "local_slope_window"


def looks_like_raw_history(path: str) -> bool:
    name = os.path.basename(path)
    if not (name.startswith("AMC_history_") and name.endswith(".npz")):
        return False
    if any(name.endswith(sfx) for sfx in IGNORED_SUFFIXES):
        return False
    if "aggregate" in name or "summary" in name:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
        return ("dM_strip" in files) and ("t_launch_s" in files)
    except Exception:
        return False


def list_raw_histories_in_dir(directory: str) -> List[str]:
    candidates = sorted(glob.glob(os.path.join(directory, "AMC_history_*.npz")))
    return [f for f in candidates if looks_like_raw_history(f)]


def expand_input_path(path_input: str) -> List[str]:
    if os.path.isfile(path_input):
        base = os.path.basename(path_input)
        if base == "event_history_dirs.txt":
            dirs: List[str] = []
            with open(path_input, "r", encoding="utf-8") as f:
                for line in f:
                    d = line.strip()
                    if d:
                        dirs.append(d)
            files: List[str] = []
            for d in dirs:
                files.extend(list_raw_histories_in_dir(d))
            return sorted(files)
        if looks_like_raw_history(path_input):
            return [path_input]
        raise RuntimeError(
            f"Unsupported input file: {path_input}. Provide a raw AMC_history_*.npz file, "
            "an event_histories_* directory, or event_history_dirs.txt."
        )

    if os.path.isdir(path_input):
        files = list_raw_histories_in_dir(path_input)
        if not files:
            raise RuntimeError(
                f"No raw AMC_history_*.npz files found in {path_input}. "
                "Derived files such as *_stream.npz are ignored by design."
            )
        return files

    raise RuntimeError(f"Input path not found: {path_input}")


def discover_history_files(inputs: Iterable[str]) -> List[str]:
    all_files: List[str] = []
    for item in inputs:
        all_files.extend(expand_input_path(item))
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def infer_radius_kpc_from_history_file(path: str) -> float:
    parent = os.path.basename(os.path.dirname(path.rstrip(os.sep)))
    marker = "_a="
    if marker not in parent:
        return np.nan
    try:
        a_pc_str = parent.split(marker, 1)[1]
        a_pc = float(a_pc_str)
        return a_pc / 1.0e3
    except Exception:
        return np.nan


def median_band(stacked: np.ndarray):
    med = np.nanmedian(stacked, axis=0)
    p16 = np.nanpercentile(stacked, 16, axis=0)
    p84 = np.nanpercentile(stacked, 84, axis=0)
    return med, p16, p84


def save_summary_csv(path: str, rows: List[Dict[str, float]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_safe_prefix(output_dir: str, history_file: str, used: Dict[str, int]) -> str:
    parent = os.path.basename(os.path.dirname(history_file.rstrip(os.sep)))
    stem = os.path.splitext(os.path.basename(history_file))[0]
    base = f"{parent}__{stem}" if parent else stem
    count = used.get(base, 0)
    used[base] = count + 1
    if count > 0:
        base = f"{base}__{count:03d}"
    return os.path.join(output_dir, base)


def finite_values(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def write_text_summary(path: str, rows: List[Dict[str, float]]) -> None:
    n = len(rows)
    slope_cg = finite_values(rows, SLOPE_CG_KEY)
    slope_local = finite_values(rows, SLOPE_LOCAL_KEY)
    axis_ratio = finite_values(rows, "final_axis_ratio")
    retention_cg = finite_values(rows, "rho_cg_retention")
    retention_local = finite_values(rows, "rho_local_retention")
    radii = finite_values(rows, "a_kpc")

    def frac(key: str) -> float:
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else np.nan

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Number of processed histories: {n}\n")
        if len(radii):
            f.write(f"Median semi-major axis [kpc]: {np.median(radii):.3f}\n")
        if len(slope_cg):
            f.write(f"Median windowed coarse-grained density slope: {np.median(slope_cg):.3f}\n")
            f.write(f"16-84 percentile coarse-grained slope: {np.percentile(slope_cg,16):.3f} to {np.percentile(slope_cg,84):.3f}\n")
        if len(slope_local):
            f.write(f"Median windowed local density slope: {np.median(slope_local):.3f}\n")
            f.write(f"16-84 percentile local slope: {np.percentile(slope_local,16):.3f} to {np.percentile(slope_local,84):.3f}\n")
        if len(axis_ratio):
            f.write(f"Median final axis ratio (major/minor): {np.median(axis_ratio):.3f}\n")
        if len(retention_cg):
            f.write(f"Median coarse-grained retention rho_f/rho_i: {np.median(retention_cg):.3e}\n")
        if len(retention_local):
            f.write(f"Median local retention rho_f/rho_i: {np.median(retention_local):.3e}\n")
        f.write(f"Fraction shallower than t^-3 (coarse-grained): {frac('is_shallower_than_t3_cg'):.3f}\n")
        f.write(f"Fraction shallower than t^-3 (local): {frac('is_shallower_than_t3_local'):.3f}\n")
        f.write(f"Fraction filamentary (axis ratio > 10): {frac('is_filamentary'):.3f}\n")
        f.write(f"Fraction with local density surviving better than coarse-grained: {frac('local_survives_better'):.3f}\n")


def plot_density(times_myr, rho_cg_med, rho_cg_p16, rho_cg_p84, rho_local_med, rho_local_p16, rho_local_p84, outpath):
    plt.figure(figsize=(7, 5))
    plt.plot(times_myr, rho_cg_med, color="k", label=r"$\rho_{\rm cg}$ median")
    plt.fill_between(times_myr, rho_cg_p16, rho_cg_p84, alpha=0.25)
    plt.plot(times_myr, rho_local_med, color="tab:red", label=r"$\rho_{\rm local}$ median")
    plt.fill_between(times_myr, rho_local_p16, rho_local_p84, alpha=0.18, color="tab:red")
    plt.yscale("log")
    plt.xlabel("Time [Myr]")
    plt.ylabel(r"Density [$M_\odot\,{\rm pc}^{-3}$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_axes(times_myr, axis_major_med, axis_major_p16, axis_major_p84, axis_minor_med, axis_minor_p16, axis_minor_p84, outpath):
    plt.figure(figsize=(7, 5))
    plt.plot(times_myr, axis_major_med, color="k", label="Major axis")
    plt.fill_between(times_myr, axis_major_p16, axis_major_p84, alpha=0.25)
    plt.plot(times_myr, axis_minor_med, color="tab:green", label="Minor axis")
    plt.fill_between(times_myr, axis_minor_p16, axis_minor_p84, alpha=0.18, color="tab:green")
    plt.yscale("log")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Spatial sigma [pc]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_slope_hist(rows: List[Dict[str, float]], outpath: str):
    slope_cg = finite_values(rows, SLOPE_CG_KEY)
    slope_local = finite_values(rows, SLOPE_LOCAL_KEY)
    plt.figure(figsize=(7, 5))
    bins = np.linspace(-4.5, 1.0, 28)
    if len(slope_cg):
        plt.hist(slope_cg, bins=bins, alpha=0.45, label=r"$\rho_{\rm cg}$ slope")
    if len(slope_local):
        plt.hist(slope_local, bins=bins, alpha=0.45, label=r"$\rho_{\rm local}$ slope")
    plt.axvline(-3.0, color="k", linestyle="--", linewidth=1.2, label=r"$t^{-3}$")
    plt.xlabel(r"Windowed slope $d\log\rho/d\log t$")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_retention_scatter(rows: List[Dict[str, float]], outpath: str):
    x = finite_values(rows, "rho_cg_retention")
    y = finite_values(rows, "rho_local_retention")
    n = min(len(x), len(y))
    plt.figure(figsize=(6, 6))
    if n:
        plt.scatter(x[:n], y[:n], alpha=0.7)
    lo, hi = 1e-10, 1.0
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="k", linewidth=1.0)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(r"Coarse-grained retention $\rho_f/\rho_i$")
    plt.ylabel(r"Local retention $\rho_f/\rho_i$")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_axisratio_vs_slope(rows: List[Dict[str, float]], outpath: str):
    x = np.array([r.get("final_axis_ratio", np.nan) for r in rows], dtype=float)
    y = np.array([r.get(SLOPE_LOCAL_KEY, np.nan) for r in rows], dtype=float)
    sel = np.isfinite(x) & np.isfinite(y) & (x > 0)
    plt.figure(figsize=(6.5, 5))
    if np.any(sel):
        plt.scatter(x[sel], y[sel], alpha=0.7)
    plt.axhline(-3.0, linestyle="--", color="k", linewidth=1.0)
    plt.xscale("log")
    plt.xlabel("Final axis ratio (major/minor)")
    plt.ylabel(r"Windowed local slope $d\log\rho/d\log t$")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_t3_comparison(times_myr, rho_cg_med, rho_local_med, outpath: str):
    finite = np.where(np.isfinite(rho_cg_med) & (rho_cg_med > 0) & (times_myr > 0))[0]
    plt.figure(figsize=(7, 5))
    if len(finite):
        i0 = finite[0]
        ref = rho_cg_med[i0] * (times_myr / times_myr[i0]) ** (-3.0)
        plt.plot(times_myr, ref, linestyle="--", color="0.4", label=r"Reference $t^{-3}$")
    plt.plot(times_myr, rho_cg_med, color="k", label=r"$\rho_{\rm cg}$ median")
    plt.plot(times_myr, rho_local_med, color="tab:red", label=r"$\rho_{\rm local}$ median")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Time [Myr]")
    plt.ylabel(r"Density [$M_\odot\,{\rm pc}^{-3}$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def write_radius_calibration_csv(path: str, rows: List[Dict[str, float]]) -> None:
    radii = sorted({float(r["a_kpc"]) for r in rows if np.isfinite(r.get("a_kpc", np.nan))})
    if not radii:
        return

    fieldnames = [
        "a_kpc",
        "n_histories",
        "cg_slope_window_median",
        "cg_slope_window_p16",
        "cg_slope_window_p84",
        "local_slope_window_median",
        "local_slope_window_p16",
        "local_slope_window_p84",
        "major_axis_growth_median",
        "minor_axis_growth_median",
        "final_axis_ratio_median",
        "rho_cg_retention_median",
        "rho_local_retention_median",
        "fraction_filamentary",
        "fraction_shallower_than_t3_cg",
        "fraction_shallower_than_t3_local",
    ]

    calib_rows: List[Dict[str, float]] = []
    for a_kpc in radii:
        sub = [r for r in rows if np.isfinite(r.get("a_kpc", np.nan)) and abs(r["a_kpc"] - a_kpc) < 1e-9]
        if not sub:
            continue

        def arr(key):
            vals = np.array([r.get(key, np.nan) for r in sub], dtype=float)
            return vals[np.isfinite(vals)]

        def frac(key):
            vals = arr(key)
            return float(np.mean(vals)) if len(vals) else np.nan

        cg = arr(SLOPE_CG_KEY)
        local = arr(SLOPE_LOCAL_KEY)
        major = arr("major_axis_growth")
        minor = arr("minor_axis_growth")
        axis_ratio = arr("final_axis_ratio")
        rho_cg_ret = arr("rho_cg_retention")
        rho_local_ret = arr("rho_local_retention")

        calib_rows.append({
            "a_kpc": a_kpc,
            "n_histories": len(sub),
            "cg_slope_window_median": float(np.median(cg)) if len(cg) else np.nan,
            "cg_slope_window_p16": float(np.percentile(cg, 16)) if len(cg) else np.nan,
            "cg_slope_window_p84": float(np.percentile(cg, 84)) if len(cg) else np.nan,
            "local_slope_window_median": float(np.median(local)) if len(local) else np.nan,
            "local_slope_window_p16": float(np.percentile(local, 16)) if len(local) else np.nan,
            "local_slope_window_p84": float(np.percentile(local, 84)) if len(local) else np.nan,
            "major_axis_growth_median": float(np.median(major)) if len(major) else np.nan,
            "minor_axis_growth_median": float(np.median(minor)) if len(minor) else np.nan,
            "final_axis_ratio_median": float(np.median(axis_ratio)) if len(axis_ratio) else np.nan,
            "rho_cg_retention_median": float(np.median(rho_cg_ret)) if len(rho_cg_ret) else np.nan,
            "rho_local_retention_median": float(np.median(rho_local_ret)) if len(rho_local_ret) else np.nan,
            "fraction_filamentary": frac("is_filamentary"),
            "fraction_shallower_than_t3_cg": frac("is_shallower_than_t3_cg"),
            "fraction_shallower_than_t3_local": frac("is_shallower_than_t3_local"),
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calib_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "input_paths",
        nargs="+",
        help=(
            "One or more raw AMC_history_*.npz files, event_histories_* directories, "
            "or event_history_dirs.txt files"
        ),
    )
    p.add_argument("--max_files", type=int, default=0, help="Maximum number of history files to analyze (0 = all)")
    p.add_argument("--total_tracers", type=int, default=6000)
    p.add_argument("--t_end_myr", type=float, default=10.0)
    p.add_argument("--dt_myr", type=float, default=0.01)
    p.add_argument("--vcirc", type=float, default=220.0)
    p.add_argument("--rsoft", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--t_min_slope_myr", type=float, default=0.5)
    p.add_argument("--t_max_slope_myr", type=float, default=5.0)
    p.add_argument(
        "--outdir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="stream_batch_output",
        help="Directory where all outputs will be written",
    )
    p.add_argument("--save_individual", action="store_true", help="Save per-history .npz and .png outputs")
    opts = p.parse_args()

    history_files = discover_history_files(opts.input_paths)
    if opts.max_files > 0:
        history_files = history_files[: opts.max_files]
    if not history_files:
        raise RuntimeError("No raw history files selected for analysis.")

    os.makedirs(opts.output_dir, exist_ok=True)
    print(f"Output directory: {opts.output_dir}")
    if os.listdir(opts.output_dir):
        print(f"[WARNING] Output directory '{opts.output_dir}' is not empty — files may be overwritten")

    rows: List[Dict[str, float]] = []
    rho_cg_all = []
    rho_local_all = []
    axis_major_all = []
    axis_minor_all = []
    times_ref = None
    prefix_counts: Dict[str, int] = {}

    print(f"Found {len(history_files)} raw history files")

    for i, hf in enumerate(history_files, start=1):
        print(f"[{i}/{len(history_files)}] Processing {hf}")
        prefix = make_safe_prefix(opts.output_dir, hf, prefix_counts)
        try:
            res = stream.run_event_stream(
                hf,
                total_tracers=opts.total_tracers,
                t_end_myr=opts.t_end_myr,
                dt_myr=opts.dt_myr,
                vcirc=opts.vcirc,
                rsoft=opts.rsoft,
                save_every=opts.save_every,
                seed=opts.seed + i - 1,
            )
        except Exception as exc:
            print(f"  Skipping due to error: {exc}")
            continue

        summ = stream.summarize_run(
            res,
            t_min_slope_myr=opts.t_min_slope_myr,
            t_max_slope_myr=opts.t_max_slope_myr,
        )
        summ["history_file"] = hf
        summ["a_kpc"] = infer_radius_kpc_from_history_file(hf)
        rows.append(summ)

        if opts.save_individual:
            stream.save_run(prefix, res)
            stream.plot_results(res, prefix)

        if times_ref is None:
            times_ref = res["times"].copy()
        elif len(res["times"]) != len(times_ref) or not np.allclose(res["times"], times_ref):
            raise RuntimeError("Inconsistent time grids across runs. Use common t_end_myr/dt_myr/save_every.")

        rho_cg_all.append(res["rho_cg"])
        rho_local_all.append(res["rho_local_nn"])
        axis_major_all.append(res["axes"][:, 0])
        axis_minor_all.append(res["axes"][:, 2])

    if not rows:
        raise RuntimeError("All selected files failed validation or processing.")

    summary_csv = os.path.join(opts.output_dir, "stream_summary.csv")
    save_summary_csv(summary_csv, rows)

    rho_cg_stack = np.array(rho_cg_all, dtype=float)
    rho_local_stack = np.array(rho_local_all, dtype=float)
    axis_major_stack = np.array(axis_major_all, dtype=float)
    axis_minor_stack = np.array(axis_minor_all, dtype=float)
    times_myr = stream.to_myr(times_ref)

    rho_cg_med, rho_cg_p16, rho_cg_p84 = median_band(rho_cg_stack)
    rho_local_med, rho_local_p16, rho_local_p84 = median_band(rho_local_stack)
    axis_major_med, axis_major_p16, axis_major_p84 = median_band(axis_major_stack)
    axis_minor_med, axis_minor_p16, axis_minor_p84 = median_band(axis_minor_stack)

    aggregate_npz = os.path.join(opts.output_dir, "stream_aggregate.npz")
    np.savez_compressed(
        aggregate_npz,
        times_myr=times_myr.astype(np.float32),
        rho_cg_median=rho_cg_med.astype(np.float32),
        rho_cg_p16=rho_cg_p16.astype(np.float32),
        rho_cg_p84=rho_cg_p84.astype(np.float32),
        rho_local_median=rho_local_med.astype(np.float32),
        rho_local_p16=rho_local_p16.astype(np.float32),
        rho_local_p84=rho_local_p84.astype(np.float32),
        axis_major_median=axis_major_med.astype(np.float32),
        axis_major_p16=axis_major_p16.astype(np.float32),
        axis_major_p84=axis_major_p84.astype(np.float32),
        axis_minor_median=axis_minor_med.astype(np.float32),
        axis_minor_p16=axis_minor_p16.astype(np.float32),
        axis_minor_p84=axis_minor_p84.astype(np.float32),
    )

    plot_density(
        times_myr, rho_cg_med, rho_cg_p16, rho_cg_p84,
        rho_local_med, rho_local_p16, rho_local_p84,
        os.path.join(opts.output_dir, "stream_density_evolution.png")
    )
    plot_axes(
        times_myr, axis_major_med, axis_major_p16, axis_major_p84,
        axis_minor_med, axis_minor_p16, axis_minor_p84,
        os.path.join(opts.output_dir, "stream_axes_evolution.png")
    )
    plot_slope_hist(rows, os.path.join(opts.output_dir, "stream_slope_histograms.png"))
    plot_retention_scatter(rows, os.path.join(opts.output_dir, "stream_retention_scatter.png"))
    plot_axisratio_vs_slope(rows, os.path.join(opts.output_dir, "stream_axisratio_vs_local_slope.png"))
    plot_t3_comparison(times_myr, rho_cg_med, rho_local_med, os.path.join(opts.output_dir, "stream_t3_comparison.png"))
    write_text_summary(os.path.join(opts.output_dir, "stream_interpretation.txt"), rows)
    write_radius_calibration_csv(os.path.join(opts.output_dir, "stream_calibration_by_radius.csv"), rows)

    print(f"Processed {len(rows)} history files successfully")
    print(f"Summary CSV: {summary_csv}")
    print(f"Aggregate NPZ: {aggregate_npz}")
    print(f"Interpretation TXT: {os.path.join(opts.output_dir, 'stream_interpretation.txt')}")
    print(f"Calibration CSV: {os.path.join(opts.output_dir, 'stream_calibration_by_radius.csv')}")
    print("Saved figures:")
    for name in [
        "stream_density_evolution.png",
        "stream_axes_evolution.png",
        "stream_slope_histograms.png",
        "stream_retention_scatter.png",
        "stream_axisratio_vs_local_slope.png",
        "stream_t3_comparison.png",
    ]:
        print(f"  - {os.path.join(opts.output_dir, name)}")


if __name__ == "__main__":
    main()
