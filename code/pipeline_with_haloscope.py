#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
try:
    import stream
except ImportError:
    stream = None

YEAR = 365.25 * 24.0 * 3600.0

IGNORED_SUFFIXES = (
    "_stream.npz",
    "_summary.npz",
    "_aggregate.npz",
)
SLOPE_CG_KEY = "cg_slope_window"
SLOPE_LOCAL_KEY = "local_slope_window"
DEFAULT_A_VALUES = [float(x) for x in np.linspace(0.5, 10.0, 20)]


def parse_a_values(raw: Optional[str]) -> List[float]:
    if raw is None:
        return list(DEFAULT_A_VALUES)
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("--a_values_kpc was provided but no valid values were found.")
    return [float(x) for x in parts]


def ensure_radius(values: List[float], radius: float, tol: float = 1.0e-9) -> List[float]:
    """Return a sorted radius list that contains the requested radius once."""
    vals = [float(v) for v in values]
    if not any(abs(v - radius) <= tol for v in vals):
        vals.append(float(radius))
    return sorted(vals)


def radius_matches(value: float, target: float, tol: float = 1.0e-6) -> bool:
    return abs(float(value) - float(target)) <= tol


def infer_history_dir(base_dir: str, profile_tag: str, a_kpc: float) -> str:
    a_pc = a_kpc * 1.0e3
    return os.path.join(base_dir, f"event_histories_{profile_tag}_a={a_pc:.4f}")


def infer_sample_file(base_dir: str, profile_tag: str, a_kpc: float) -> Optional[str]:
    """
    Locate the Monte Carlo sample file for a given radius.

    The preferred filename is AMC_samples_a=<a_pc>_<profile_tag>.npz.
    A glob fallback is used to remain robust against small changes in suffixes.
    """
    a_pc = a_kpc * 1.0e3
    exact = os.path.join(base_dir, f"AMC_samples_a={a_pc:.4f}_{profile_tag}.npz")
    if os.path.isfile(exact):
        return exact

    pattern = os.path.join(base_dir, f"AMC_samples_a={a_pc:.4f}_*.npz")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None

    # Prefer files whose suffix contains profile_tag when possible.
    tagged = [m for m in matches if profile_tag in os.path.basename(m)]
    return tagged[0] if tagged else matches[0]


def safe_stats(x: np.ndarray, positive_only: bool = False) -> Dict[str, float]:
    """
    Return robust summary statistics for one realization-level array.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if positive_only:
        arr = arr[arr > 0.0]

    keys = ["mean", "median", "p16", "p84", "p90", "p95", "p99", "max", "n_finite", "n_positive"]
    if len(arr) == 0:
        return {k: (0 if k in ("n_finite", "n_positive") else np.nan) for k in keys}

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "n_finite": int(len(arr)),
        "n_positive": int(np.sum(arr > 0.0)),
    }


def load_encounter_summary_for_radius(
    mc_base_dir: str,
    profile_tag: str,
    a_kpc: float,
    output_root: str,
) -> Optional[Dict[str, object]]:
    """
    Load realization-level encounter-probability quantities from the Monte Carlo
    sample file and save a per-radius JSON dictionary.

    Requires the updated MonteCarlo.py that writes P_encounter_orbit,
    t_visible_orbit, and R_stream_visible into AMC_samples_*.npz.
    """
    sample_file = infer_sample_file(mc_base_dir, profile_tag, a_kpc)
    if sample_file is None:
        print(f"[WARNING] No AMC_samples file found for a={a_kpc:.3f} kpc")
        return None

    with np.load(sample_file, allow_pickle=False) as data:
        files = set(data.files)

        def arr(name: str) -> np.ndarray:
            if name not in files:
                return np.array([], dtype=float)
            return np.asarray(data[name], dtype=float)

        p_enc = arr("P_encounter_orbit")
        t_vis = arr("t_visible_orbit")
        r_stream = arr("R_stream_visible")
        n_stream = arr("n_stream_orbit")
        gamma_diff = arr("Gamma_diffuse_orbit")
        m_stream = arr("M_stream")
        sigma_t = arr("sigma_t")
        sigma_l = arr("sigma_l")
        v_rel_eff = arr("v_rel_effective_kms")

        meta = {
            "T_exp_yr": float(np.asarray(data["T_exp_yr"]).item()) if "T_exp_yr" in files else np.nan,
            "v_rel_encounter_kms": float(np.asarray(data["v_rel_encounter_kms"]).item()) if "v_rel_encounter_kms" in files else np.nan,
            "use_earth_relative_velocity": float(np.asarray(data["use_earth_relative_velocity"]).item()) if "use_earth_relative_velocity" in files else 0.0,
            "v_earth_kms": float(np.asarray(data["v_earth_kms"]).item()) if "v_earth_kms" in files else np.nan,
            "mean_mmc": float(np.asarray(data["mean_mmc"]).item()) if "mean_mmc" in files else np.nan,
            "f_amc": float(np.asarray(data["f_amc"]).item()) if "f_amc" in files else np.nan,
        }

    if len(p_enc) == 0:
        print(
            f"[WARNING] {os.path.basename(sample_file)} has no P_encounter_orbit. "
            "Use the updated MonteCarlo.py with encounter-probability output."
        )

    summary: Dict[str, object] = {
        "a_kpc": float(a_kpc),
        "a_pc": float(a_kpc * 1.0e3),
        "sample_file": sample_file,
        "metadata": meta,
        "P_encounter": safe_stats(p_enc, positive_only=False),
        "P_encounter_positive": safe_stats(p_enc, positive_only=True),
        "t_visible_s": safe_stats(t_vis, positive_only=True),
        "t_visible_yr": safe_stats(t_vis / YEAR if len(t_vis) else t_vis, positive_only=True),
        "R_stream_visible_pc": safe_stats(r_stream, positive_only=True),
        "n_stream_orbit_pc3": safe_stats(n_stream, positive_only=True),
        "Gamma_diffuse_orbit_s_inv": safe_stats(gamma_diff, positive_only=True),
        "M_stream_Msun": safe_stats(m_stream, positive_only=True),
        "sigma_t_km_s": safe_stats(sigma_t, positive_only=True),
        "sigma_l_km_s": safe_stats(sigma_l, positive_only=True),
        "v_rel_effective_km_s": safe_stats(v_rel_eff, positive_only=True),
    }

    outdir = os.path.join(output_root, "encounter_probability")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"P_encounter_summary_a={a_kpc:.2f}kpc.json")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    summary["summary_file"] = outfile
    print(f"[INFO] Saved encounter summary: {outfile}")
    return summary



def save_earth_encounter_distribution(
    mc_base_dir: str,
    profile_tag: str,
    earth_radius_kpc: float,
    output_root: str,
) -> Optional[Dict[str, str]]:
    """
    Save the raw Solar-neighborhood encounter-probability distribution.

    This is meant for the explicit Earth/Solar-circle diagnostic. It uses the
    AMC_samples file at a=earth_radius_kpc and stores the realization-level
    arrays needed to inspect the tail of P_encounter.
    """
    sample_file = infer_sample_file(mc_base_dir, profile_tag, earth_radius_kpc)
    if sample_file is None:
        print(f"[WARNING] No AMC_samples file found for Earth radius a={earth_radius_kpc:.3f} kpc")
        return None

    with np.load(sample_file, allow_pickle=False) as data:
        files = set(data.files)
        required = ["P_encounter_orbit", "n_stream_orbit", "t_visible_orbit", "R_stream_visible"]
        missing = [name for name in required if name not in files]
        if missing:
            print(f"[WARNING] Cannot save Earth encounter distribution; missing fields: {missing}")
            return None

        P = np.asarray(data["P_encounter_orbit"], dtype=float)
        n_stream = np.asarray(data["n_stream_orbit"], dtype=float)
        t_vis_s = np.asarray(data["t_visible_orbit"], dtype=float)
        R_vis_pc = np.asarray(data["R_stream_visible"], dtype=float)
        M_stream = np.asarray(data["M_stream"], dtype=float) if "M_stream" in files else np.full_like(P, np.nan)
        sigma_t = np.asarray(data["sigma_t"], dtype=float) if "sigma_t" in files else np.full_like(P, np.nan)
        sigma_l = np.asarray(data["sigma_l"], dtype=float) if "sigma_l" in files else np.full_like(P, np.nan)
        v_rel_eff = np.asarray(data["v_rel_effective_kms"], dtype=float) if "v_rel_effective_kms" in files else np.full_like(P, np.nan)
        T_exp_yr = float(np.asarray(data["T_exp_yr"]).item()) if "T_exp_yr" in files else np.nan
        v_rel_kms = float(np.asarray(data["v_rel_encounter_kms"]).item()) if "v_rel_encounter_kms" in files else np.nan
        v_earth_kms = float(np.asarray(data["v_earth_kms"]).item()) if "v_earth_kms" in files else np.nan
        use_earth_relative_velocity = int(np.asarray(data["use_earth_relative_velocity"]).item()) if "use_earth_relative_velocity" in files else 0

    outdir = os.path.join(output_root, "encounter_probability", "earth_location")
    os.makedirs(outdir, exist_ok=True)

    npz_path = os.path.join(outdir, f"P_encounter_distribution_a={earth_radius_kpc:.2f}kpc.npz")
    np.savez_compressed(
        npz_path,
        a_kpc=np.array(earth_radius_kpc, dtype=np.float64),
        T_exp_yr=np.array(T_exp_yr, dtype=np.float64),
        v_rel_encounter_kms=np.array(v_rel_kms, dtype=np.float64),
        P_encounter_orbit=P.astype(np.float64),
        n_stream_orbit=n_stream.astype(np.float64),
        t_visible_orbit_s=t_vis_s.astype(np.float64),
        t_visible_orbit_yr=(t_vis_s / YEAR).astype(np.float64),
        R_stream_visible_pc=R_vis_pc.astype(np.float64),
        M_stream_Msun=M_stream.astype(np.float64),
        sigma_t_km_s=sigma_t.astype(np.float64),
        sigma_l_km_s=sigma_l.astype(np.float64),
        v_rel_effective_km_s=v_rel_eff.astype(np.float64),
        v_earth_kms=np.array(v_earth_kms, dtype=np.float64),
        use_earth_relative_velocity=np.array(use_earth_relative_velocity, dtype=np.int8),
    )

    csv_path = os.path.join(outdir, f"P_encounter_distribution_a={earth_radius_kpc:.2f}kpc.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "P_encounter",
            "n_stream_orbit_pc^-3",
            "t_visible_yr",
            "R_stream_visible_pc",
            "M_stream_Msun",
            "sigma_t_km_s",
            "sigma_l_km_s",
            "v_rel_effective_km_s",
        ])
        for row in zip(P, n_stream, t_vis_s / YEAR, R_vis_pc, M_stream, sigma_t, sigma_l, v_rel_eff):
            writer.writerow([float(x) if np.isfinite(x) else "nan" for x in row])

    summary = {
        "a_kpc": earth_radius_kpc,
        "sample_file": sample_file,
        "distribution_npz": npz_path,
        "distribution_csv": csv_path,
        "T_exp_yr": T_exp_yr,
        "v_rel_encounter_kms": v_rel_kms,
        "v_earth_kms": v_earth_kms,
        "use_earth_relative_velocity": use_earth_relative_velocity,
        "v_rel_effective_km_s": safe_stats(v_rel_eff, positive_only=True),
        "P_encounter": safe_stats(P, positive_only=False),
        "P_encounter_positive": safe_stats(P, positive_only=True),
        "n_realizations": int(np.sum(np.isfinite(P))),
    }

    json_path = os.path.join(outdir, f"P_encounter_earth_summary_a={earth_radius_kpc:.2f}kpc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[INFO] Saved Earth/Solar-neighborhood encounter distribution:")
    print(f"       {npz_path}")
    print(f"       {csv_path}")
    print(f"       {json_path}")
    return {"npz": npz_path, "csv": csv_path, "json": json_path}

def flatten_encounter_summary(summary: Dict[str, object]) -> Dict[str, float]:
    """
    Flatten the per-radius encounter dictionary into one CSV row.
    """
    row: Dict[str, float] = {
        "a_kpc": float(summary.get("a_kpc", np.nan)),
        "a_pc": float(summary.get("a_pc", np.nan)),
    }

    meta = summary.get("metadata", {})
    if isinstance(meta, dict):
        for k, v in meta.items():
            row[k] = float(v) if np.isfinite(v) else np.nan

    for group in [
        "P_encounter",
        "P_encounter_positive",
        "t_visible_yr",
        "R_stream_visible_pc",
        "n_stream_orbit_pc3",
        "Gamma_diffuse_orbit_s_inv",
        "M_stream_Msun",
        "sigma_t_km_s",
        "sigma_l_km_s",
        "v_rel_effective_km_s",
    ]:
        stats = summary.get(group, {})
        if not isinstance(stats, dict):
            continue
        for stat_key, value in stats.items():
            row[f"{group}_{stat_key}"] = float(value) if np.isfinite(value) else np.nan

    return row


def save_global_encounter_summary(output_root: str, summaries: List[Dict[str, object]]) -> None:
    """
    Save a global JSON list and a flattened CSV table for encounter probabilities.
    """
    if not summaries:
        return

    outdir = os.path.join(output_root, "encounter_probability")
    os.makedirs(outdir, exist_ok=True)

    json_path = os.path.join(outdir, "P_encounter_all_radii.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    rows = [flatten_encounter_summary(s) for s in summaries]
    csv_path = os.path.join(outdir, "P_encounter_all_radii.csv")
    save_summary_csv(csv_path, rows)

    print(f"[INFO] Saved global encounter summaries:")
    print(f"       {json_path}")
    print(f"       {csv_path}")


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


def median_band(stacked: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust percentile bands that do not emit warnings when a whole time-slice is NaN.
    """
    stacked = np.asarray(stacked, dtype=float)
    if stacked.ndim != 2:
        raise ValueError("median_band expects a 2D array of shape (n_runs, n_times)")

    n_time = stacked.shape[1]
    med = np.full(n_time, np.nan, dtype=float)
    p16 = np.full(n_time, np.nan, dtype=float)
    p84 = np.full(n_time, np.nan, dtype=float)

    for i in range(n_time):
        col = stacked[:, i]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            continue
        med[i] = np.median(col)
        p16[i] = np.percentile(col, 16)
        p84[i] = np.percentile(col, 84)

    return med, p16, p84


def finite_values(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def save_summary_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(path: str, rows: List[Dict[str, float]]) -> None:
    n = len(rows)
    slope_cg = finite_values(rows, SLOPE_CG_KEY)
    slope_local = finite_values(rows, SLOPE_LOCAL_KEY)
    axis_ratio = finite_values(rows, "final_axis_ratio")
    retention_cg = finite_values(rows, "rho_cg_retention")
    retention_local = finite_values(rows, "rho_local_retention")
    radii = finite_values(rows, "a_kpc")
    tmin_dyn = finite_values(rows, "t_min_slope_myr")
    tmax_dyn = finite_values(rows, "t_max_slope_myr")

    def frac(key: str) -> float:
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else np.nan

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Number of processed histories: {n}\n")
        if len(radii):
            f.write(f"Median semi-major axis [kpc]: {np.median(radii):.3f}\n")
        if len(tmin_dyn):
            f.write(f"Median dynamical t_min_slope_myr: {np.median(tmin_dyn):.3f}\n")
        if len(tmax_dyn):
            f.write(f"Median dynamical t_max_slope_myr: {np.median(tmax_dyn):.3f}\n")
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


def make_safe_prefix(output_dir: str, history_file: str, used: Dict[str, int]) -> str:
    parent = os.path.basename(os.path.dirname(history_file.rstrip(os.sep)))
    stem = os.path.splitext(os.path.basename(history_file))[0]
    base = f"{parent}__{stem}" if parent else stem
    count = used.get(base, 0)
    used[base] = count + 1
    if count > 0:
        base = f"{base}__{count:03d}"
    return os.path.join(output_dir, base)


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


def plot_t3_comparison(times_myr, rho_cg_med, rho_local_med, outpath: str):
    finite = np.where(np.isfinite(rho_cg_med) & (rho_cg_med > 0) & np.isfinite(times_myr) & (times_myr > 0))[0]
    plt.figure(figsize=(7, 5))
    if len(finite):
        i0 = finite[0]
        tref = times_myr[i0]
        ref = np.full_like(times_myr, np.nan, dtype=float)
        mask = np.isfinite(times_myr) & (times_myr > 0)
        ref[mask] = rho_cg_med[i0] * (times_myr[mask] / tref) ** (-3.0)
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
        "t_min_slope_myr_median",
        "t_max_slope_myr_median",
        "median_t_cross_eff_myr_median",
        "orbital_period_myr_median",
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

        def arr(key: str) -> np.ndarray:
            vals = np.array([r.get(key, np.nan) for r in sub], dtype=float)
            return vals[np.isfinite(vals)]

        def frac(key: str) -> float:
            vals = arr(key)
            return float(np.mean(vals)) if len(vals) else np.nan

        cg = arr(SLOPE_CG_KEY)
        local = arr(SLOPE_LOCAL_KEY)
        major = arr("major_axis_growth")
        minor = arr("minor_axis_growth")
        axis_ratio = arr("final_axis_ratio")
        rho_cg_ret = arr("rho_cg_retention")
        rho_local_ret = arr("rho_local_retention")
        tmin_dyn = arr("t_min_slope_myr")
        tmax_dyn = arr("t_max_slope_myr")
        tcross_eff = arr("median_t_cross_eff_myr")
        torb = arr("orbital_period_myr")

        calib_rows.append({
            "a_kpc": a_kpc,
            "n_histories": len(sub),
            "cg_slope_window_median": float(np.median(cg)) if len(cg) else np.nan,
            "cg_slope_window_p16": float(np.percentile(cg, 16)) if len(cg) else np.nan,
            "cg_slope_window_p84": float(np.percentile(cg, 84)) if len(cg) else np.nan,
            "local_slope_window_median": float(np.median(local)) if len(local) else np.nan,
            "local_slope_window_p16": float(np.percentile(local, 16)) if len(local) else np.nan,
            "local_slope_window_p84": float(np.percentile(local, 84)) if len(local) else np.nan,
            "t_min_slope_myr_median": float(np.median(tmin_dyn)) if len(tmin_dyn) else np.nan,
            "t_max_slope_myr_median": float(np.median(tmax_dyn)) if len(tmax_dyn) else np.nan,
            "median_t_cross_eff_myr_median": float(np.median(tcross_eff)) if len(tcross_eff) else np.nan,
            "orbital_period_myr_median": float(np.median(torb)) if len(torb) else np.nan,
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


def process_histories_for_radius(
    history_dir: str,
    output_dir: str,
    max_files: int,
    total_tracers: int,
    t_end_myr: float,
    dt_myr: float,
    vcirc: float,
    rsoft: float,
    save_every: int,
    seed: int,
    t_min_slope_myr: float,
    t_max_slope_myr: float,
    save_individual: bool,
    plot_density_contours: bool,
    plot_density_sequences: bool,
    contour_bins: int,
    contour_extent_percentile: float,
    save_snapshots: bool,
    contour_max_files: int,
) -> List[Dict[str, float]]:
    history_files = list_raw_histories_in_dir(history_dir)
    if max_files > 0:
        history_files = history_files[:max_files]
    if not history_files:
        print(f"[WARNING] No raw histories found in {history_dir}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    rows: List[Dict[str, float]] = []
    rho_cg_all, rho_local_all, axis_major_all, axis_minor_all = [], [], [], []
    times_ref = None
    prefix_counts: Dict[str, int] = {}
    a_kpc = float(os.path.basename(history_dir).split("_a=")[-1]) / 1.0e3

    print(f"Found {len(history_files)} raw history files in {history_dir}")
    for i, hf in enumerate(history_files, start=1):
        print(f"  [{i}/{len(history_files)}] {os.path.basename(hf)}")
        prefix = make_safe_prefix(output_dir, hf, prefix_counts)
        try:
            res = stream.run_event_stream(
                hf,
                total_tracers=total_tracers,
                t_end_myr=t_end_myr,
                dt_myr=dt_myr,
                vcirc=vcirc,
                rsoft=rsoft,
                save_every=save_every,
                seed=seed + i - 1,
            )
        except Exception as exc:
            print(f"    Skipping due to error: {exc}")
            continue

        summ = stream.summarize_run(
            res,
            t_min_slope_myr=t_min_slope_myr,
            t_max_slope_myr=t_max_slope_myr,
        )
        summ["history_file"] = hf
        summ["a_kpc"] = a_kpc
        rows.append(summ)

        if save_individual:
            # stream.save_run in the updated stream.py accepts save_snapshots.
            # The fallback keeps compatibility with older stream.py versions.
            try:
                stream.save_run(prefix, res, save_snapshots=save_snapshots)
            except TypeError:
                stream.save_run(prefix, res)
            stream.plot_results(res, prefix)

        # Optional publication-oriented projected density maps in the stream frame.
        # These require the updated stream.py containing plot_stream_density_contour
        # and snapshots with tracer masses.
        make_contours_for_this_file = (
            plot_density_contours
            and (contour_max_files <= 0 or i <= contour_max_files)
        )
        if make_contours_for_this_file:
            if not hasattr(stream, "plot_stream_density_contour"):
                raise RuntimeError(
                    "plot_density_contours=True, but the imported stream.py does not "
                    "define plot_stream_density_contour. Use the updated stream.py."
                )
            title = rf"$a={a_kpc:.1f}\,\mathrm{{kpc}}$, $t={t_end_myr:.1f}\,\mathrm{{Myr}}$"
            stream.plot_stream_density_contour(
                res,
                prefix + "_density_contour_xy.png",
                snapshot_index=-1,
                bins=contour_bins,
                extent_percentile=contour_extent_percentile,
                title=title,
            )

        if plot_density_sequences and make_contours_for_this_file:
            if not hasattr(stream, "plot_stream_density_contour_sequence"):
                raise RuntimeError(
                    "plot_density_sequences=True, but the imported stream.py does not "
                    "define plot_stream_density_contour_sequence. Use the updated stream.py."
                )
            stream.plot_stream_density_contour_sequence(
                res,
                prefix,
                bins=contour_bins,
                extent_percentile=contour_extent_percentile,
            )

        if times_ref is None:
            times_ref = res["times"].copy()
        elif len(res["times"]) != len(times_ref) or not np.allclose(res["times"], times_ref):
            raise RuntimeError("Inconsistent time grids across runs. Use common t_end_myr/dt_myr/save_every.")

        rho_cg_all.append(res["rho_cg"])
        rho_local_all.append(res["rho_local_nn"])
        axis_major_all.append(res["axes"][:, 0])
        axis_minor_all.append(res["axes"][:, 2])

    if not rows:
        return []

    summary_csv = os.path.join(output_dir, "stream_summary.csv")
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

    np.savez_compressed(
        os.path.join(output_dir, "stream_aggregate.npz"),
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

    plot_density(times_myr, rho_cg_med, rho_cg_p16, rho_cg_p84, rho_local_med, rho_local_p16, rho_local_p84,
                 os.path.join(output_dir, "stream_density_evolution.png"))
    plot_axes(times_myr, axis_major_med, axis_major_p16, axis_major_p84, axis_minor_med, axis_minor_p16, axis_minor_p84,
              os.path.join(output_dir, "stream_axes_evolution.png"))
    plot_slope_hist(rows, os.path.join(output_dir, "stream_slope_histograms.png"))
    plot_t3_comparison(times_myr, rho_cg_med, rho_local_med, os.path.join(output_dir, "stream_t3_comparison.png"))
    write_text_summary(os.path.join(output_dir, "stream_interpretation.txt"), rows)
    write_radius_calibration_csv(os.path.join(output_dir, "stream_calibration_by_radius.csv"), rows)
    return rows



C_KMS = 299792.458
FLASH_NU_LOW_HZ = 100.0e6
FLASH_NU_HIGH_HZ = 300.0e6
FLASH_NU_HZ = (FLASH_NU_LOW_HZ * FLASH_NU_HIGH_HZ) ** 0.5
FLASH_Q = 5.0e5
ADMX_NU_LOW_HZ = 0.6e9
ADMX_NU_HIGH_HZ = 2.0e9
ADMX_NU_HZ = (ADMX_NU_LOW_HZ * ADMX_NU_HIGH_HZ) ** 0.5
ADMX_Q = 6.0e4
flash_color = "#4C72B0"
admx_color  = "#DD8452"

def _first_existing_npz_key(data, keys):
    files = set(data.files)
    for key in keys:
        if key in files:
            return key
    return None


def _safe_array_from_npz(data, key, fallback=None):
    if key in data.files:
        return np.asarray(data[key], dtype=float)
    if fallback is None:
        return np.array([], dtype=float)
    return np.asarray(fallback, dtype=float)


def _stream_delta_nu_hz(sigma_l_kms, sigma_t_kms, nu_hz):
    sigma_l_kms = np.asarray(sigma_l_kms, dtype=float)
    sigma_t_kms = np.asarray(sigma_t_kms, dtype=float)
    sigma_v2 = np.maximum(sigma_l_kms, 0.0)**2 + 2.0 * np.maximum(sigma_t_kms, 0.0)**2
    return float(nu_hz) * sigma_v2 / C_KMS**2


def plot_haloscope_linewidth_density_solar(
    sample_file: str,
    output_root: str,
    a_kpc: float = 8.5,
    density_time_myr: float = 1.0,
    max_points: int = 60000,
    seed: int = 12345,
) -> Optional[str]:
    """
    Produce a detector-facing diagnostic for the Solar-neighborhood sample.

    The plot shows the intrinsic stream linewidth, Delta nu_stream, against the
    analytic ballistic stream-density contrast rho_track(t)/rho_loc at a chosen
    reference time. Horizontal lines show the cavity bandwidths for FLASH and
    ADMX, and dotted lines show the approximate virialized-halo linewidth
    Delta nu/nu ~ 10^{-6}.

    The routine prefers linewidth arrays written by MonteCarlo_with_haloscope.py.
    If they are absent, it recomputes them from sigma_l and sigma_t.
    """
    print("[INFO] Haloscope plot style: two cavity bands only; rasterized scatter; opaque legend.")
    if sample_file is None or not os.path.isfile(sample_file):
        print(f"[WARNING] Cannot make haloscope linewidth plot; missing sample file: {sample_file}")
        return None

    with np.load(sample_file, allow_pickle=False) as d:
        required = ["M_stream", "sigma_l", "sigma_t", "rho_loc"]
        missing = [k for k in required if k not in d.files]
        if missing:
            print(f"[WARNING] Cannot make haloscope linewidth plot; missing fields: {missing}")
            return None

        M_stream = np.asarray(d["M_stream"], dtype=float)
        sigma_l = np.asarray(d["sigma_l"], dtype=float)
        sigma_t = np.asarray(d["sigma_t"], dtype=float)
        rho_loc = np.asarray(d["rho_loc"], dtype=float)
        P_enc = _safe_array_from_npz(d, "P_encounter_orbit", np.full_like(M_stream, np.nan))

        R0_key = _first_existing_npz_key(d, ["R0", "R_i", "R_initial"])
        l0_key = _first_existing_npz_key(d, ["l0", "R0", "R_i", "R_initial"])
        if R0_key is None or l0_key is None:
            print("[WARNING] Cannot make haloscope linewidth plot; missing R0/l0 or R_initial fallback.")
            return None
        R0 = np.asarray(d[R0_key], dtype=float)
        l0 = np.asarray(d[l0_key], dtype=float)

        dnu_flash = _safe_array_from_npz(d, "delta_nu_flash_hz")
        dnu_admx_low = _safe_array_from_npz(d, "delta_nu_admx_low_hz")
        dnu_admx_high = _safe_array_from_npz(d, "delta_nu_admx_high_hz")

    if len(dnu_flash) != len(M_stream):
        dnu_flash = _stream_delta_nu_hz(sigma_l, sigma_t, FLASH_NU_HZ)
    if len(dnu_admx_low) != len(M_stream):
        dnu_admx_low = _stream_delta_nu_hz(sigma_l, sigma_t, ADMX_NU_LOW_HZ)
    if len(dnu_admx_high) != len(M_stream):
        dnu_admx_high = _stream_delta_nu_hz(sigma_l, sigma_t, ADMX_NU_HIGH_HZ)

    t_s = float(density_time_myr) * 1.0e6 * YEAR
    R_s = R0 + np.maximum(sigma_t, 0.0) * t_s / 3.086e13
    l_s = l0 + np.maximum(sigma_l, 0.0) * t_s / 3.086e13
    volume = np.pi * R_s**2 * l_s
    rho_track = np.where(volume > 0.0, M_stream / volume, np.nan)
    density_contrast = rho_track / rho_loc

    valid = (
        np.isfinite(density_contrast) & (density_contrast > 0.0)
        & np.isfinite(dnu_flash) & (dnu_flash > 0.0)
        & np.isfinite(dnu_admx_low) & (dnu_admx_low > 0.0)
        & np.isfinite(dnu_admx_high) & (dnu_admx_high > 0.0)
    )
    if not np.any(valid):
        print("[WARNING] No valid realizations for haloscope linewidth plot.")
        return None

    idx = np.where(valid)[0]
    if max_points is not None and max_points > 0 and len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=int(max_points), replace=False)

    x = density_contrast[idx]
    y_flash = dnu_flash[idx]
    y_admx_low = dnu_admx_low[idx]
    y_admx_high = dnu_admx_high[idx]
    p = P_enc[idx] if len(P_enc) == len(M_stream) else np.full_like(x, np.nan)

    # Use encounter probability only to modulate point visibility; keep the
    # figure readable even when all probabilities are zero.
    alpha = np.full_like(x, 0.22, dtype=float)
    if np.any(np.isfinite(p) & (p > 0.0)):
        pp = np.where(np.isfinite(p) & (p > 0.0), p, np.nan)
        logp = np.log10(pp)
        lo, hi = np.nanpercentile(logp, [5, 99])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            alpha = 0.08 + 0.42 * np.clip((np.nan_to_num(logp, nan=lo) - lo) / (hi - lo), 0.0, 1.0)

    outdir = os.path.join(output_root, "haloscope_linewidth")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"haloscope_linewidth_vs_density_a={a_kpc:.2f}kpc.pdf")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # The scatter clouds contain many Monte Carlo realizations. Rasterize only
    # these artists, leaving axes, labels, bands, and legend as vector objects in
    # the PDF. We show one representative linewidth per experiment, evaluated at
    # the geometric-center frequency of the experimental range. The horizontal
    # shaded regions show the corresponding cavity-bandwidth ranges.
    y_flash_mid = _stream_delta_nu_hz(sigma_l[idx], sigma_t[idx], FLASH_NU_HZ)
    y_admx_mid = _stream_delta_nu_hz(sigma_l[idx], sigma_t[idx], ADMX_NU_HZ)

    ax.scatter(
        x,
        y_flash_mid,
        s=5,
        alpha=0.22,
        color=flash_color,
        linewidths=0,
        rasterized=True,
        zorder=2,
        label=rf"FLASH stream linewidth",
    )
    ax.scatter(
        x,
        y_admx_mid,
        s=5,
        alpha=0.16,
        color=admx_color,
        linewidths=0,
        rasterized=True,
        zorder=2,
        label=rf"ADMX stream linewidth",
    )

    flash_bw_low = FLASH_NU_LOW_HZ / FLASH_Q
    flash_bw_high = FLASH_NU_HIGH_HZ / FLASH_Q
    admx_bw_low = ADMX_NU_LOW_HZ / ADMX_Q
    admx_bw_high = ADMX_NU_HIGH_HZ / ADMX_Q

    ax.axhspan(
        flash_bw_low,
        flash_bw_high,
        alpha=0.18,
        color=flash_color,
        zorder=1,
        label=rf"FLASH cavity band",
    )
    ax.axhspan(
        admx_bw_low,
        admx_bw_high,
        alpha=0.18,
        color=admx_color,
        zorder=1,
        label=rf"ADMX cavity band",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"$\rho_{{\rm track}}(t={density_time_myr:g}\,\mathrm{{Myr}})/\rho_\odot$", fontsize=15)
    ax.set_ylabel(r"Intrinsic stream linewidth $\Delta\nu_{\rm stream}$ [Hz]", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    legend_handles = [

    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor=flash_color,
        markeredgecolor='none',
        markersize=9,
        label="FLASH stream linewidth",
    ),

    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor=admx_color,
        markeredgecolor='none',
        markersize=9,
        label="ADMX stream linewidth",
    ),

    Patch(
        facecolor=flash_color,
        alpha=0.18,
        label="FLASH cavity band",
    ),

    Patch(
        facecolor=admx_color,
        alpha=0.18,
        label="ADMX cavity band",
    ),
]

    ax.legend(
     handles=legend_handles,
     fontsize=9,
     loc="best",
     frameon=True,
     framealpha=1.0,
     facecolor="white",
     edgecolor="0.6",
    )
    ax.set_title(rf"Solar-neighborhood streams, $a={a_kpc:.1f}\,\mathrm{{kpc}}$", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    summary_path = os.path.join(outdir, f"haloscope_linewidth_summary_a={a_kpc:.2f}kpc.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Sample file: {sample_file}\n")
        f.write(f"Density reference time [Myr]: {density_time_myr}\n")
        f.write(f"Number of plotted realizations: {len(idx)}\n")
        f.write(f"FLASH frequency range [Hz]: {FLASH_NU_LOW_HZ:.6e} -- {FLASH_NU_HIGH_HZ:.6e}\n")
        f.write(f"FLASH cavity bandwidth range [Hz]: {flash_bw_low:.6e} -- {flash_bw_high:.6e}\n")
        f.write(f"ADMX frequency range [Hz]: {ADMX_NU_LOW_HZ:.6e} -- {ADMX_NU_HIGH_HZ:.6e}\n")
        f.write(f"ADMX cavity bandwidth range [Hz]: {admx_bw_low:.6e} -- {admx_bw_high:.6e}\n")
        for name, arr in [
            ("density_contrast", x),
            ("delta_nu_FLASH_mid_Hz", y_flash_mid),
            ("delta_nu_ADMX_mid_Hz", y_admx_mid),
        ]:
            f.write(
                f"{name}: median={np.nanmedian(arr):.6e}, "
                f"p16={np.nanpercentile(arr,16):.6e}, "
                f"p84={np.nanpercentile(arr,84):.6e}, "
                f"p99={np.nanpercentile(arr,99):.6e}\n"
            )
    print(f"[INFO] Saved haloscope linewidth-density plot: {outpath}")
    print(f"[INFO] Saved haloscope linewidth summary: {summary_path}")
    return outpath

def run_montecarlo_for_radius(
    montecarlo_script: str,
    a_kpc: float,
    n_amc: int,
    profile: str,
    galaxy_id: str,
    m_a: float,
    mass_function_id: str,
    save_event_histories: bool,
    n_history_keep: int,
    T_exp_yr: float,
    v_rel_encounter_kms: float,
    use_earth_relative_velocity: bool,
    v_earth_kms: float,
) -> None:
    cmd = [
        "python3", montecarlo_script,
        "-a", f"{a_kpc}",
        "-N", f"{n_amc}",
        "-profile", profile,
        "-galaxyID", galaxy_id,
        "-m_a", f"{m_a}",
        "-ID", mass_function_id,
    ]
    if save_event_histories:
        cmd += ["--save_event_histories", "--n_history_keep", f"{n_history_keep}"]

    # These options require the updated MonteCarlo.py that computes
    # realization-level encounter probabilities.
    cmd += [
        "--T_exp_yr", f"{T_exp_yr}",
        "--v_rel_encounter_kms", f"{v_rel_encounter_kms}",
    ]
    if use_earth_relative_velocity:
        cmd += ["--use_earth_relative_velocity", "--v_earth_kms", f"{v_earth_kms}"]

    subprocess.run(cmd, check=True)


def save_global_calibration(output_root: str, all_rows: List[Dict[str, float]]) -> None:
    save_summary_csv(os.path.join(output_root, "all_stream_summaries.csv"), all_rows)
    write_radius_calibration_csv(os.path.join(output_root, "global_stream_calibration_by_radius.csv"), all_rows)


def main():
    p = argparse.ArgumentParser(description="Unified AMC Monte Carlo + stream pipeline")
    p.add_argument(
        "--a_values_kpc",
        type=str,
        default=None,
        help="Comma-separated list of radii in kpc (default: 20-point grid from 0.5 to 10)",
    )
    p.add_argument("--mc_base_dir", type=str, default="../MC")
    p.add_argument("--profile_tag", type=str, default="PL_powerlaw", help="History directory tag, e.g. PL_powerlaw")
    p.add_argument("--montecarlo_script", type=str, default="MonteCarlo.py")
    p.add_argument("--output_root", type=str, default="pipeline_outputs")
    p.add_argument("--run_montecarlo", action="store_true")
    p.add_argument("--run_streams", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--n_amc", type=int, default=10000)
    p.add_argument("--profile", type=str, default="PL")
    p.add_argument("--galaxy_id", type=str, default="MW")
    p.add_argument("--m_a", type=float, default=5.0e-5)
    p.add_argument("--mass_function_id", type=str, default="powerlaw")
    p.add_argument("--n_history_keep", type=int, default=25)
    p.add_argument("--T_exp_yr", type=float, default=10.0,
                   help="Experiment duration used for P_encounter in the updated MonteCarlo.py.")
    p.add_argument("--use_earth_relative_velocity", action="store_true",
                   help="Forward --use_earth_relative_velocity to MonteCarlo.py so v_rel is computed from the stream bulk velocity and Solar-neighborhood velocity.")
    p.add_argument("--v_rel_encounter_kms", type=float, default=232.0,
                   help="Fallback relative stream-detector speed used for P_encounter in the updated MonteCarlo.py.")
    p.add_argument("--v_earth_kms", type=float, default=232.0,
                   help="Detector/Solar-neighborhood circular speed forwarded to MonteCarlo.py when --use_earth_relative_velocity is set.")
    p.add_argument("--skip_encounter_summary", action="store_true",
                   help="Do not harvest P_encounter statistics from AMC_samples_*.npz.")
    p.add_argument("--include_earth_radius", action="store_true",
                   help="Force inclusion of the Solar-neighborhood radius in --a_values_kpc and save its P_encounter distribution.")
    p.add_argument("--earth_radius_kpc", type=float, default=8.5,
                   help="Solar-neighborhood radius to force into the run when --include_earth_radius is used.")
    p.add_argument("--plot_only_haloscope", action="store_true",
                   help="Only make the Solar-neighborhood haloscope linewidth-vs-density plot; do not run Monte Carlo or stream reconstruction.")
    p.add_argument("--plot_haloscope_linewidth_density", action="store_true",
                   help="Make a Solar-neighborhood linewidth-vs-density plot from the updated Monte Carlo sample file.")
    p.add_argument("--haloscope_density_time_myr", type=float, default=1.0,
                   help="Reference time used to evaluate rho_track/rho_loc in the linewidth-vs-density plot.")
    p.add_argument("--haloscope_max_points", type=int, default=60000,
                   help="Maximum number of Monte Carlo realizations displayed in the haloscope scatter plot; <=0 plots all valid points.")

    p.add_argument("--max_files", type=int, default=25)
    p.add_argument("--total_tracers", type=int, default=6000)
    p.add_argument("--t_end_myr", type=float, default=10.0)
    p.add_argument("--dt_myr", type=float, default=0.01)
    p.add_argument("--vcirc", type=float, default=220.0)
    p.add_argument("--rsoft", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--t_min_slope_myr", type=float, default=0.5)
    p.add_argument("--t_max_slope_myr", type=float, default=5.0)
    p.add_argument("--save_individual", action="store_true")

    # Optional stream morphology diagnostics. These use the updated stream.py.
    p.add_argument("--plot_density_contours", action="store_true",
                   help="Save one projected density contour map per selected stream history.")
    p.add_argument("--plot_density_sequences", action="store_true",
                   help="Save a short sequence of projected density contour maps per selected stream history.")
    p.add_argument("--contour_bins", type=int, default=180,
                   help="Number of bins per axis for projected density contour maps.")
    p.add_argument("--contour_extent_percentile", type=float, default=99.5,
                   help="Percentile used to set the contour-map field of view.")
    p.add_argument("--contour_max_files", type=int, default=2,
                   help="Maximum number of histories per radius for which contour maps are saved; <=0 means all.")
    p.add_argument("--save_snapshots", action="store_true",
                   help="Store tracer snapshots in individual .npz files for later contour-map post-processing.")
    opts = p.parse_args()

    if opts.plot_only_haloscope:
        os.makedirs(opts.output_root, exist_ok=True)
        sample_file = infer_sample_file(opts.mc_base_dir, opts.profile_tag, opts.earth_radius_kpc)
        plot_haloscope_linewidth_density_solar(
            sample_file=sample_file,
            output_root=opts.output_root,
            a_kpc=opts.earth_radius_kpc,
            density_time_myr=opts.haloscope_density_time_myr,
            max_points=opts.haloscope_max_points,
        )
        print("Done.")
        return

    if not opts.run_montecarlo and not opts.run_streams:
        opts.run_montecarlo = True
        opts.run_streams = True

    a_values_kpc = parse_a_values(opts.a_values_kpc)
    if opts.a_values_kpc is None:
        print("[INFO] Using default 20-point radius grid (0.5–10 kpc)")
    if opts.include_earth_radius:
        a_values_kpc = ensure_radius(a_values_kpc, opts.earth_radius_kpc)
        print(f"[INFO] Forcing Solar-neighborhood radius a={opts.earth_radius_kpc:.3f} kpc into the run")
    print(f"[INFO] Radii used: {a_values_kpc}")

    os.makedirs(opts.output_root, exist_ok=True)
    all_rows: List[Dict[str, float]] = []
    encounter_summaries: List[Dict[str, object]] = []

    for idx, a_kpc in enumerate(a_values_kpc):
        print(f"\n=== Radius a = {a_kpc:.2f} kpc ===")
        history_dir = infer_history_dir(opts.mc_base_dir, opts.profile_tag, a_kpc)

        if opts.run_montecarlo:
            if opts.overwrite or not os.path.isdir(history_dir):
                print("[INFO] Running MonteCarlo.py ...")
                run_montecarlo_for_radius(
                    opts.montecarlo_script,
                    a_kpc,
                    opts.n_amc,
                    opts.profile,
                    opts.galaxy_id,
                    opts.m_a,
                    opts.mass_function_id,
                    True,
                    opts.n_history_keep,
                    opts.T_exp_yr,
                    opts.v_rel_encounter_kms,
                    opts.use_earth_relative_velocity,
                    opts.v_earth_kms,
                )
            else:
                print(f"[INFO] Found existing histories at {history_dir} — skipping MonteCarlo stage")

        if not opts.skip_encounter_summary:
            encounter_summary = load_encounter_summary_for_radius(
                opts.mc_base_dir,
                opts.profile_tag,
                a_kpc,
                opts.output_root,
            )
            if encounter_summary is not None:
                encounter_summaries.append(encounter_summary)

        if opts.run_streams:
            if not os.path.isdir(history_dir):
                print(f"[WARNING] Missing history directory: {history_dir}")
                continue
            outdir = os.path.join(opts.output_root, f"stream_batch_output_a={a_kpc:.2f}kpc")
            rows = process_histories_for_radius(
                history_dir=history_dir,
                output_dir=outdir,
                max_files=opts.max_files,
                total_tracers=opts.total_tracers,
                t_end_myr=opts.t_end_myr,
                dt_myr=opts.dt_myr,
                vcirc=opts.vcirc,
                rsoft=opts.rsoft,
                save_every=opts.save_every,
                seed=opts.seed + 1000 * idx,
                t_min_slope_myr=opts.t_min_slope_myr,
                t_max_slope_myr=opts.t_max_slope_myr,
                save_individual=opts.save_individual,
                plot_density_contours=opts.plot_density_contours,
                plot_density_sequences=opts.plot_density_sequences,
                contour_bins=opts.contour_bins,
                contour_extent_percentile=opts.contour_extent_percentile,
                save_snapshots=opts.save_snapshots,
                contour_max_files=opts.contour_max_files,
            )
            all_rows.extend(rows)

    if all_rows:
        save_global_calibration(opts.output_root, all_rows)
        print(f"\nSaved global calibration to: {os.path.join(opts.output_root, 'global_stream_calibration_by_radius.csv')}")

    if encounter_summaries:
        save_global_encounter_summary(opts.output_root, encounter_summaries)

    if opts.include_earth_radius and not opts.skip_encounter_summary:
        save_earth_encounter_distribution(
            opts.mc_base_dir,
            opts.profile_tag,
            opts.earth_radius_kpc,
            opts.output_root,
        )

    if opts.plot_haloscope_linewidth_density or opts.include_earth_radius:
        sample_file = infer_sample_file(opts.mc_base_dir, opts.profile_tag, opts.earth_radius_kpc)
        plot_haloscope_linewidth_density_solar(
            sample_file=sample_file,
            output_root=opts.output_root,
            a_kpc=opts.earth_radius_kpc,
            density_time_myr=opts.haloscope_density_time_myr,
            max_points=opts.haloscope_max_points,
        )

    print("Done.")


if __name__ == "__main__":
    main()
