#!/usr/bin/env python3
import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RADIUS_RE = re.compile(r"_a=([0-9]+\.[0-9]+)_")


def parse_radius_pc_from_filename(path: str) -> float:
    m = RADIUS_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Could not parse radius from filename: {path}")
    return float(m.group(1))


def percentile_summary(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(x)), float(np.percentile(x, 16)), float(np.percentile(x, 84))


def discover_sample_files(mc_dir: str, suffix: str) -> List[str]:
    pattern = os.path.join(mc_dir, f"AMC_samples_a=*_{suffix}.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No AMC_samples files found with pattern: {pattern}")
    return files


def discover_rate_files(mc_dir: str, suffix: str) -> List[str]:
    pattern = os.path.join(mc_dir, f"AMC_rates_a=*_{suffix}.txt")
    return sorted(glob.glob(pattern))


def load_samples_summary(sample_files: List[str]) -> Dict[str, np.ndarray]:
    a_kpc = []

    m_med, m_p16, m_p84 = [], [], []
    s_med, s_p16, s_p84 = [], [], []
    n_med, n_p16, n_p84 = [], [], []

    for path in sample_files:
        a_pc = parse_radius_pc_from_filename(path)
        a_kpc.append(a_pc / 1.0e3)

        data = np.load(path, allow_pickle=False)

        mstream = np.asarray(data["M_stream"], dtype=float)
        sigma_l = np.asarray(data["sigma_l"], dtype=float)

        # New preferred realization-level number density
        if "n_stream_orbit" in data.files:
            nstream = np.asarray(data["n_stream_orbit"], dtype=float)
        else:
            nstream = np.array([], dtype=float)

        # Positive-only summaries for log-like observables
        mstream = mstream[np.isfinite(mstream) & (mstream > 0.0)]
        sigma_l = sigma_l[np.isfinite(sigma_l) & (sigma_l > 0.0)]
        nstream = nstream[np.isfinite(nstream) & (nstream > 0.0)]

        mm, m16, m84 = percentile_summary(mstream)
        sm, s16, s84 = percentile_summary(sigma_l)
        nm, n16, n84 = percentile_summary(nstream)

        m_med.append(mm)
        m_p16.append(m16)
        m_p84.append(m84)

        s_med.append(sm)
        s_p16.append(s16)
        s_p84.append(s84)

        n_med.append(nm)
        n_p16.append(n16)
        n_p84.append(n84)

    order = np.argsort(a_kpc)
    return {
        "a_kpc": np.array(a_kpc)[order],
        "M_med": np.array(m_med)[order],
        "M_p16": np.array(m_p16)[order],
        "M_p84": np.array(m_p84)[order],
        "sigma_l_med": np.array(s_med)[order],
        "sigma_l_p16": np.array(s_p16)[order],
        "sigma_l_p84": np.array(s_p84)[order],
        "nstream_med": np.array(n_med)[order],
        "nstream_p16": np.array(n_p16)[order],
        "nstream_p84": np.array(n_p84)[order],
    }


def load_rates_summary_fallback(rate_files: List[str]) -> Dict[str, np.ndarray]:
    """
    Fallback for older MonteCarlo outputs that do not contain n_stream_orbit in AMC_samples.
    Supports both old and new AMC_rates formats.
    """
    if not rate_files:
        return {"a_kpc": np.array([]), "nstream_med": np.array([]), "nstream_p16": np.array([]), "nstream_p84": np.array([])}

    a_kpc = []
    n_med, n_p16, n_p84 = [], [], []

    for path in rate_files:
        a_pc = parse_radius_pc_from_filename(path)
        a_kpc.append(a_pc / 1.0e3)

        arr = np.genfromtxt(path, delimiter=",", comments="#", autostrip=True)
        vals = arr if arr.ndim == 1 else arr[0]

        # New format:
        # a_pc, N_AMC, M_stream_mean, sigma_t_mean, sigma_l_mean,
        # Gamma_diffuse, Gamma_disrupt, N_disrupt,
        # nstream_mean, nstream_median, nstream_p16, nstream_p84
        if len(vals) >= 12:
            n_med.append(float(vals[9]))
            n_p16.append(float(vals[10]))
            n_p84.append(float(vals[11]))
        else:
            # Old format has no uncertainty; keep median only
            # a_pc, N_AMC, M_stream_mean, sigma_t_mean, sigma_l_mean, Gamma_diffuse, Gamma_disrupt, N_disrupt
            n_med.append(np.nan)
            n_p16.append(np.nan)
            n_p84.append(np.nan)

    order = np.argsort(a_kpc)
    return {
        "a_kpc": np.array(a_kpc)[order],
        "nstream_med": np.array(n_med)[order],
        "nstream_p16": np.array(n_p16)[order],
        "nstream_p84": np.array(n_p84)[order],
    }


def merge_nstream(samples: Dict[str, np.ndarray], rates_fallback: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Prefer realization-level summaries from AMC_samples.
    If they are missing/NaN, fall back to AMC_rates summaries when available.
    """
    a = samples["a_kpc"].copy()
    n_med = samples["nstream_med"].copy()
    n_p16 = samples["nstream_p16"].copy()
    n_p84 = samples["nstream_p84"].copy()

    if len(rates_fallback["a_kpc"]) == 0:
        return {"a_kpc": a, "nstream_med": n_med, "nstream_p16": n_p16, "nstream_p84": n_p84}

    fallback_map = {
        float(ak): (nm, n16, n84)
        for ak, nm, n16, n84 in zip(
            rates_fallback["a_kpc"],
            rates_fallback["nstream_med"],
            rates_fallback["nstream_p16"],
            rates_fallback["nstream_p84"],
        )
    }

    for i, ak in enumerate(a):
        if np.isfinite(n_med[i]):
            continue
        key = float(ak)
        if key in fallback_map:
            nm, n16, n84 = fallback_map[key]
            n_med[i] = nm
            n_p16[i] = n16
            n_p84[i] = n84

    return {"a_kpc": a, "nstream_med": n_med, "nstream_p16": n_p16, "nstream_p84": n_p84}


def plot_with_band(
    x: np.ndarray,
    y_med: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    xlabel: str,
    ylabel: str,
    output: str,
    ylog: bool = True,
):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    finite_band = np.isfinite(x) & np.isfinite(y_lo) & np.isfinite(y_hi) & (y_lo > 0.0 if ylog else True) & (y_hi > 0.0 if ylog else True)
    if np.any(finite_band):
        ax.fill_between(x[finite_band], y_lo[finite_band], y_hi[finite_band], alpha=0.15)

    finite_med = np.isfinite(x) & np.isfinite(y_med) & (y_med > 0.0 if ylog else True)
    if np.any(finite_med):
        ax.plot(x[finite_med], y_med[finite_med], linewidth=2.2)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    if ylog:
        ax.set_yscale("log")

    ax.set_xlim(max(0.0, np.nanmin(x) - 0.1), np.nanmax(x) + 0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mc_dir",
        default="../MC",
        help="Directory containing AMC_samples_*.npz and AMC_rates_*.txt",
    )
    parser.add_argument(
        "--suffix",
        default="PL_powerlaw",
        help="Suffix used by MonteCarlo.py, e.g. PL_powerlaw",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory where the plots will be written",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sample_files = discover_sample_files(args.mc_dir, args.suffix)
    rate_files = discover_rate_files(args.mc_dir, args.suffix)

    samples = load_samples_summary(sample_files)
    rates_fallback = load_rates_summary_fallback(rate_files)
    nstream = merge_nstream(samples, rates_fallback)

    plot_with_band(
        samples["a_kpc"],
        samples["sigma_l_med"],
        samples["sigma_l_p16"],
        samples["sigma_l_p84"],
        xlabel="Semi-major axis [kpc]",
        ylabel=r"$\sigma_l$ [km s$^{-1}$]",
        output=os.path.join(args.output_dir, "sigma_l_vs_radius.png"),
        ylog=True,
    )

    plot_with_band(
        samples["a_kpc"],
        samples["M_med"],
        samples["M_p16"],
        samples["M_p84"],
        xlabel="Semi-major axis [kpc]",
        ylabel=r"$M_{\rm stream}$ [$M_\odot$]",
        output=os.path.join(args.output_dir, "Mstream_vs_radius.png"),
        ylog=True,
    )

    plot_with_band(
        nstream["a_kpc"],
        nstream["nstream_med"],
        nstream["nstream_p16"],
        nstream["nstream_p84"],
        xlabel="Semi-major axis [kpc]",
        ylabel=r"$n_{\rm stream}$ [pc$^{-3}$]",
        output=os.path.join(args.output_dir, "nstream_vs_radius.png"),
        ylog=True,
    )

    print("Saved:")
    print(os.path.join(args.output_dir, "sigma_l_vs_radius.png"))
    print(os.path.join(args.output_dir, "Mstream_vs_radius.png"))
    print(os.path.join(args.output_dir, "nstream_vs_radius.png"))


if __name__ == "__main__":
    main()
