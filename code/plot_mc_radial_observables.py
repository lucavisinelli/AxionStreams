#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


RADIUS_RE = re.compile(r"_a=([0-9]+\.[0-9]+)_")
BATCH_RE = re.compile(r"stream_batch_output_a=([0-9]+(?:\.[0-9]+)?)kpc")


def parse_radius_pc_from_filename(path: str) -> float:
    m = RADIUS_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Could not parse radius from filename: {path}")
    return float(m.group(1))


def parse_radius_from_batch_dir(path: str) -> float:
    m = BATCH_RE.search(path)
    if m is None:
        raise ValueError(f"Could not parse radius from stream batch path: {path}")
    return float(m.group(1))


def percentile_summary(x: np.ndarray, positive_only: bool = True) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if positive_only:
        x = x[x > 0.0]
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
    p_med, p_p90, p_p99, p_max = [], [], [], []

    for path in sample_files:
        a_pc = parse_radius_pc_from_filename(path)
        a_kpc.append(a_pc / 1.0e3)

        with np.load(path, allow_pickle=False) as data:
            mstream = np.asarray(data["M_stream"], dtype=float)
            sigma_l = np.asarray(data["sigma_l"], dtype=float)
            nstream = np.asarray(data["n_stream_orbit"], dtype=float) if "n_stream_orbit" in data.files else np.array([])
            penc = np.asarray(data["P_encounter_orbit"], dtype=float) if "P_encounter_orbit" in data.files else np.array([])

        mm, m16, m84 = percentile_summary(mstream, positive_only=True)
        sm, s16, s84 = percentile_summary(sigma_l, positive_only=True)
        nm, n16, n84 = percentile_summary(nstream, positive_only=True)

        penc = penc[np.isfinite(penc)]
        if len(penc):
            p_med.append(float(np.median(penc)))
            p_p90.append(float(np.percentile(penc, 90)))
            p_p99.append(float(np.percentile(penc, 99)))
            p_max.append(float(np.max(penc)))
        else:
            p_med.append(np.nan)
            p_p90.append(np.nan)
            p_p99.append(np.nan)
            p_max.append(np.nan)

        m_med.append(mm); m_p16.append(m16); m_p84.append(m84)
        s_med.append(sm); s_p16.append(s16); s_p84.append(s84)
        n_med.append(nm); n_p16.append(n16); n_p84.append(n84)

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
        "Penc_med": np.array(p_med)[order],
        "Penc_p90": np.array(p_p90)[order],
        "Penc_p99": np.array(p_p99)[order],
        "Penc_max": np.array(p_max)[order],
        "sample_files": np.array(sample_files, dtype=object)[order],
    }


def load_rates_summary_fallback(rate_files: List[str]) -> Dict[str, np.ndarray]:
    if not rate_files:
        return {"a_kpc": np.array([]), "nstream_med": np.array([]), "nstream_p16": np.array([]), "nstream_p84": np.array([])}

    a_kpc = []
    n_med, n_p16, n_p84 = [], [], []

    for path in rate_files:
        a_pc = parse_radius_pc_from_filename(path)
        a_kpc.append(a_pc / 1.0e3)
        arr = np.genfromtxt(path, delimiter=",", comments="#", autostrip=True)
        vals = arr if arr.ndim == 1 else arr[0]
        if len(vals) >= 12:
            n_med.append(float(vals[9]))
            n_p16.append(float(vals[10]))
            n_p84.append(float(vals[11]))
        else:
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


def discover_stream_calibration_files(pipeline_dir: str) -> List[str]:
    pattern = os.path.join(pipeline_dir, "stream_batch_output_a=*kpc", "stream_calibration_by_radius.csv")
    return sorted(glob.glob(pattern))


def _axis_ratio_stats_from_stream_summary(calib_path: str) -> Tuple[float, float, float]:
    """
    Compute median and central 68% interval of the final major/minor axis ratio
    from the per-history stream_summary.csv file stored next to
    stream_calibration_by_radius.csv.
    """
    summary_path = os.path.join(os.path.dirname(calib_path), "stream_summary.csv")
    if not os.path.isfile(summary_path):
        return np.nan, np.nan, np.nan

    vals = []
    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row.get("final_axis_ratio", np.nan))
            except Exception:
                val = np.nan
            if np.isfinite(val) and val > 0.0:
                vals.append(val)

    if not vals:
        return np.nan, np.nan, np.nan

    vals = np.asarray(vals, dtype=float)
    return (
        float(np.median(vals)),
        float(np.percentile(vals, 16)),
        float(np.percentile(vals, 84)),
    )


def load_stream_calibration(calib_files: List[str]) -> Dict[str, np.ndarray]:
    rows = []
    for path in calib_files:
        axis_med, axis_p16, axis_p84 = _axis_ratio_stats_from_stream_summary(path)

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                if not row.get("a_kpc"):
                    row["a_kpc"] = parse_radius_from_batch_dir(path)

                # Add realization-level 16--84% axis-ratio statistics whenever
                # possible. These are used by plot_axis_ratio_vs_radius to draw
                # the 1 sigma Monte Carlo band.
                if np.isfinite(axis_med):
                    row["final_axis_ratio_median"] = axis_med
                    row["final_axis_ratio_p16"] = axis_p16
                    row["final_axis_ratio_p84"] = axis_p84

                rows.append(row)

    if not rows:
        return {"a_kpc": np.array([])}

    keys = set().union(*(r.keys() for r in rows))
    out = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            try:
                out[k].append(float(r.get(k, np.nan)))
            except Exception:
                out[k].append(np.nan)

    a = np.array(out["a_kpc"], dtype=float)
    order = np.argsort(a)
    return {k: np.array(v, dtype=float)[order] for k, v in out.items()}


def discover_aggregate_files(pipeline_dir: str) -> Dict[float, str]:
    pattern = os.path.join(pipeline_dir, "stream_batch_output_a=*kpc", "stream_aggregate.npz")
    files = sorted(glob.glob(pattern))
    out = {}
    for path in files:
        try:
            out[parse_radius_from_batch_dir(path)] = path
        except ValueError:
            pass
    return out


def isolated_log_spike_mask(
    x: np.ndarray,
    y: np.ndarray,
    factor: float = 8.0,
    min_neighbors: int = 2,
) -> np.ndarray:
    """Return True for isolated one-bin upward spikes in a positive series.

    This is intended only for morphology diagnostics such as the final
    major/minor axis ratio, where a single pathological covariance estimate can
    produce a large isolated point. It does not smooth the data; it only masks
    isolated bins whose value is much larger than both neighboring bins.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bad = np.zeros_like(y, dtype=bool)

    finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(finite) < 3:
        return bad

    for i in range(1, len(y) - 1):
        if not finite[i]:
            continue
        if not (finite[i - 1] and finite[i + 1]):
            continue
        neighbor_scale = max(y[i - 1], y[i + 1])
        if neighbor_scale <= 0.0:
            continue
        if y[i] > factor * neighbor_scale:
            bad[i] = True

    return bad


def plot_axis_ratio_vs_radius(
    stream_cal: Dict[str, np.ndarray],
    output: str,
    filter_isolated_spikes: bool = True,
    spike_factor: float = 8.0,
    show_rejected: bool = True,
):
    """Plot the final major/minor axis ratio with robust handling of isolated spikes.

    The preferred input columns are
        final_axis_ratio_median, final_axis_ratio_p16, final_axis_ratio_p84.
    If the percentile columns are absent, the median is plotted without a band.
    """
    a = np.asarray(stream_cal["a_kpc"], dtype=float)
    y = np.asarray(stream_cal["final_axis_ratio_median"], dtype=float)

    lo = np.asarray(
        stream_cal.get("final_axis_ratio_p16", np.full_like(a, np.nan)),
        dtype=float,
    )
    hi = np.asarray(
        stream_cal.get("final_axis_ratio_p84", np.full_like(a, np.nan)),
        dtype=float,
    )

    rejected = np.zeros_like(y, dtype=bool)
    if filter_isolated_spikes:
        rejected = isolated_log_spike_mask(a, y, factor=spike_factor)

    keep = ~rejected

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    finite_band = (
        np.isfinite(a)
        & np.isfinite(lo)
        & np.isfinite(hi)
        & (lo > 0.0)
        & (hi > 0.0)
    )
    if np.any(finite_band):
        ax.fill_between(
            a[finite_band],
            lo[finite_band],
            hi[finite_band],
            alpha=0.18,
            linewidth=0,
            edgecolor="none",
            label=r"$1\sigma$",
        )

    finite_med = keep & np.isfinite(a) & np.isfinite(y) & (y > 0.0)
    if np.any(finite_med):
        ax.plot(
            a[finite_med],
            y[finite_med],
            linewidth=2.2,
            label="median",
        )

    if show_rejected and np.any(rejected):
        mask = rejected & np.isfinite(a) & np.isfinite(y) & (y > 0.0)
        ax.plot(
            a[mask],
            y[mask],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.8,
            label="isolated bin",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Semi-major axis [kpc]", fontsize=16)
    ax.set_ylabel("Major-to-minor axis ratio", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(0.0, 10.0)

    # Keep the vertical range focused on the robust trend, while retaining any
    # excluded point as an open marker if requested.
    robust_vals = y[finite_med]
    if len(robust_vals):
        ymin = max(1.0, 0.6 * np.nanmin(robust_vals))
        ymax = 2.5 * np.nanmax(robust_vals)
        ax.set_ylim(ymin, ymax)

    ax.legend(fontsize=12, frameon=True, framealpha=1.0, facecolor="white")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_with_band(x, y_med, y_lo, y_hi, xlabel, ylabel, output, ylog=True, axhline=None, axhline_label=None, label=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.asarray(x, dtype=float)
    y_med = np.asarray(y_med, dtype=float)
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)

    finite_band = np.isfinite(x) & np.isfinite(y_lo) & np.isfinite(y_hi)
    if ylog:
        finite_band &= (y_lo > 0.0) & (y_hi > 0.0)
    if np.any(finite_band):
        ax.fill_between(x[finite_band], y_lo[finite_band], y_hi[finite_band], alpha=0.18, linewidth=0, edgecolor='none')

    finite_med = np.isfinite(x) & np.isfinite(y_med)
    if ylog:
        finite_med &= y_med > 0.0
    if np.any(finite_med):
        ax.plot(x[finite_med], y_med[finite_med], linewidth=2.2, label=label)

    if axhline is not None:
        ax.axhline(axhline, linestyle="--", linewidth=1.3, color="0.35", label=axhline_label)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(0.0, 10.0)
    if label or axhline_label:
        ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight",facecolor="white",edgecolor="none")
    plt.close(fig)


def plot_two_bands(x, y1, y2, labels, xlabel, ylabel, output, ylog=False, axhline=None, axhline_label=None, legend_kwargs=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for (med, lo, hi), lab in zip([y1, y2], labels):
        med = np.asarray(med, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        finite_band = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
        if ylog:
            finite_band &= (lo > 0.0) & (hi > 0.0)
        if np.any(finite_band):
            ax.fill_between(x[finite_band], lo[finite_band], hi[finite_band], alpha=0.16, linewidth=0, edgecolor='none')
        finite_med = np.isfinite(x) & np.isfinite(med)
        if ylog:
            finite_med &= med > 0.0
        if np.any(finite_med):
            ax.plot(x[finite_med], med[finite_med], linewidth=2.2, label=lab)

    if axhline is not None:
        ax.axhline(axhline, linestyle="--", linewidth=1.4, color="0.25", label=axhline_label)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(0.0, 10.0)
    if legend_kwargs is None:
        legend_kwargs = {}
    ax.legend(**legend_kwargs)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight",facecolor="white",edgecolor="none")
    plt.close(fig)

def plot_encounter_probability(samples: Dict[str, np.ndarray], output: str):

    x = np.asarray(samples["a_kpc"], dtype=float)
    pmax = np.asarray(samples["Penc_max"], dtype=float)

    fpos = np.full_like(x, np.nan, dtype=float)

    sample_paths = samples.get("sample_files", None)

    if sample_paths is not None:

        for i, path in enumerate(sample_paths):

            try:

                with np.load(path, allow_pickle=False) as data:

                    if "P_encounter_orbit" not in data.files:
                        continue

                    P = np.asarray(
                        data["P_encounter_orbit"],
                        dtype=float,
                    )

                    P = P[np.isfinite(P)]

                    if len(P):
                        fpos[i] = np.mean(P > 0.0)

            except Exception as exc:

                print(
                    f"[WARNING] Could not read {path}: {exc}"
                )

    order = np.argsort(x)

    x = x[order]
    pmax = pmax[order]
    fpos = fpos[order]

    fig, ax1 = plt.subplots(figsize=(7.2, 5.2))

    mask_p = (
        np.isfinite(x)
        & np.isfinite(pmax)
        & (pmax > 0.0)
    )

    line1, = ax1.plot(
        x[mask_p],
        pmax[mask_p],
        marker="o",
        linewidth=2.2,
        markersize=6,
        label=r"$P_{\rm enc}^{\rm max}$",
    )

    ax1.set_yscale("log")

    ax1.set_xlabel(
        "Semi-major axis [kpc]",
        fontsize=16,
    )

    ax1.set_ylabel(
        r"Maximum encounter probability",
        fontsize=16,
    )

    ax1.tick_params(axis="both", labelsize=14)

    ax1.set_xlim(0.0, 10.0)

    ax2 = ax1.twinx()

    mask_f = (
        np.isfinite(x)
        & np.isfinite(fpos)
        & (fpos > 0.0)
    )

    line2, = ax2.plot(
        x[mask_f],
        fpos[mask_f],
        marker="s",
        linestyle="--",
        linewidth=2.2,
        markersize=6,
        label=r"$f(P_{\rm enc}>0)$",
    )

    ax2.set_yscale("log")

    ax2.set_ylabel(
        r"Fraction with $P_{\rm enc}>0$",
        fontsize=16,
    )

    ax2.tick_params(axis="both", labelsize=14)

    ax1.legend(
        handles=[line1, line2],
        fontsize=12,
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
    )

    fig.tight_layout()

    fig.savefig(
        output,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close(fig)

def _normalize_to_initial(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return t and y/y0, where y0 is the first finite positive point."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ratio = np.full_like(y, np.nan, dtype=float)
    mask0 = np.isfinite(t) & np.isfinite(y) & (t > 0.0) & (y > 0.0)
    if not np.any(mask0):
        return t, ratio
    y0 = y[np.where(mask0)[0][0]]
    good = np.isfinite(y) & (y > 0.0)
    ratio[good] = y[good] / y0
    return t, ratio


def _get_first_existing_npz_key(data, keys: List[str]) -> Optional[str]:
    """Return the first key that exists in an opened np.load object."""
    files = set(data.files)
    for k in keys:
        if k in files:
            return k
    return None


def _rho_track_logsigma_bands_from_samples(
    sample_path: str,
    times_myr: np.ndarray,
    require_disrupted: bool = False,
    pctokm: float = 3.0856775814913673e13,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute median-like log-space rho_track and 1/2-sigma bands.

    This reconstructs the same ballistic density track used by MonteCarlo.py:
        rho_track = M_stream/[pi R_s(t)^2 l_s(t)],
        R_s(t)=R0+sigma_t t, l_s(t)=l0+sigma_l t.

    The spread is computed in log10(rho_track), which is more appropriate for
    the highly skewed density distribution. The returned central curve is
    10**<log10 rho>, and the bands are 10**(<log10 rho> +/- n sigma_log).
    """
    times_myr = np.asarray(times_myr, dtype=float)
    rho_med = np.full_like(times_myr, np.nan, dtype=float)
    rho_1lo = np.full_like(times_myr, np.nan, dtype=float)
    rho_1hi = np.full_like(times_myr, np.nan, dtype=float)
    rho_2lo = np.full_like(times_myr, np.nan, dtype=float)
    rho_2hi = np.full_like(times_myr, np.nan, dtype=float)

    with np.load(sample_path, allow_pickle=False) as d:
        required = ["M_stream", "sigma_t", "sigma_l"]
        missing = [k for k in required if k not in d.files]
        if missing:
            print(f"[WARNING] {sample_path} lacks required rho_track fields: {missing}")
            return rho_med, rho_1lo, rho_1hi, rho_2lo, rho_2hi

        M_stream = np.asarray(d["M_stream"], dtype=float)
        sigma_t = np.asarray(d["sigma_t"], dtype=float)
        sigma_l = np.asarray(d["sigma_l"], dtype=float)

        R0_key = _get_first_existing_npz_key(d, ["R0", "R_i", "R_initial"])
        l0_key = _get_first_existing_npz_key(d, ["l0", "R0", "R_i", "R_initial"])
        if R0_key is None or l0_key is None:
            print(f"[WARNING] {sample_path} lacks R0/l0 or R_i fallback; cannot compute rho_track.")
            return rho_med, rho_1lo, rho_1hi, rho_2lo, rho_2hi
        R0 = np.asarray(d[R0_key], dtype=float)
        l0 = np.asarray(d[l0_key], dtype=float)

        if require_disrupted and "disrupted" in d.files:
            disrupted = np.asarray(d["disrupted"]).astype(bool)
        else:
            disrupted = np.ones_like(M_stream, dtype=bool)

    base = (
        disrupted
        & np.isfinite(M_stream) & (M_stream > 0.0)
        & np.isfinite(sigma_t) & (sigma_t >= 0.0)
        & np.isfinite(sigma_l) & (sigma_l >= 0.0)
        & np.isfinite(R0) & (R0 > 0.0)
        & np.isfinite(l0) & (l0 > 0.0)
    )
    if not np.any(base):
        print(f"[WARNING] No valid positive stream realizations in {sample_path} for rho_track.")
        return rho_med, rho_1lo, rho_1hi, rho_2lo, rho_2hi

    M = M_stream[base]
    st = sigma_t[base]
    sl = sigma_l[base]
    r0 = R0[base]
    ell0 = l0[base]

    sec_per_myr = 1.0e6 * 365.25 * 24.0 * 3600.0
    for j, tm in enumerate(times_myr):
        if not np.isfinite(tm) or tm <= 0.0:
            continue
        ts = tm * sec_per_myr
        Rs = r0 + st * ts / pctokm
        ls = ell0 + sl * ts / pctokm
        rho = M / (np.pi * Rs * Rs * ls)
        rho = rho[np.isfinite(rho) & (rho > 0.0)]
        if len(rho):
            logrho = np.log10(rho)
            mu = np.mean(logrho)
            sig = np.std(logrho)
            rho_med[j] = 10.0 ** mu
            rho_1lo[j] = 10.0 ** (mu - sig)
            rho_1hi[j] = 10.0 ** (mu + sig)
            rho_2lo[j] = 10.0 ** (mu - 2.0 * sig)
            rho_2hi[j] = 10.0 ** (mu + 2.0 * sig)
    return rho_med, rho_1lo, rho_1hi, rho_2lo, rho_2hi


def _find_sample_file_for_radius(sample_files: List[str], radius_kpc: float, tol: float = 5.0e-3) -> Optional[str]:
    candidates = []
    for path in sample_files:
        try:
            ak = parse_radius_pc_from_filename(path) / 1.0e3
        except ValueError:
            continue
        candidates.append((abs(ak - float(radius_kpc)), ak, path))
    if not candidates:
        return None
    dist, ak, path = min(candidates, key=lambda x: x[0])
    return path if dist <= tol else None


def plot_compare_aggregate(
    aggregate_files: Dict[float, str],
    radii: List[float],
    output_dir: str,
    match_tol: float = 5.0e-3,
    sample_files: Optional[List[str]] = None,
    track_radius_kpc: float = 8.5,
):
    """Compare time evolution for selected radii.

    Produces:
      * density_evolution_compare.pdf: normalized rho_cg/rho_cg(t0)
      * density_local_evolution_compare.pdf: normalized rho_local/rho_local(t0)
      * density_track_evolution_solar.pdf: absolute ballistic rho_track at a=8.5 kpc only, with median and 1/2-sigma log-space bands
      * axis_ratio_evolution_compare.pdf
    """
    if not aggregate_files:
        print("[WARNING] No stream_aggregate.npz files found. Skipping comparison plots.")
        return

    available = sorted(float(x) for x in aggregate_files.keys())
    chosen = []
    used = set()
    for r in radii:
        r = float(r)
        nearest = min(available, key=lambda x: abs(x - r))
        if abs(nearest - r) <= match_tol:
            if nearest not in used:
                chosen.append((nearest, aggregate_files[nearest]))
                used.add(nearest)
        else:
            print(
                f"[WARNING] Requested comparison radius a={r:g} kpc was not found "
                f"within tolerance {match_tol:g}. Available radii: "
                + ", ".join(f"{x:g}" for x in available)
            )

    if len(chosen) < 2:
        print("[WARNING] Fewer than two requested comparison radii were found. Skipping comparison plots.")
        return

    print("[INFO] Comparison radii used: " + ", ".join(f"{r:g} kpc" for r, _ in chosen))

    # 1) Normalized rho_cg diagnostic.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for r, path in chosen:
        with np.load(path, allow_pickle=False) as d:
            t = np.asarray(d["times_myr"], dtype=float)
            rho = np.asarray(d["rho_cg_median"], dtype=float)
        t, ratio = _normalize_to_initial(t, rho)
        mask = np.isfinite(t) & np.isfinite(ratio) & (t > 0.0) & (ratio > 0.0)
        if np.any(mask):
            ax.plot(t[mask], ratio[mask], linewidth=2.2, label=rf"$a={r:g}\,$kpc")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time [Myr]", fontsize=16)
    ax.set_ylabel(r"$\rho_{\rm cg}(t)/\rho_{\rm cg}(t_0)$", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14, loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "density_evolution_compare.pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    # 2) Normalized rho_local diagnostic. This overwrites any previous absolute local-density plot.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    any_local = False
    for r, path in chosen:
        with np.load(path, allow_pickle=False) as d:
            t = np.asarray(d["times_myr"], dtype=float)
            key = _get_first_existing_npz_key(d, ["rho_local_median", "rho_local_nn_median", "rho_local"])
            if key is None:
                print(f"[WARNING] {path} has no local-density median field; available keys: {list(d.files)}")
                continue
            rho = np.asarray(d[key], dtype=float)
        t, ratio = _normalize_to_initial(t, rho)
        mask = np.isfinite(t) & np.isfinite(ratio) & (t > 0.0) & (ratio > 0.0)
        if np.any(mask):
            ax.plot(t[mask], ratio[mask], linewidth=2.2, label=rf"$a={r:g}\,$kpc")
            any_local = True
    if any_local:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time [Myr]", fontsize=16)
        ax.set_ylabel(r"$\rho_{\rm local}(t)/\rho_{\rm local}(t_0)$", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14, loc="lower left", frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "density_local_evolution_compare.pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    else:
        print("[WARNING] No normalized rho_local curves were produced.")
    plt.close(fig)

    # 3) Absolute analytic rho_track at the Solar radius only.
    #    This is the operational density entering the t_vis and P_encounter estimates,
    #    so it is most useful for direct-detection discussion at a=8.5 kpc.
    if sample_files:
        r_track = float(track_radius_kpc)
        sample_path = _find_sample_file_for_radius(sample_files, r_track, tol=match_tol)

        agg_candidates = []
        for ak, apath in aggregate_files.items():
            agg_candidates.append((abs(float(ak) - r_track), float(ak), apath))
        agg_path = None
        matched_track_radius = None
        if agg_candidates:
            dist, matched_track_radius, agg_path_candidate = min(agg_candidates, key=lambda x: x[0])
            if dist <= match_tol:
                agg_path = agg_path_candidate

        if sample_path is None:
            print(f"[WARNING] No AMC_samples file found for rho_track at a={r_track:g} kpc.")
        elif agg_path is None:
            print(
                f"[WARNING] No stream_aggregate.npz found for rho_track time grid at "
                f"a={r_track:g} kpc within tolerance {match_tol:g}."
            )
        else:
            with np.load(agg_path, allow_pickle=False) as d_agg:
                times_myr = np.asarray(d_agg["times_myr"], dtype=float)

            med, one_lo, one_hi, two_lo, two_hi = _rho_track_logsigma_bands_from_samples(sample_path, times_myr)

            finite_med = np.isfinite(times_myr) & np.isfinite(med) & (times_myr > 0.0) & (med > 0.0)
            finite_1sig = (
                np.isfinite(times_myr)
                & np.isfinite(one_lo)
                & np.isfinite(one_hi)
                & (times_myr > 0.0)
                & (one_lo > 0.0)
                & (one_hi > 0.0)
            )
            finite_2sig = (
                np.isfinite(times_myr)
                & np.isfinite(two_lo)
                & np.isfinite(two_hi)
                & (times_myr > 0.0)
                & (two_lo > 0.0)
                & (two_hi > 0.0)
            )

            if np.any(finite_med):
                fig, ax = plt.subplots(figsize=(7.2, 5.2))
                if np.any(finite_2sig):
                    ax.fill_between(
                        times_myr[finite_2sig],
                        two_lo[finite_2sig],
                        two_hi[finite_2sig],
                        color="#9ecae1",
                        alpha=0.50,
                        linewidth=0,
                        edgecolor="none",
                        label=r"$2\sigma$",
                    )
                if np.any(finite_1sig):
                    ax.fill_between(
                        times_myr[finite_1sig],
                        one_lo[finite_1sig],
                        one_hi[finite_1sig],
                        color="#3182bd",
                        alpha=0.55,
                        linewidth=0,
                        edgecolor="none",
                        label=r"$1\sigma$",
                    )
                ax.plot(
                    times_myr[finite_med],
                    med[finite_med],
                    color="black",
                    linewidth=2.4,
                    label="median",
                )

                rho_sun = 1.19e-2
                ax.axhline(
                    rho_sun,
                    linestyle="--",
                    linewidth=1.3,
                    color="0.35",
                    label=r"$\rho_{\rm DM}(R_\odot)$",
                )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Time [Myr]", fontsize=16)
                ax.set_ylabel(r"$\rho_{\rm track}$ [$M_\odot\,{\rm pc}^{-3}$]", fontsize=16)
                ax.tick_params(axis="both", labelsize=14)
                ax.legend(fontsize=12, loc="lower left", frameon=True)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(output_dir, "density_track_evolution_solar.pdf"),
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )
                plt.close(fig)
                print(f"[INFO] Saved rho_track Solar-radius plot for a={matched_track_radius:g} kpc.")
            else:
                print(f"[WARNING] rho_track at a={r_track:g} kpc has no finite positive median values.")
    else:
        print("[WARNING] No sample files were passed; skipping density_track_evolution_solar.pdf.")

    # 4) Axis-ratio comparison.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for r, path in chosen:
        with np.load(path, allow_pickle=False) as d:
            t = np.asarray(d["times_myr"], dtype=float)
            major = np.asarray(d["axis_major_median"], dtype=float)
            minor = np.asarray(d["axis_minor_median"], dtype=float)
        ratio = major / minor
        mask = np.isfinite(t) & np.isfinite(ratio) & (t > 0.0) & (ratio > 0.0)
        if np.any(mask):
            ax.plot(t[mask], ratio[mask], linewidth=2.2, label=rf"$a={r:g}\,$kpc")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time [Myr]", fontsize=16)
    ax.set_ylabel("Axis ratio major/minor", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "axis_ratio_evolution_compare.pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_dir", default="../MC")
    parser.add_argument("--suffix", default="PL_powerlaw")
    parser.add_argument("--pipeline_dir", default="pipeline_outputs",
                        help="Directory containing stream_batch_output_a=*kpc subdirectories.")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--compare_radii", default="0.3,1.0,10.0",
                        help="Comma-separated radii for aggregate comparison plots. Default includes inner, intermediate, and outer cases.")
    parser.add_argument("--axis_ratio_spike_filter", action="store_true", default=True,
                        help="Mask isolated one-bin spikes in axis_ratio_vs_radius.pdf and show them as open markers.")
    parser.add_argument("--no_axis_ratio_spike_filter", dest="axis_ratio_spike_filter",
                        action="store_false",
                        help="Disable isolated-spike filtering for axis_ratio_vs_radius.pdf.")
    parser.add_argument("--axis_ratio_spike_factor", type=float, default=8.0,
                        help="A bin is flagged as isolated if it exceeds both neighboring bins by this factor.")
    parser.add_argument("--plot_radial_pencounter", action="store_true",
                        help="Also make Pencounter_vs_radius.pdf. Off by default because Earth-encounter probabilities are physically meaningful at the Solar radius.")
    parser.add_argument("--compare_radius_tol", type=float, default=5.0e-3,
                        help="Tolerance in kpc for matching --compare_radii to available stream_aggregate.npz directories.")
    parser.add_argument("--track_radius_kpc", type=float, default=8.5,
                        help="Radius in kpc used for the absolute rho_track plot. Default is the Solar radius, 8.5 kpc.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sample_files = discover_sample_files(args.mc_dir, args.suffix)
    rate_files = discover_rate_files(args.mc_dir, args.suffix)

    samples = load_samples_summary(sample_files)
    rates_fallback = load_rates_summary_fallback(rate_files)
    nstream = merge_nstream(samples, rates_fallback)

    plot_with_band(samples["a_kpc"], samples["sigma_l_med"], samples["sigma_l_p16"], samples["sigma_l_p84"],
                   xlabel="Semi-major axis [kpc]", ylabel=r"$\sigma_l$ [km s$^{-1}$]",
                   output=os.path.join(args.output_dir, "sigma_l_vs_radius.pdf"), ylog=True)

    plot_with_band(samples["a_kpc"], samples["M_med"], samples["M_p16"], samples["M_p84"],
                   xlabel="Semi-major axis [kpc]", ylabel=r"$M_{\rm stream}$ [$M_\odot$]",
                   output=os.path.join(args.output_dir, "Mstream_vs_radius.pdf"), ylog=True)

    plot_with_band(nstream["a_kpc"], nstream["nstream_med"], nstream["nstream_p16"], nstream["nstream_p84"],
                   xlabel="Semi-major axis [kpc]", ylabel=r"$n_{\rm stream}$ [pc$^{-3}$]",
                   output=os.path.join(args.output_dir, "nstream_vs_radius.pdf"), ylog=True)

    calib_files = discover_stream_calibration_files(args.pipeline_dir)
    if calib_files:
        stream_cal = load_stream_calibration(calib_files)
        a = stream_cal["a_kpc"]

        if "cg_slope_window_median" in stream_cal:
            plot_two_bands(
                a,
                (stream_cal["cg_slope_window_median"], stream_cal["cg_slope_window_p16"], stream_cal["cg_slope_window_p84"]),
                (stream_cal["local_slope_window_median"], stream_cal["local_slope_window_p16"], stream_cal["local_slope_window_p84"]),
                labels=(r"$\rho_{\rm cg}$", r"$\rho_{\rm local}$"),
                xlabel="Semi-major axis [kpc]",
                ylabel=r"Density slope $\gamma$",
                output=os.path.join(args.output_dir, "slope_vs_radius.pdf"),
                ylog=False,
                axhline=-3.0,
                axhline_label=r"$t^{-3}$",
                legend_kwargs={"fontsize": 18,"loc": "upper right","frameon": True},
            )

        if "final_axis_ratio_median" in stream_cal:
            plot_axis_ratio_vs_radius(
                stream_cal,
                output=os.path.join(args.output_dir, "axis_ratio_vs_radius.pdf"),
                filter_isolated_spikes=args.axis_ratio_spike_filter,
                spike_factor=args.axis_ratio_spike_factor,
                show_rejected=True,
            )

        if "rho_cg_retention_median" in stream_cal:
            plot_two_bands(
                a,
                (stream_cal["rho_cg_retention_median"], stream_cal["rho_cg_retention_median"], stream_cal["rho_cg_retention_median"]),
                (stream_cal["rho_local_retention_median"], stream_cal["rho_local_retention_median"], stream_cal["rho_local_retention_median"]),
                labels=(r"$\rho_{\rm cg}$", r"$\rho_{\rm local}$"),
                xlabel="Semi-major axis [kpc]",
                ylabel=r"Density retention $\rho_f/\rho_i$",
                output=os.path.join(args.output_dir, "density_retention_vs_radius.pdf"),
                ylog=True,
            )

        if "fraction_filamentary" in stream_cal:
            plot_with_band(
                a,
                stream_cal["fraction_filamentary"],
                stream_cal["fraction_filamentary"],
                stream_cal["fraction_filamentary"],
                xlabel="Semi-major axis [kpc]",
                ylabel="Fraction filamentary",
                output=os.path.join(args.output_dir, "fraction_filamentary_vs_radius.pdf"),
                ylog=False,
            )

    if args.plot_radial_pencounter and np.any(np.isfinite(samples["Penc_max"])):
        plot_encounter_probability(samples, os.path.join(args.output_dir, "Pencounter_vs_radius.pdf"))

    aggregate_files = discover_aggregate_files(args.pipeline_dir)
    compare_radii = [float(x.strip()) for x in args.compare_radii.split(",") if x.strip()]
    plot_compare_aggregate(
        aggregate_files,
        compare_radii,
        args.output_dir,
        match_tol=args.compare_radius_tol,
        sample_files=sample_files,
        track_radius_kpc=args.track_radius_kpc,
    )

    print("Saved plots to:", args.output_dir)


if __name__ == "__main__":
    main()
