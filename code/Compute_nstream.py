#!/usr/bin/env python3
import glob
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import dirs
import MilkyWay
import Andromeda

# -----------------------------
# PARAMETERS
# -----------------------------
a_array = np.linspace(0.5, 10.0, 20)  # kpc

N_AMC = 10000
profile = "PL"
galaxyID = "MW"
m_a = 5.0e-5
f_AMC = 1.0
M_mc_mean = 8.9e-15  # Msun
IDstr = "powerlaw"

MonteCarlo_script = "MonteCarlo.py"
output_dir = dirs.montecarlo_dir

RUN_MONTECARLO = True
OVERWRITE = True

# New: ask MonteCarlo.py to save event histories for a small subset
SAVE_EVENT_HISTORIES = True
N_HISTORY_KEEP = 25

# -----------------------------
# GALAXY SELECTION
# -----------------------------
if galaxyID == "MW":
    Galaxy = MilkyWay
elif galaxyID == "M31":
    Galaxy = Andromeda
else:
    raise ValueError("Invalid galaxyID.")

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# STORAGE
# -----------------------------
Gamma_ratio_median_list = []
Gamma_diffuse_median_list = []
Gamma_disrupt_list = []
Mstream_median_list = []
sigma_t_median_list = []
sigma_l_median_list = []

Mstream_p16_list, Mstream_p84_list = [], []
sigma_l_p16_list, sigma_l_p84_list = [], []
Gamma_ratio_p16_list, Gamma_ratio_p84_list = [], []
n_stream_median_list = []
n_stream_p16_list, n_stream_p84_list = [], []

history_dirs = []

# -----------------------------
# LOOP OVER RADII
# -----------------------------
for a0 in a_array:
    print(f"Processing a0 = {a0:.2f} kpc ...")

    a0_pc = a0 * 1e3
    pattern = os.path.join(output_dir, f"AMC_rates_a={a0_pc:.4f}_*.txt")
    sample_pattern = os.path.join(output_dir, f"AMC_samples_a={a0_pc:.4f}_*.npz")

    rate_files = glob.glob(pattern)
    sample_files = glob.glob(sample_pattern)

    need_run = OVERWRITE or (len(rate_files) == 0 or len(sample_files) == 0)

    if RUN_MONTECARLO and need_run:
        print(f"  Running MonteCarlo.py for a0 = {a0:.2f} kpc")
        cmd = [
            "python3", MonteCarlo_script,
            "-a", f"{a0}",
            "-N", f"{N_AMC}",
            "-profile", profile,
            "-galaxyID", galaxyID,
            "-m_a", f"{m_a}",
            "-ID", IDstr,
        ]
        if SAVE_EVENT_HISTORIES:
            cmd += ["--save_event_histories", "--n_history_keep", f"{N_HISTORY_KEEP}"]

        subprocess.run(cmd, check=True)

        rate_files = glob.glob(pattern)
        sample_files = glob.glob(sample_pattern)

    if SAVE_EVENT_HISTORIES:
        hist_pattern = os.path.join(output_dir, f"event_histories_*_a={a0_pc:.4f}")
        history_match = sorted(glob.glob(hist_pattern))
        if history_match:
            history_dirs.append(history_match[-1])

    if len(rate_files) == 0 or len(sample_files) == 0:
        print(f"  Warning: missing output files for a0 = {a0_pc:.4f} pc")
        Gamma_ratio_median_list.append(np.nan)
        Gamma_diffuse_median_list.append(np.nan)
        Gamma_disrupt_list.append(np.nan)
        Mstream_median_list.append(np.nan)
        sigma_t_median_list.append(np.nan)
        sigma_l_median_list.append(np.nan)

        Mstream_p16_list.append(np.nan)
        Mstream_p84_list.append(np.nan)
        sigma_l_p16_list.append(np.nan)
        sigma_l_p84_list.append(np.nan)
        Gamma_ratio_p16_list.append(np.nan)
        Gamma_ratio_p84_list.append(np.nan)

        n_stream_median_list.append(np.nan)
        n_stream_p16_list.append(np.nan)
        n_stream_p84_list.append(np.nan)
        continue

    rate_file = max(rate_files, key=os.path.getmtime)
    sample_file = max(sample_files, key=os.path.getmtime)

    rho_loc = float(Galaxy.rhoNFW(a0_pc))

    rates = np.loadtxt(rate_file, delimiter=',')
    row = rates if rates.ndim == 1 else rates[0]

    Gamma_disrupt = float(row[6])
    Gamma_disrupt_list.append(Gamma_disrupt)

    data = np.load(sample_file)

    Mstream_samples = data["M_stream"].astype(float)
    sigma_t_samples = data["sigma_t"].astype(float)
    sigma_l_samples = data["sigma_l"].astype(float)
    Gamma_diff_samples = data["Gamma_diffuse_orbit"].astype(float)

    pos_M = Mstream_samples > 0
    if np.any(pos_M):
        Mstream_median = np.median(Mstream_samples[pos_M])
        Mstream_p16, Mstream_p84 = np.percentile(Mstream_samples[pos_M], [16, 84])
    else:
        Mstream_median = np.nan
        Mstream_p16, Mstream_p84 = np.nan, np.nan

    pos_sig_t = sigma_t_samples > 0
    sigma_t_median = np.median(sigma_t_samples[pos_sig_t]) if np.any(pos_sig_t) else np.nan

    pos_sig_l = sigma_l_samples > 0
    if np.any(pos_sig_l):
        sigma_l_median = np.median(sigma_l_samples[pos_sig_l])
        sigma_l_p16, sigma_l_p84 = np.percentile(sigma_l_samples[pos_sig_l], [16, 84])
    else:
        sigma_l_median = np.nan
        sigma_l_p16, sigma_l_p84 = np.nan, np.nan

    pos_G = Gamma_diff_samples > 0
    if np.any(pos_G):
        Gamma_diff_median = np.median(Gamma_diff_samples[pos_G])

        Gamma_ratio_samples = rho_loc * Gamma_disrupt / Gamma_diff_samples[pos_G]
        Gamma_ratio_median = np.median(Gamma_ratio_samples)
        Gamma_ratio_p16, Gamma_ratio_p84 = np.percentile(Gamma_ratio_samples, [16, 84])

        n_stream_samples = f_AMC * Gamma_ratio_samples / M_mc_mean
        n_stream_median = np.median(n_stream_samples)
        n_stream_p16, n_stream_p84 = np.percentile(n_stream_samples, [16, 84])
    else:
        Gamma_diff_median = np.nan
        Gamma_ratio_median = np.nan
        Gamma_ratio_p16, Gamma_ratio_p84 = np.nan, np.nan
        n_stream_median = np.nan
        n_stream_p16, n_stream_p84 = np.nan, np.nan

    Mstream_median_list.append(Mstream_median)
    sigma_t_median_list.append(sigma_t_median)
    sigma_l_median_list.append(sigma_l_median)
    Gamma_diffuse_median_list.append(Gamma_diff_median)
    Gamma_ratio_median_list.append(Gamma_ratio_median)

    Mstream_p16_list.append(Mstream_p16)
    Mstream_p84_list.append(Mstream_p84)
    sigma_l_p16_list.append(sigma_l_p16)
    sigma_l_p84_list.append(sigma_l_p84)
    Gamma_ratio_p16_list.append(Gamma_ratio_p16)
    Gamma_ratio_p84_list.append(Gamma_ratio_p84)

    n_stream_median_list.append(n_stream_median)
    n_stream_p16_list.append(n_stream_p16)
    n_stream_p84_list.append(n_stream_p84)

# -----------------------------
# Convert to arrays
# -----------------------------
a_array = np.asarray(a_array)

Gamma_ratio_median_list = np.asarray(Gamma_ratio_median_list)
Gamma_diffuse_median_list = np.asarray(Gamma_diffuse_median_list)
Gamma_disrupt_list = np.asarray(Gamma_disrupt_list)

Mstream_median_list = np.asarray(Mstream_median_list)
sigma_t_median_list = np.asarray(sigma_t_median_list)
sigma_l_median_list = np.asarray(sigma_l_median_list)

Mstream_p16_list = np.asarray(Mstream_p16_list)
Mstream_p84_list = np.asarray(Mstream_p84_list)
sigma_l_p16_list = np.asarray(sigma_l_p16_list)
sigma_l_p84_list = np.asarray(sigma_l_p84_list)
Gamma_ratio_p16_list = np.asarray(Gamma_ratio_p16_list)
Gamma_ratio_p84_list = np.asarray(Gamma_ratio_p84_list)

n_stream_median_list = np.asarray(n_stream_median_list)
n_stream_p16_list = np.asarray(n_stream_p16_list)
n_stream_p84_list = np.asarray(n_stream_p84_list)

# -----------------------------
# Save results
# -----------------------------
results_array = np.column_stack([
    a_array,
    Gamma_ratio_median_list,
    Mstream_median_list,
    sigma_t_median_list,
    sigma_l_median_list,
    Gamma_diffuse_median_list,
    Gamma_disrupt_list,
    Gamma_ratio_p16_list,
    Gamma_ratio_p84_list,
    Mstream_p16_list,
    Mstream_p84_list,
    sigma_l_p16_list,
    sigma_l_p84_list,
    n_stream_median_list,
    n_stream_p16_list,
    n_stream_p84_list
])

np.savetxt(
    os.path.join(output_dir, "Gamma_ratio_vs_radius.txt"),
    results_array,
    header=(
        "a0_kpc, "
        "rho_loc_times_Gamma_disrupt_over_Gamma_diffuse_median, "
        "Mstream_median_Msun, sigma_t_median_km_s, sigma_l_median_km_s, "
        "Gamma_diffuse_median, Gamma_disrupt, "
        "Gamma_ratio_p16, Gamma_ratio_p84, "
        "Mstream_p16, Mstream_p84, sigma_l_p16, sigma_l_p84, "
        "n_stream_median_pc^-3, n_stream_p16, n_stream_p84"
    ),
    delimiter=", "
)

if SAVE_EVENT_HISTORIES and history_dirs:
    with open(os.path.join(output_dir, "event_history_dirs.txt"), "w") as f:
        for d in history_dirs:
            f.write(d + "\n")

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

band_color = '#1f77b4'
band_alpha = 0.25

plt.figure(figsize=(7, 5))
plt.plot(a_array, n_stream_median_list, linestyle='-', color='k')
plt.fill_between(
    a_array,
    n_stream_p16_list,
    n_stream_p84_list,
    color=band_color,
    alpha=band_alpha,
    edgecolor=band_color,
    linewidth=0.5
)
plt.margins(x=0, y=0)
plt.xlabel("Semi-major axis [kpc]")
plt.ylabel(r"$n_{\rm stream}\ [{\rm pc^{-3}}]$")
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Gamma_ratio_vs_radius.png"))
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(a_array, sigma_l_median_list, linestyle='-', color='k')
plt.fill_between(
    a_array,
    sigma_l_p16_list,
    sigma_l_p84_list,
    color=band_color,
    alpha=band_alpha,
    edgecolor=band_color,
    linewidth=0.5
)
plt.margins(x=0, y=0)
plt.xlabel("Semi-major axis [kpc]")
plt.ylabel(r"$\langle \sigma_l \rangle\ [{\rm km\,s^{-1}}]$")
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sigma_l_vs_radius.png"))
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(a_array, Mstream_median_list, linestyle='-', color='k')
plt.fill_between(
    a_array,
    Mstream_p16_list,
    Mstream_p84_list,
    color=band_color,
    alpha=band_alpha,
    edgecolor=band_color,
    linewidth=0.5
)
plt.margins(x=0, y=0)
plt.xlabel("Semi-major axis [kpc]")
plt.ylabel(r"$\langle M_{\rm stream}\rangle\ [M_\odot]$")
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Mstream_vs_radius.png"))
plt.show()
