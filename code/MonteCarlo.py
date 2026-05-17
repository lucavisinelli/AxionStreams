#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

import AMC
import perturbations as PB
import mass_function as MF
import orbits
import tools
import dirs
import MilkyWay
import Andromeda

SAVE_OUTPUT = True
VERBOSE = False
G = 4.302275e-3   # (km/s)^2 pc/Msun
YEAR = 365.25 * 24.0 * 3600.0

N_STREAM_TIMES = 50
STREAM_TMAX_YR = 1.0e8
RHO_THR_FACTOR = 10.0

# Fiducial experimental parameters for stream-encounter estimates
DEFAULT_T_EXP_YR = 10.0
DEFAULT_V_REL_ENCOUNTER_KMS = 220.0
DEFAULT_V_EARTH_KMS = 232.0
PCTOKM = 3.086e13


def stream_encounter_probability(
    n_stream_pc3: float,
    R0_pc: float,
    sigma_t_kms: float,
    t_visible_s: float,
    v_rel_kms: float = DEFAULT_V_REL_ENCOUNTER_KMS,
    T_exp_yr: float = DEFAULT_T_EXP_YR,
    cap_at_unity: bool = True,
) -> float:
    """
    Estimate the probability that a detector encounters a stream during an
    experiment of duration T_exp_yr.

    The effective stream radius is evaluated at the visibility time,

        R_s = R0 + sigma_t t_vis,

    with units converted from km/s and seconds to pc. The probability is

        P = n_stream pi R_s^2 v_rel T_exp.

    Here n_stream is the realization-level effective number density of streams
    above the adopted density threshold. This is a geometric encounter estimate,
    not a haloscope signal-to-noise calculation.
    """
    vals = [n_stream_pc3, R0_pc, sigma_t_kms, t_visible_s, v_rel_kms, T_exp_yr]
    if not all(np.isfinite(vals)):
        return 0.0
    if n_stream_pc3 <= 0.0 or R0_pc <= 0.0 or sigma_t_kms < 0.0 or t_visible_s <= 0.0:
        return 0.0
    if v_rel_kms <= 0.0 or T_exp_yr <= 0.0:
        return 0.0

    R_s_pc = R0_pc + sigma_t_kms * t_visible_s / PCTOKM
    v_rel_pcyr = v_rel_kms * YEAR / PCTOKM
    P = n_stream_pc3 * np.pi * R_s_pc**2 * v_rel_pcyr * T_exp_yr
    if not np.isfinite(P) or P < 0.0:
        return 0.0
    if cap_at_unity:
        P = min(P, 1.0)
    return float(P)


def earth_velocity_vector(v_earth_kms: float = DEFAULT_V_EARTH_KMS) -> np.ndarray:
    """
    Simple Solar-neighborhood approximation for the detector bulk velocity.

    We place the observer at Galactocentric azimuth phi=0 and take the
    local circular motion to point along +y. Annual and daily motions are
    intentionally not included here; they are small corrections to the bulk
    stream-crossing probability and are better treated in a dedicated
    time-dependent follow-up.
    """
    if not np.isfinite(v_earth_kms) or v_earth_kms <= 0.0:
        v_earth_kms = DEFAULT_V_EARTH_KMS
    return np.array([0.0, float(v_earth_kms), 0.0], dtype=np.float64)


def effective_relative_velocity_kms(
    stream_bulk_velocity: np.ndarray,
    use_earth_relative_velocity: bool,
    v_rel_fallback_kms: float = DEFAULT_V_REL_ENCOUNTER_KMS,
    v_earth_kms: float = DEFAULT_V_EARTH_KMS,
) -> float:
    """
    Return the detector-stream relative speed used in P_encounter.

    If use_earth_relative_velocity=False, the user-supplied fallback value is
    used.  If True, the relative speed is computed from the mass-weighted bulk
    velocity of the stripped debris and a simple Solar-neighborhood detector
    velocity vector.
    """
    if not use_earth_relative_velocity:
        return float(v_rel_fallback_kms) if np.isfinite(v_rel_fallback_kms) and v_rel_fallback_kms > 0.0 else DEFAULT_V_REL_ENCOUNTER_KMS

    v_stream = np.asarray(stream_bulk_velocity, dtype=float).reshape(-1)
    if v_stream.shape != (3,) or not np.all(np.isfinite(v_stream)):
        return float(v_rel_fallback_kms) if np.isfinite(v_rel_fallback_kms) and v_rel_fallback_kms > 0.0 else DEFAULT_V_REL_ENCOUNTER_KMS

    v_earth = earth_velocity_vector(v_earth_kms)
    v_rel = float(np.linalg.norm(v_stream - v_earth))
    if not np.isfinite(v_rel) or v_rel <= 0.0:
        return float(v_rel_fallback_kms) if np.isfinite(v_rel_fallback_kms) and v_rel_fallback_kms > 0.0 else DEFAULT_V_REL_ENCOUNTER_KMS
    return v_rel


def build_stream_tracks(M_stream, sigma_t, sigma_l, R0, l0, t_eval_s, pctokm=3.086e13):
    if M_stream <= 0.0 or sigma_t < 0.0 or sigma_l < 0.0 or R0 <= 0.0 or l0 <= 0.0:
        n = len(t_eval_s)
        return np.zeros(n), np.zeros(n), np.zeros(n)

    R_s = R0 + sigma_t * t_eval_s / pctokm
    l_s = l0 + sigma_l * t_eval_s / pctokm
    volume = np.pi * R_s**2 * l_s
    rho_stream = np.where(volume > 0.0, M_stream / volume, 0.0)
    return l_s, R_s, rho_stream


def scalarize_history(history: Dict[str, list], summary: Dict[str, float]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, val in history.items():
        if isinstance(val, list):
            if key == "event_index":
                out[key] = np.asarray(val, dtype=np.int32)
            else:
                out[key] = np.asarray(val, dtype=np.float32)
        else:
            if isinstance(val, (int, np.integer)):
                out[key] = np.array(val, dtype=np.int32)
            elif isinstance(val, (float, np.floating)):
                out[key] = np.array(val, dtype=np.float32)
            else:
                out[key] = np.array(str(val))
    for key, val in summary.items():
        if isinstance(val, (int, np.integer)):
            out[key] = np.array(val, dtype=np.int32)
        elif isinstance(val, (float, np.floating)):
            out[key] = np.array(val, dtype=np.float32)
        else:
            out[key] = np.array(str(val))
    return out


def approximate_launch_state(r_launch_pc: float, v_amc_kms: float, phi_launch: float):
    c = np.cos(phi_launch)
    s = np.sin(phi_launch)
    pos0 = np.array([r_launch_pc * c, r_launch_pc * s, 0.0], dtype=np.float64)
    e1 = np.array([-s, c, 0.0], dtype=np.float64)   # tangential direction
    e2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # out-of-plane direction
    e3 = np.array([c, s, 0.0], dtype=np.float64)   # radial direction
    vel0 = v_amc_kms * e1
    return pos0, vel0, e1, e2, e3


def safe_geometric_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    if len(x) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(x))))


def safe_percentiles(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(x)), float(np.percentile(x, 16)), float(np.percentile(x, 84))


def simulate_one_amc(
    j: int,
    M_init: float,
    rho_init: float,
    a: float,
    e: float,
    psi: float,
    profile: str,
    Galaxy,
    Mp: float,
    Tage: float,
    sig_rel: float,
    N_cut: int,
    t_eval_s: np.ndarray,
    mean_mmc: float,
    f_amc: float,
    T_exp_yr: float = DEFAULT_T_EXP_YR,
    v_rel_encounter_kms: float = DEFAULT_V_REL_ENCOUNTER_KMS,
    use_earth_relative_velocity: bool = False,
    v_earth_kms: float = DEFAULT_V_EARTH_KMS,
    stream_counting_mode: str = "disrupted",
    min_stream_fraction: float = 1e-3,
    save_history: bool = False,
):
    minicluster = AMC.AMC(M=M_init, rho=rho_init, profile=profile)

    M_initial = minicluster.M
    R_initial = minicluster.R
    rho_initial = minicluster.rho

    orb = orbits.elliptic_orbit(a, e, Galaxy)

    E_test = PB.Elist(sig_rel, 1.0, Mp, minicluster.M, minicluster.Rrms2())
    bmax = ((E_test / minicluster.Ebind) * N_cut) ** 0.25

    Ntotal = min(N_cut, int(PB.Ntotal_ecc(Tage, bmax, orb, psi, galaxy=Galaxy, b0=0.0)))

    n_t = 500
    tlist_orb = np.linspace(0, orb.T_orb, n_t)
    rlist = orb.calc_r(tlist_orb)
    rho_DM_list = Galaxy.rhoNFW(rlist)
    rho_loc = float(np.mean(rho_DM_list))
    n_amc_loc = float(f_amc * rho_loc / mean_mmc) if mean_mmc > 0.0 else 0.0

    if Ntotal == 0:
        Gamma_diffuse_orbit = 0.0
        disrupted = False
        n_stream_orbit = 0.0
        t_visible_orbit = 0.0
        P_encounter_orbit = 0.0
        R_stream_visible = 0.0
        v_stream_bulk = np.zeros(3, dtype=np.float64)
        v_rel_effective_kms = effective_relative_velocity_kms(
            v_stream_bulk, use_earth_relative_velocity, v_rel_encounter_kms, v_earth_kms
        )
        summary_tuple = (
            M_initial, R_initial, rho_initial,
            minicluster.M, minicluster.R, minicluster.rho,
            0, 0,
            0.0, 0.0, 0.0,
            R_initial, R_initial,
            Gamma_diffuse_orbit,
            disrupted,
            rho_loc,
            n_amc_loc,
            n_stream_orbit,
            t_visible_orbit,
            R_stream_visible,
            P_encounter_orbit,
            v_rel_effective_kms,
            v_stream_bulk[0], v_stream_bulk[1], v_stream_bulk[2],
        )
        return summary_tuple, None

    blist = PB.dPdb(bmax, Nsamples=Ntotal)
    v_amc_list, r_interaction_list = PB.dPdVamc(orb, psi, bmax, Nsamples=Ntotal, galaxy=Galaxy)
    Vlist = np.array(PB.dPdV(v_amc_list, Galaxy.sigma(r_interaction_list), Nsamples=Ntotal))

    N_enc = 0
    M_cut = 1e-25
    stripped_mass_total = 0.0
    stripped_kick_sq_weighted = 0.0
    stripped_vel_weighted = np.zeros(3, dtype=np.float64)

    history: Optional[Dict[str, list]] = None
    if save_history:
        history = {
            "event_index": [],
            "t_launch_s": [],
            "phi_launch": [],
            "r_launch_pc": [],
            "v_amc_kms": [],
            "x0_pc": [],
            "y0_pc": [],
            "z0_pc": [],
            "vx0_kms": [],
            "vy0_kms": [],
            "vz0_kms": [],
            "e1x": [],
            "e1y": [],
            "e1z": [],
            "e2x": [],
            "e2y": [],
            "e2z": [],
            "e3x": [],
            "e3y": [],
            "e3z": [],
            "M_before": [],
            "R_before": [],
            "rho_before": [],
            "dE": [],
            "dM_strip": [],
            "dv_kick": [],
            "sigma_t_event": [],
            "sigma_l_event": [],
            "b": [],
            "Vrel": [],
        }

    t_launch_list = (np.arange(Ntotal, dtype=np.float64) + 0.5) / Ntotal * Tage
    phi_launch_list = np.mod(2.0 * np.pi * t_launch_list / max(orb.T_orb, 1.0), 2.0 * np.pi)

    for i in range(Ntotal):
        if minicluster.M <= M_cut:
            break

        M_before = minicluster.M
        R_before = minicluster.R
        rho_before = minicluster.rho

        dE = PB.Elist(
            Vlist[i],
            blist[i],
            Mp,
            minicluster.M,
            minicluster.Rrms2(),
        )
        dv_kick_sq = max(0.0, 2.0 * dE / M_before)
        dv_kick = np.sqrt(dv_kick_sq)
        sigma_t_event = np.sqrt(G * M_before / R_before) if R_before > 0 else 0.0

        minicluster.perturb(dE)
        N_enc += 1

        dM_strip = max(0.0, M_before - minicluster.M)
        if dM_strip > 0.0:
            stripped_mass_total += dM_strip
            stripped_kick_sq_weighted += dM_strip * dv_kick_sq

            phi_launch = float(phi_launch_list[i])
            r_launch = float(r_interaction_list[i])
            v_launch = float(v_amc_list[i])
            pos0, vel0, e1, e2, e3 = approximate_launch_state(r_launch, v_launch, phi_launch)
            stripped_vel_weighted += dM_strip * vel0

            if history is not None:

                history["event_index"].append(i)
                history["t_launch_s"].append(t_launch_list[i])
                history["phi_launch"].append(phi_launch)
                history["r_launch_pc"].append(r_launch)
                history["v_amc_kms"].append(v_launch)
                history["x0_pc"].append(float(pos0[0]))
                history["y0_pc"].append(float(pos0[1]))
                history["z0_pc"].append(float(pos0[2]))
                history["vx0_kms"].append(float(vel0[0]))
                history["vy0_kms"].append(float(vel0[1]))
                history["vz0_kms"].append(float(vel0[2]))
                history["e1x"].append(float(e1[0]))
                history["e1y"].append(float(e1[1]))
                history["e1z"].append(float(e1[2]))
                history["e2x"].append(float(e2[0]))
                history["e2y"].append(float(e2[1]))
                history["e2z"].append(float(e2[2]))
                history["e3x"].append(float(e3[0]))
                history["e3y"].append(float(e3[1]))
                history["e3z"].append(float(e3[2]))
                history["M_before"].append(M_before)
                history["R_before"].append(R_before)
                history["rho_before"].append(rho_before)
                history["dE"].append(dE)
                history["dM_strip"].append(dM_strip)
                history["dv_kick"].append(dv_kick)
                history["sigma_t_event"].append(sigma_t_event)
                history["sigma_l_event"].append(dv_kick)
                history["b"].append(float(blist[i]))
                history["Vrel"].append(float(Vlist[i]))

        if minicluster.M <= M_cut:
            break

    f_unbound = max(0.0, 1.0 - minicluster.M / M_initial)
    M_stream = f_unbound * M_initial

    sigma_l = np.sqrt(stripped_kick_sq_weighted / stripped_mass_total) if stripped_mass_total > 0 else 0.0
    sigma_t = np.sqrt(G * M_initial / R_initial) if R_initial > 0 else 0.0

    l0 = R_initial
    R0 = R_initial

    _, _, rho_track = build_stream_tracks(M_stream, sigma_t, sigma_l, R0, l0, t_eval_s)
    rho_threshold = RHO_THR_FACTOR * rho_loc
    if M_stream > 0.0 and np.any(rho_track > rho_threshold):
        idx_last = np.where(rho_track > rho_threshold)[0][-1]
        t_visible_orbit = float(t_eval_s[idx_last])
        Gamma_diffuse_orbit = 1.0 / t_visible_orbit if t_visible_orbit > 0.0 else 0.0
    else:
        t_visible_orbit = 0.0
        Gamma_diffuse_orbit = 0.0

    disrupted = bool(minicluster.M <= M_cut)

    f_stream = M_stream / M_initial if M_initial > 0.0 else 0.0

    if stream_counting_mode == "disrupted":
        contributes_to_stream_population = disrupted
    elif stream_counting_mode == "stripped":
        contributes_to_stream_population = (M_stream > 0.0)
    elif stream_counting_mode == "threshold":
        contributes_to_stream_population = (f_stream >= min_stream_fraction)
    else:
        raise ValueError(f"Unknown stream_counting_mode={stream_counting_mode}")

    if contributes_to_stream_population and Gamma_diffuse_orbit > 0.0 and n_amc_loc > 0.0:
        n_stream_orbit = n_amc_loc / (Tage * Gamma_diffuse_orbit)
    else:
        n_stream_orbit = 0.0

    if stripped_mass_total > 0.0:
        v_stream_bulk = stripped_vel_weighted / stripped_mass_total
    else:
        v_stream_bulk = np.zeros(3, dtype=np.float64)

    v_rel_effective_kms = effective_relative_velocity_kms(
        v_stream_bulk,
        use_earth_relative_velocity,
        v_rel_fallback_kms=v_rel_encounter_kms,
        v_earth_kms=v_earth_kms,
    )

    R_stream_visible = R0 + sigma_t * t_visible_orbit / PCTOKM if t_visible_orbit > 0.0 else 0.0
    P_encounter_orbit = stream_encounter_probability(
        n_stream_orbit,
        R0,
        sigma_t,
        t_visible_orbit,
        v_rel_kms=v_rel_effective_kms,
        T_exp_yr=T_exp_yr,
        cap_at_unity=True,
    )

    summary_tuple = (
        M_initial, R_initial, rho_initial,
        minicluster.M, minicluster.R, minicluster.rho,
        Ntotal, N_enc,
        M_stream, sigma_t, sigma_l,
        R0, l0,
        Gamma_diffuse_orbit,
        disrupted,
        rho_loc,
        n_amc_loc,
        n_stream_orbit,
        t_visible_orbit,
        R_stream_visible,
        P_encounter_orbit,
        v_rel_effective_kms,
        v_stream_bulk[0], v_stream_bulk[1], v_stream_bulk[2],
    )

    if history is not None:
        summary_meta = {
            "sample_index": j,
            "M_initial": M_initial,
            "R_initial": R_initial,
            "rho_initial": rho_initial,
            "M_final": minicluster.M,
            "R_final": minicluster.R,
            "rho_final": minicluster.rho,
            "M_stream": M_stream,
            "sigma_t_global": sigma_t,
            "sigma_l_global": sigma_l,
            "Gamma_diffuse_orbit": Gamma_diffuse_orbit,
            "rho_loc": rho_loc,
            "n_amc_loc": n_amc_loc,
            "n_stream_orbit": n_stream_orbit,
            "t_visible_orbit_s": t_visible_orbit,
            "R_stream_visible_pc": R_stream_visible,
            "P_encounter_orbit": P_encounter_orbit,
            "v_rel_effective_kms": v_rel_effective_kms,
            "v_stream_bulk_x_kms": float(v_stream_bulk[0]),
            "v_stream_bulk_y_kms": float(v_stream_bulk[1]),
            "v_stream_bulk_z_kms": float(v_stream_bulk[2]),
            "T_exp_yr": T_exp_yr,
            "v_rel_encounter_kms": v_rel_encounter_kms,
            "use_earth_relative_velocity": int(use_earth_relative_velocity),
            "v_earth_kms": v_earth_kms,
            "stream_counting_mode": stream_counting_mode,
            "min_stream_fraction": min_stream_fraction,
            "stream_population_flag": int(contributes_to_stream_population),
            "a_pc": a,
            "e": e,
            "psi": psi,
            "Tage_s": Tage,
            "profile": profile,
            "galaxy": getattr(Galaxy, "__name__", "Galaxy"),
            "orbital_period_s": getattr(orb, "T_orb", np.nan),
            "launch_geometry": "planar_tangential_approximation",
        }
        return summary_tuple, scalarize_history(history, summary_meta)

    return summary_tuple, None


def Run_AMC_MonteCarlo(
    a0,
    N_AMC,
    m_a,
    profile,
    AMC_MF,
    galaxyID="MW",
    circular=False,
    IDstr="",
    save_event_histories=False,
    n_history_keep=100,
    f_amc=1.0,
    T_exp_yr=DEFAULT_T_EXP_YR,
    v_rel_encounter_kms=DEFAULT_V_REL_ENCOUNTER_KMS,
    use_earth_relative_velocity=False,
    v_earth_kms=DEFAULT_V_EARTH_KMS,
    stream_counting_mode="disrupted",
    min_stream_fraction=1e-3,
):
    if galaxyID == "MW":
        Galaxy = MilkyWay
    elif galaxyID == "M31":
        Galaxy = Andromeda
    else:
        raise ValueError("Invalid galaxyID")

    a0 *= 1e3
    Tage = 4.26e17
    N_cut = int(1e5)
    Mp = Galaxy.M_star_avg
    sig_rel = np.sqrt(2.0) * Galaxy.sigma(a0)

    t_eval_yr = np.geomspace(1.0, STREAM_TMAX_YR, N_STREAM_TIMES)
    t_eval_s = t_eval_yr * YEAR

    M_list, rho_list = AMC_MF.sample_AMCs_logflat(n_samples=N_AMC)
    mean_mmc = float(np.mean(M_list))
    e_list = np.zeros(N_AMC) if circular else PB.sample_ecc(N_AMC)
    psi_list = np.random.uniform(-np.pi / 2, np.pi / 2, size=N_AMC)

    n_history_keep = max(0, min(int(n_history_keep), int(N_AMC)))
    if save_event_histories and n_history_keep > 0:
        rng = np.random.default_rng(12345)
        chosen = rng.choice(N_AMC, size=n_history_keep, replace=False)
        history_indices = set(int(x) for x in chosen)
    else:
        history_indices = set()

    if HAVE_JOBLIB:
        results = Parallel(n_jobs=-1)(
            delayed(simulate_one_amc)(
                j, M_list[j], rho_list[j], a0, e_list[j], psi_list[j],
                profile, Galaxy, Mp, Tage, sig_rel, N_cut, t_eval_s,
                mean_mmc, f_amc,
                T_exp_yr, v_rel_encounter_kms,
                use_earth_relative_velocity, v_earth_kms,
                stream_counting_mode, min_stream_fraction,
                j in history_indices,
            )
            for j in range(N_AMC)
        )
    else:
        results = [
            simulate_one_amc(
                j, M_list[j], rho_list[j], a0, e_list[j], psi_list[j],
                profile, Galaxy, Mp, Tage, sig_rel, N_cut, t_eval_s,
                mean_mmc, f_amc,
                T_exp_yr, v_rel_encounter_kms,
                use_earth_relative_velocity, v_earth_kms,
                stream_counting_mode, min_stream_fraction,
                j in history_indices,
            )
            for j in range(N_AMC)
        ]

    summaries = np.array([r[0] for r in results], dtype=object)
    histories = [r[1] for r in results]

    M_i = np.array(summaries[:, 0], dtype=np.float64)
    R_i = np.array(summaries[:, 1], dtype=np.float64)
    rho_i = np.array(summaries[:, 2], dtype=np.float64)
    M_f = np.array(summaries[:, 3], dtype=np.float64)
    R_f = np.array(summaries[:, 4], dtype=np.float64)
    rho_f = np.array(summaries[:, 5], dtype=np.float64)
    Ntotal_list = np.array(summaries[:, 6], dtype=np.int64)
    Nenc_list = np.array(summaries[:, 7], dtype=np.int64)
    M_stream_list = np.array(summaries[:, 8], dtype=np.float64)
    sigma_t_list = np.array(summaries[:, 9], dtype=np.float64)
    sigma_l_list = np.array(summaries[:, 10], dtype=np.float64)
    R0_list = np.array(summaries[:, 11], dtype=np.float64)
    l0_list = np.array(summaries[:, 12], dtype=np.float64)
    Gamma_diff_list = np.array(summaries[:, 13], dtype=np.float64)
    disrupted_flags = np.array(summaries[:, 14], dtype=bool)
    rho_loc_list = np.array(summaries[:, 15], dtype=np.float64)
    n_amc_loc_list = np.array(summaries[:, 16], dtype=np.float64)
    n_stream_orbit_list = np.array(summaries[:, 17], dtype=np.float64)
    t_visible_orbit_list = np.array(summaries[:, 18], dtype=np.float64)
    R_stream_visible_list = np.array(summaries[:, 19], dtype=np.float64)
    P_encounter_orbit_list = np.array(summaries[:, 20], dtype=np.float64)
    v_rel_effective_list = np.array(summaries[:, 21], dtype=np.float64)
    v_stream_bulk_x_list = np.array(summaries[:, 22], dtype=np.float64)
    v_stream_bulk_y_list = np.array(summaries[:, 23], dtype=np.float64)
    v_stream_bulk_z_list = np.array(summaries[:, 24], dtype=np.float64)

    Gamma_diffuse = safe_geometric_mean(Gamma_diff_list)
    sigma_l_mean = safe_geometric_mean(sigma_l_list)
    sigma_t_mean = safe_geometric_mean(sigma_t_list)
    M_stream_mean = safe_geometric_mean(M_stream_list)

    Gamma_disrupt = np.sum(disrupted_flags) / (N_AMC * Tage)

    # Ensemble average and percentile summary of realization-level n_stream contributions
    nstream_mean = float(np.mean(n_stream_orbit_list[np.isfinite(n_stream_orbit_list)])) if np.any(np.isfinite(n_stream_orbit_list)) else 0.0
    nstream_med, nstream_p16, nstream_p84 = safe_percentiles(n_stream_orbit_list)
    p_enc_med, p_enc_p16, p_enc_p84 = safe_percentiles(P_encounter_orbit_list)
    p_enc_p90 = float(np.percentile(P_encounter_orbit_list[np.isfinite(P_encounter_orbit_list)], 90)) if np.any(np.isfinite(P_encounter_orbit_list)) else np.nan
    p_enc_p99 = float(np.percentile(P_encounter_orbit_list[np.isfinite(P_encounter_orbit_list)], 99)) if np.any(np.isfinite(P_encounter_orbit_list)) else np.nan
    p_enc_max = float(np.nanmax(P_encounter_orbit_list)) if np.any(np.isfinite(P_encounter_orbit_list)) else np.nan
    tvis_years = np.asarray(t_visible_orbit_list, dtype=float) / YEAR
    tvis_years = tvis_years[np.isfinite(tvis_years) & (tvis_years > 0.0)]
    tvis_med, tvis_p16, tvis_p84 = safe_percentiles(tvis_years)
    Rvis_med, Rvis_p16, Rvis_p84 = safe_percentiles(R_stream_visible_list)
    vrel_med, vrel_p16, vrel_p84 = safe_percentiles(v_rel_effective_list)

    if VERBOSE:
        print(f"Gamma_diffuse = {Gamma_diffuse}")
        print(f"Gamma_disrupt = {Gamma_disrupt}")
        print(f"<M_stream> = {M_stream_mean}")
        print(f"<sigma_t> = {sigma_t_mean}")
        print(f"<sigma_l> = {sigma_l_mean}")
        print(f"<n_stream> = {nstream_mean}")
        print(f"P_encounter median = {p_enc_med}")
        print(f"P_encounter p99 = {p_enc_p99}")

    if SAVE_OUTPUT:
        file_suffix = tools.generate_suffix(profile, AMC_MF, circular=circular, IDstr=IDstr, verbose=False)

        output_file = os.path.join(dirs.montecarlo_dir, f"AMC_rates_a={a0:.4f}_{file_suffix}.txt")
        header = (
            "a_pc, N_AMC, M_stream_mean_Msun, sigma_t_mean_km_s, sigma_l_mean_km_s, "
            "Gamma_diffuse, Gamma_disrupt, N_disrupt, "
            "nstream_mean_pc^-3, nstream_median_pc^-3, nstream_p16_pc^-3, nstream_p84_pc^-3, "
            "tvis_median_yr, tvis_p16_yr, tvis_p84_yr, "
            "Rstream_vis_median_pc, Rstream_vis_p16_pc, Rstream_vis_p84_pc, "
            "Penc_median, Penc_p16, Penc_p84, Penc_p90, Penc_p99, Penc_max, "
            "vrel_effective_median_km_s, vrel_effective_p16_km_s, vrel_effective_p84_km_s, "
            "T_exp_yr, v_rel_encounter_km_s, use_earth_relative_velocity, v_earth_km_s\n"
        )
        data_line = np.array([
            a0, N_AMC, M_stream_mean, sigma_t_mean, sigma_l_mean,
            Gamma_diffuse, Gamma_disrupt, np.sum(disrupted_flags),
            nstream_mean, nstream_med, nstream_p16, nstream_p84,
            tvis_med, tvis_p16, tvis_p84,
            Rvis_med, Rvis_p16, Rvis_p84,
            p_enc_med, p_enc_p16, p_enc_p84, p_enc_p90, p_enc_p99, p_enc_max,
            vrel_med, vrel_p16, vrel_p84,
            T_exp_yr, v_rel_encounter_kms, int(use_earth_relative_velocity), v_earth_kms
        ], dtype=np.float64)
        np.savetxt(output_file, data_line.reshape(1, -1), header=header, delimiter=", ")

        samples_file = os.path.join(dirs.montecarlo_dir, f"AMC_samples_a={a0:.4f}_{file_suffix}.npz")
        np.savez_compressed(
            samples_file,
            M_i=M_i.astype(np.float32),
            R_i=R_i.astype(np.float32),
            rho_i=rho_i.astype(np.float32),
            M_f=M_f.astype(np.float32),
            R_f=R_f.astype(np.float32),
            rho_f=rho_f.astype(np.float32),
            e=e_list.astype(np.float32),
            psi=psi_list.astype(np.float32),
            Ntotal=Ntotal_list.astype(np.uint32),
            Nenc=Nenc_list.astype(np.uint32),
            M_stream=M_stream_list.astype(np.float32),
            sigma_t=sigma_t_list.astype(np.float32),
            sigma_l=sigma_l_list.astype(np.float32),
            R0=R0_list.astype(np.float32),
            l0=l0_list.astype(np.float32),
            Gamma_diffuse_orbit=Gamma_diff_list.astype(np.float32),
            disrupted=disrupted_flags.astype(np.uint8),
            rho_loc=rho_loc_list.astype(np.float32),
            n_amc_loc=n_amc_loc_list.astype(np.float32),
            n_stream_orbit=n_stream_orbit_list.astype(np.float32),
            t_visible_orbit=t_visible_orbit_list.astype(np.float32),
            R_stream_visible=R_stream_visible_list.astype(np.float32),
            P_encounter_orbit=P_encounter_orbit_list.astype(np.float32),
            v_rel_effective_kms=v_rel_effective_list.astype(np.float32),
            v_stream_bulk_x_kms=v_stream_bulk_x_list.astype(np.float32),
            v_stream_bulk_y_kms=v_stream_bulk_y_list.astype(np.float32),
            v_stream_bulk_z_kms=v_stream_bulk_z_list.astype(np.float32),
            T_exp_yr=np.array(T_exp_yr, dtype=np.float32),
            v_rel_encounter_kms=np.array(v_rel_encounter_kms, dtype=np.float32),
            use_earth_relative_velocity=np.array(int(use_earth_relative_velocity), dtype=np.int8),
            v_earth_kms=np.array(v_earth_kms, dtype=np.float32),
            mean_mmc=np.array(mean_mmc, dtype=np.float32),
            f_amc=np.array(f_amc, dtype=np.float32),
        )

        if save_event_histories:
            hist_dir = os.path.join(dirs.montecarlo_dir, f"event_histories_{file_suffix}_a={a0:.4f}")
            os.makedirs(hist_dir, exist_ok=True)

            saved = 0
            for j in sorted(history_indices):
                hist = histories[j]
                if hist is None:
                    continue
                if "dM_strip" in hist and len(hist["dM_strip"]) == 0:
                    continue
                np.savez_compressed(
                    os.path.join(hist_dir, f"AMC_history_{j:06d}.npz"),
                    **hist
                )
                saved += 1

            if VERBOSE or save_event_histories:
                print(f"Saved {saved} event histories to {hist_dir}")

    return {
        "Gamma_disrupt": Gamma_disrupt,
        "Gamma_diffuse": Gamma_diffuse,
        "M_stream_mean": M_stream_mean,
        "sigma_t_mean": sigma_t_mean,
        "sigma_l_mean": sigma_l_mean,
        "N_disrupt": int(np.sum(disrupted_flags)),
        "nstream_mean": nstream_mean,
        "nstream_median": nstream_med,
        "nstream_p16": nstream_p16,
        "nstream_p84": nstream_p84,
        "P_encounter_median": p_enc_med,
        "P_encounter_p90": p_enc_p90,
        "P_encounter_p99": p_enc_p99,
        "P_encounter_max": p_enc_max,
        "tvis_median_yr": tvis_med,
        "Rstream_vis_median_pc": Rvis_med,
        "vrel_effective_median_kms": vrel_med,
        "vrel_effective_p16_kms": vrel_p16,
        "vrel_effective_p84_kms": vrel_p84,
        "use_earth_relative_velocity": int(use_earth_relative_velocity),
        "v_earth_kms": v_earth_kms,
    }


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--semi_major_axis", type=float, required=True)
    parser.add_argument("-N", "--AMC_number", type=int, default=100000)
    parser.add_argument("-profile", "--profile", type=str, default="PL")
    parser.add_argument("-galaxyID", "--galaxyID", type=str, default="MW")
    parser.add_argument("-m_a", "--m_a", type=float, default=50e-6)
    parser.add_argument("-ID", "--mass_function_ID", type=str, default="delta_c")
    parser.add_argument("-circ", "--circular", action="store_true")
    parser.add_argument("-IDstr", "--IDstr", type=str, default="")
    parser.add_argument("--save_event_histories", action="store_true")
    parser.add_argument("--n_history_keep", type=int, default=100)
    parser.add_argument("--f_amc", type=float, default=1.0)
    parser.add_argument("--T_exp_yr", type=float, default=DEFAULT_T_EXP_YR,
                        help="Experiment duration in years for stream encounter probability")
    parser.add_argument("--v_rel_encounter_kms", type=float, default=DEFAULT_V_REL_ENCOUNTER_KMS,
                        help="Fallback relative detector-stream speed in km/s for encounter probability")
    parser.add_argument("--use_earth_relative_velocity", action="store_true",
                        help="Compute v_rel realization-by-realization from the stream bulk velocity and a simple Earth/Solar-neighborhood velocity vector.")
    parser.add_argument("--v_earth_kms", type=float, default=DEFAULT_V_EARTH_KMS,
                        help="Magnitude of the detector/Solar-neighborhood circular velocity used when --use_earth_relative_velocity is set.")
    parser.add_argument("--stream_counting_mode", type=str, default="disrupted",
                        choices=["disrupted", "stripped", "threshold"],
                        help="Criterion used to include streams in the encounter-probability population.")
    parser.add_argument("--min_stream_fraction", type=float, default=1e-3,
                        help="Minimum stripped mass fraction used when stream_counting_mode=threshold.")
    return parser.parse_args(args)


if __name__ == "__main__":
    opts = getOptions()
    AMC_MF = MF.get_mass_function(opts.mass_function_ID, opts.m_a, opts.profile)
    AMC_MF.label = opts.mass_function_ID
    rates = Run_AMC_MonteCarlo(
        opts.semi_major_axis,
        opts.AMC_number,
        opts.m_a,
        opts.profile,
        AMC_MF,
        opts.galaxyID,
        opts.circular,
        opts.IDstr,
        opts.save_event_histories,
        opts.n_history_keep,
        opts.f_amc,
        opts.T_exp_yr,
        opts.v_rel_encounter_kms,
        opts.use_earth_relative_velocity,
        opts.v_earth_kms,
        opts.stream_counting_mode,
        opts.min_stream_fraction,
    )
    print(rates)
