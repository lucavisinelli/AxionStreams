#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

GN = 4.301e-3   # (km/s)^2 pc / Msun
PCTOKM = 3.086e13
YEAR = 365.25 * 24.0 * 3600.0

# Numerical safety floors
MIN_R0_PC = 1.0e-9
MIN_SIGMA_KMS = 0.0
MIN_AXIS_EIG = 1.0e-30
MAX_BASIS_COND = 1.0e8
MIN_EFFECTIVE_SIGMA = 1.0e-12


@dataclass
class Packet:
    t_launch: float     # seconds
    mass: float         # Msun
    pos0: np.ndarray    # pc
    vel0: np.ndarray    # km/s
    sigma_t: float      # km/s
    sigma_l: float      # km/s
    R0: float           # pc
    basis: Optional[np.ndarray] = None  # shape (3,3), rows are e1,e2,e3


def myr_to_pc_over_kms(myr: float) -> float:
    sec = myr * 1e6 * YEAR
    return sec / PCTOKM


def seconds_to_pc_over_kms(sec: float) -> float:
    return sec / PCTOKM


def sec_to_myr(sec: float) -> float:
    return float(sec) / (1.0e6 * YEAR)


def to_myr(times_pc_over_kms: np.ndarray) -> np.ndarray:
    return np.asarray(times_pc_over_kms) * PCTOKM / (1e6 * YEAR)


def sanitize_scalar(value: float, default: float, min_value: Optional[float] = None) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    if min_value is not None and v < min_value:
        return default
    return v


def sanitize_vec3(vec: np.ndarray, default: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return np.zeros(3, dtype=float) if default is None else np.asarray(default, dtype=float)
    return arr


def mw_acceleration_loghalo(pos: np.ndarray, vc: float = 220.0, rsoft: float = 0.05) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N,3)")
    r2 = np.sum(pos**2, axis=1) + float(rsoft)**2
    r2 = np.where(np.isfinite(r2) & (r2 > 0.0), r2, np.inf)
    acc = -(float(vc)**2) * pos / r2[:, None]
    acc[~np.isfinite(acc)] = 0.0
    return acc


def leapfrog_step(pos: np.ndarray, vel: np.ndarray, dt: float, acc_func) -> Tuple[np.ndarray, np.ndarray]:
    a0 = acc_func(pos)
    vel_half = vel + 0.5 * a0 * dt
    pos_new = pos + vel_half * dt
    a1 = acc_func(pos_new)
    vel_new = vel_half + 0.5 * a1 * dt
    pos_new = np.where(np.isfinite(pos_new), pos_new, 0.0)
    vel_new = np.where(np.isfinite(vel_new), vel_new, 0.0)
    return pos_new, vel_new


def covariance_eigs(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.array([np.nan, np.nan, np.nan])
    if not np.all(np.isfinite(points)):
        points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return np.array([np.nan, np.nan, np.nan])
    xc = points - np.mean(points, axis=0)
    cov = np.cov(xc.T)
    if not np.all(np.isfinite(cov)):
        return np.array([np.nan, np.nan, np.nan])
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(vals)[::-1]
    return vals


def coarse_grained_density(pos: np.ndarray, total_mass: float) -> float:
    if len(pos) < 10 or not np.isfinite(total_mass) or total_mass <= 0.0:
        return np.nan
    vals = covariance_eigs(pos)
    if np.any(~np.isfinite(vals)) or np.any(vals <= MIN_AXIS_EIG):
        return np.nan
    sigmas = np.sqrt(vals)
    volume = 4.0 * np.pi * sigmas[0] * sigmas[1] * sigmas[2] / 3.0
    if not np.isfinite(volume) or volume <= 0.0:
        return np.nan
    return float(total_mass / volume)


def nearest_neighbor_local_density(pos: np.ndarray, masses: np.ndarray, k: int = 32) -> float:
    pos = np.asarray(pos, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n = len(pos)
    if n < max(8, k + 1) or len(masses) != n:
        return np.nan
    total_mass = np.sum(masses)
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        return np.nan
    finite_mask = np.all(np.isfinite(pos), axis=1) & np.isfinite(masses) & (masses >= 0.0)
    pos = pos[finite_mask]
    masses = masses[finite_mask]
    n = len(pos)
    if n < max(8, k + 1):
        return np.nan
    diff = pos[:, None, :] - pos[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    kth = np.partition(dist2, k - 1, axis=1)[:, k - 1]
    rk = np.sqrt(kth)
    vol = (4.0 / 3.0) * np.pi * rk**3
    mbar = np.mean(masses)
    rho = np.where((vol > 0.0) & np.isfinite(vol), k * mbar / vol, np.nan)
    return float(np.nanmax(rho)) if np.any(np.isfinite(rho)) else np.nan


def load_history(history_file: str) -> Dict[str, np.ndarray]:
    data = np.load(history_file, allow_pickle=False)
    return {k: data[k] for k in data.files}


def tangent_basis_from_velocity(vel0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vel0 = sanitize_vec3(vel0, default=np.array([1.0, 0.0, 0.0]))
    vnorm = np.linalg.norm(vel0)
    if not np.isfinite(vnorm) or vnorm <= 0.0:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = vel0 / vnorm

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(e1, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    e2 = np.cross(e1, ref)
    e2n = np.linalg.norm(e2)
    if not np.isfinite(e2n) or e2n <= 0.0:
        e2 = np.array([0.0, 1.0, 0.0])
        e2n = 1.0
    e2 /= e2n

    e3 = np.cross(e1, e2)
    e3n = np.linalg.norm(e3)
    if not np.isfinite(e3n) or e3n <= 0.0:
        e3 = np.array([0.0, 0.0, 1.0])
        e3n = 1.0
    e3 /= e3n

    return e1, e2, e3


def orthonormalize_basis(trial: np.ndarray) -> Optional[np.ndarray]:
    if trial is None:
        return None
    trial = np.asarray(trial, dtype=float)
    if trial.shape != (3, 3) or not np.all(np.isfinite(trial)):
        return None
    norms = np.linalg.norm(trial, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        return None
    trial = trial / norms[:, None]
    try:
        q, _ = np.linalg.qr(trial.T)
        basis = q.T
    except np.linalg.LinAlgError:
        return None
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        return None
    try:
        if np.linalg.cond(basis) > MAX_BASIS_COND:
            return None
    except np.linalg.LinAlgError:
        return None
    return basis


def basis_from_history(hist: Dict[str, np.ndarray], idx: int) -> Optional[np.ndarray]:
    needed = {
        "e1x", "e1y", "e1z",
        "e2x", "e2y", "e2z",
        "e3x", "e3y", "e3z",
    }
    if not needed.issubset(hist.keys()):
        return None
    trial = np.array([
        [hist["e1x"][idx], hist["e1y"][idx], hist["e1z"][idx]],
        [hist["e2x"][idx], hist["e2y"][idx], hist["e2z"][idx]],
        [hist["e3x"][idx], hist["e3y"][idx], hist["e3z"][idx]],
    ], dtype=float)
    return orthonormalize_basis(trial)


def reconstruct_launch_state_from_history(hist: Dict[str, np.ndarray], idx: int) -> Tuple[np.ndarray, np.ndarray]:
    explicit_keys = {"x0_pc", "y0_pc", "z0_pc", "vx0_kms", "vy0_kms", "vz0_kms"}
    if explicit_keys.issubset(hist.keys()):
        pos0 = np.array([hist["x0_pc"][idx], hist["y0_pc"][idx], hist["z0_pc"][idx]], dtype=float)
        vel0 = np.array([hist["vx0_kms"][idx], hist["vy0_kms"][idx], hist["vz0_kms"][idx]], dtype=float)
        return sanitize_vec3(pos0), sanitize_vec3(vel0)

    required = {"r_launch_pc", "v_amc_kms"}
    if not required.issubset(hist.keys()):
        raise RuntimeError(
            "History file lacks both explicit launch-state fields "
            "(x0_pc, ..., vz0_kms) and reconstruction fields "
            "(r_launch_pc, v_amc_kms)."
        )

    r = sanitize_scalar(hist["r_launch_pc"][idx], default=0.0, min_value=0.0)
    v = sanitize_scalar(hist["v_amc_kms"][idx], default=0.0, min_value=0.0)
    phi = sanitize_scalar(hist["phi_launch"][idx], default=0.0) if "phi_launch" in hist else 0.0

    c, s = np.cos(phi), np.sin(phi)
    pos0 = np.array([r * c, r * s, 0.0], dtype=float)
    tangential = np.array([-s, c, 0.0], dtype=float)
    vel0 = v * tangential
    return sanitize_vec3(pos0), sanitize_vec3(vel0)


def packet_from_history(hist: Dict[str, np.ndarray], idx: int) -> Packet:
    pos0, vel0 = reconstruct_launch_state_from_history(hist, idx)
    basis = basis_from_history(hist, idx)

    if "sigma_l_event" in hist:
        sigma_l = sanitize_scalar(hist["sigma_l_event"][idx], default=0.0, min_value=MIN_SIGMA_KMS)
    elif "dv_kick" in hist:
        sigma_l = sanitize_scalar(hist["dv_kick"][idx], default=0.0, min_value=MIN_SIGMA_KMS)
    else:
        sigma_l = 0.0

    sigma_t = sanitize_scalar(hist["sigma_t_event"][idx], default=0.0, min_value=MIN_SIGMA_KMS) if "sigma_t_event" in hist else 0.0

    if "R_before" in hist:
        R0 = sanitize_scalar(hist["R_before"][idx], default=MIN_R0_PC, min_value=MIN_R0_PC)
    elif "R0" in hist:
        R0 = sanitize_scalar(hist["R0"][idx], default=MIN_R0_PC, min_value=MIN_R0_PC)
    else:
        R0 = MIN_R0_PC

    mass = sanitize_scalar(hist["dM_strip"][idx], default=0.0, min_value=0.0)

    return Packet(
        t_launch=sanitize_scalar(hist["t_launch_s"][idx], default=0.0),
        mass=mass,
        pos0=pos0,
        vel0=vel0,
        sigma_t=sigma_t,
        sigma_l=sigma_l,
        R0=R0,
        basis=basis,
    )


def spawn_packet_tracers(packet: Packet, n_tracers: int, rng: np.random.Generator):
    R0 = sanitize_scalar(packet.R0, default=MIN_R0_PC, min_value=MIN_R0_PC)
    sigma_l = sanitize_scalar(packet.sigma_l, default=0.0, min_value=0.0)
    sigma_t = sanitize_scalar(packet.sigma_t, default=0.0, min_value=0.0)

    pos0 = sanitize_vec3(packet.pos0)
    vel0 = sanitize_vec3(packet.vel0)

    basis = orthonormalize_basis(packet.basis)
    if basis is None:
        basis = np.vstack(tangent_basis_from_velocity(vel0))
    basis = orthonormalize_basis(basis)
    if basis is None:
        basis = np.eye(3, dtype=float)

    coeff_pos = np.column_stack([
        rng.normal(0.0, R0, size=n_tracers),
        rng.normal(0.0, R0, size=n_tracers),
        rng.normal(0.0, R0, size=n_tracers),
    ])
    coeff_vel = np.column_stack([
        rng.normal(0.0, sigma_l, size=n_tracers),
        rng.normal(0.0, sigma_t, size=n_tracers),
        rng.normal(0.0, sigma_t, size=n_tracers),
    ])

    # Explicit linear combination avoids sporadic backend matmul warnings
    pos_offsets = (
        coeff_pos[:, 0:1] * basis[0][None, :] +
        coeff_pos[:, 1:2] * basis[1][None, :] +
        coeff_pos[:, 2:3] * basis[2][None, :]
    )
    vel_offsets = (
        coeff_vel[:, 0:1] * basis[0][None, :] +
        coeff_vel[:, 1:2] * basis[1][None, :] +
        coeff_vel[:, 2:3] * basis[2][None, :]
    )

    pos = pos0[None, :] + pos_offsets
    vel = vel0[None, :] + vel_offsets
    pos = np.where(np.isfinite(pos), pos, 0.0)
    vel = np.where(np.isfinite(vel), vel, 0.0)

    packet_mass = sanitize_scalar(packet.mass, default=0.0, min_value=0.0)
    masses = np.full(n_tracers, packet_mass / max(n_tracers, 1), dtype=float)
    return pos, vel, masses


def allocate_tracers(packet_masses: np.ndarray, total_tracers: int, min_per_packet: int = 4) -> np.ndarray:
    packet_masses = np.asarray(packet_masses, dtype=float)
    packet_masses = np.where(np.isfinite(packet_masses) & (packet_masses > 0.0), packet_masses, 0.0)
    if packet_masses.sum() <= 0.0:
        return np.zeros_like(packet_masses, dtype=int)

    if total_tracers < len(packet_masses) * min_per_packet:
        min_per_packet = max(1, total_tracers // max(1, len(packet_masses)))

    weights = packet_masses / packet_masses.sum()
    counts = np.maximum(min_per_packet, np.floor(total_tracers * weights).astype(int))

    while counts.sum() > total_tracers:
        order = np.argsort(-counts)
        changed = False
        for k in order:
            if counts[k] > min_per_packet:
                counts[k] -= 1
                changed = True
                if counts.sum() <= total_tracers:
                    break
        if not changed:
            break

    while counts.sum() < total_tracers:
        k = int(np.argmax(weights))
        counts[k] += 1

    return counts


def validate_history_or_raise(history_file: str, hist: Dict[str, np.ndarray]) -> None:
    if "M_stream" in hist and "dM_strip" not in hist:
        raise RuntimeError(
            f"{history_file} looks like an AMC summary file (e.g. AMC_samples_*.npz), "
            "not an event-history file. stream.py expects AMC_history_*.npz."
        )
    if "dM_strip" not in hist:
        raise RuntimeError(
            f"{history_file} is missing dM_strip. "
            "stream.py expects an event-history file AMC_history_*.npz."
        )
    if len(hist["dM_strip"]) == 0:
        raise RuntimeError(
            f"{history_file} contains no stripping events (empty dM_strip array). "
            "Choose another AMC_history_*.npz file."
        )


def characteristic_packet_times_myr(packets: List[Packet]) -> Tuple[float, float, float]:
    """
    Return three characteristic times in Myr:
      - median transverse crossing time  R0 / sigma_t
      - median longitudinal time         R0 / sigma_l
      - median effective expansion time  R0 / max(sigma_t, sigma_l)
    """
    if not packets:
        return np.nan, np.nan, np.nan

    t_cross_t = []
    t_cross_l = []
    t_cross_eff = []

    for p in packets:
        R0 = sanitize_scalar(p.R0, default=np.nan, min_value=MIN_R0_PC)
        sig_t = sanitize_scalar(p.sigma_t, default=np.nan, min_value=0.0)
        sig_l = sanitize_scalar(p.sigma_l, default=np.nan, min_value=0.0)
        if not np.isfinite(R0):
            continue

        if np.isfinite(sig_t) and sig_t > MIN_EFFECTIVE_SIGMA:
            t_cross_t.append(sec_to_myr(R0 * PCTOKM / sig_t))
        if np.isfinite(sig_l) and sig_l > MIN_EFFECTIVE_SIGMA:
            t_cross_l.append(sec_to_myr(R0 * PCTOKM / sig_l))

        sig_eff = max(sig_t if np.isfinite(sig_t) else 0.0,
                      sig_l if np.isfinite(sig_l) else 0.0,
                      MIN_EFFECTIVE_SIGMA)
        t_cross_eff.append(sec_to_myr(R0 * PCTOKM / sig_eff))

    def med(x):
        x = np.array(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.median(x)) if len(x) else np.nan

    return med(t_cross_t), med(t_cross_l), med(t_cross_eff)


def choose_dynamic_slope_window(
    times_myr: np.ndarray,
    metadata: Dict[str, float],
    base_t_min_myr: float = 0.5,
    base_t_max_myr: float = 5.0,
) -> Tuple[float, float]:
    """
    Choose a stream-dependent fitting window.

    Strategy:
      - start after a few effective packet expansion times,
      - stop before very late sparse/noisy times,
      - if orbital period is available, do not extend beyond ~half an orbit,
      - keep the result inside the user-provided broad bounds when possible.
    """
    t_end = float(np.nanmax(times_myr)) if len(times_myr) else np.nan
    t_cross_eff = metadata.get("median_t_cross_eff_myr", np.nan)
    t_orb = metadata.get("orbital_period_myr", np.nan)

    # Dynamical candidates
    dyn_min = np.nan
    dyn_max = np.nan

    if np.isfinite(t_cross_eff) and t_cross_eff > 0.0:
        dyn_min = 3.0 * t_cross_eff
        dyn_max = 30.0 * t_cross_eff

    # Use user-provided window as broad prior, not as the final fixed answer
    t_min = dyn_min if np.isfinite(dyn_min) else base_t_min_myr
    t_max = dyn_max if np.isfinite(dyn_max) else base_t_max_myr

    # Keep inside a sensible overall range
    if np.isfinite(base_t_min_myr):
        t_min = max(t_min, 0.25 * base_t_min_myr)
    if np.isfinite(base_t_max_myr):
        t_max = min(t_max, 2.0 * base_t_max_myr)

    # If orbital period is available, do not fit past ~half an orbit
    if np.isfinite(t_orb) and t_orb > 0.0:
        t_max = min(t_max, 0.5 * t_orb)

    # Never go too close to the very end of the run
    if np.isfinite(t_end) and t_end > 0.0:
        t_max = min(t_max, 0.8 * t_end)

    # Ensure ordering and enough lever arm
    if not np.isfinite(t_min) or t_min <= 0.0:
        t_min = base_t_min_myr
    if not np.isfinite(t_max) or t_max <= t_min:
        t_max = max(base_t_max_myr, 3.0 * t_min)

    return float(t_min), float(t_max)


def run_event_stream(
    history_file: str,
    total_tracers: int = 6000,
    t_end_myr: float = 10.0,
    dt_myr: float = 0.01,
    vcirc: float = 220.0,
    rsoft: float = 0.05,
    save_every: int = 20,
    seed: int = 1234,
):
    hist = load_history(history_file)
    validate_history_or_raise(history_file, hist)

    valid = np.asarray(hist["dM_strip"], dtype=float)
    valid = np.isfinite(valid) & (valid > 0.0)
    indices = np.where(valid)[0]
    if len(indices) == 0:
        raise RuntimeError(
            f"{history_file} contains dM_strip, but all entries are zero or invalid. "
            "Choose another AMC_history_*.npz file."
        )

    packets = [packet_from_history(hist, i) for i in indices]
    packets = [p for p in packets if np.isfinite(p.mass) and p.mass > 0.0]
    if not packets:
        raise RuntimeError(f"{history_file} contains no valid positive-mass stripping packets.")

    packet_masses = np.array([p.mass for p in packets], dtype=float)
    tracer_counts = allocate_tracers(packet_masses, total_tracers)
    rng = np.random.default_rng(seed)

    t_end = myr_to_pc_over_kms(t_end_myr)
    dt = myr_to_pc_over_kms(dt_myr)

    launch_times_abs = np.array([seconds_to_pc_over_kms(p.t_launch) for p in packets], dtype=float)
    finite_launch = np.isfinite(launch_times_abs)
    if not np.any(finite_launch):
        launch_times_abs = np.zeros_like(launch_times_abs)
    else:
        launch_times_abs = np.where(finite_launch, launch_times_abs, np.nanmin(launch_times_abs[finite_launch]))
    launch_times = launch_times_abs - np.min(launch_times_abs)

    t_start = 0.0
    n_steps = int(np.ceil((t_end - t_start) / dt))

    pos = np.empty((0, 3), dtype=float)
    vel = np.empty((0, 3), dtype=float)
    masses = np.empty((0,), dtype=float)
    launched = np.zeros(len(packets), dtype=bool)

    times = []
    rho_cg = []
    rho_local_nn = []
    axes = []
    vel_axes = []
    n_active = []
    snapshots: List[Dict[str, np.ndarray]] = []

    # Characteristic dynamical times from packets
    median_t_cross_t_myr, median_t_cross_l_myr, median_t_cross_eff_myr = characteristic_packet_times_myr(packets)
    orbital_period_myr = np.nan
    if "orbital_period_s" in hist:
        try:
            orbital_period_myr = sec_to_myr(float(hist["orbital_period_s"]))
        except Exception:
            orbital_period_myr = np.nan

    for step in range(n_steps + 1):
        t = t_start + step * dt

        to_launch = np.where((~launched) & (launch_times <= t))[0]
        for j in to_launch:
            p = packets[j]
            ntr = int(max(1, tracer_counts[j]))
            ppos, pvel, pmass = spawn_packet_tracers(p, ntr, rng)
            pos = np.vstack([pos, ppos])
            vel = np.vstack([vel, pvel])
            masses = np.concatenate([masses, pmass])
            launched[j] = True

        if step % save_every == 0:
            times.append(t)
            n_active.append(len(pos))
            total_mass = masses.sum() if len(masses) else 0.0
            rho_cg.append(coarse_grained_density(pos, total_mass))
            rho_local_nn.append(nearest_neighbor_local_density(pos, masses))
            eig_pos = covariance_eigs(pos)
            eig_vel = covariance_eigs(vel)
            axes.append(np.sqrt(np.maximum(eig_pos, 0.0)) if np.all(np.isfinite(eig_pos)) else np.array([np.nan, np.nan, np.nan]))
            vel_axes.append(np.sqrt(np.maximum(eig_vel, 0.0)) if np.all(np.isfinite(eig_vel)) else np.array([np.nan, np.nan, np.nan]))
            snapshots.append({"t": t, "pos": pos.copy(), "vel": vel.copy(), "masses": masses.copy()})

        if step < n_steps and len(pos) > 0:
            pos, vel = leapfrog_step(pos, vel, dt, lambda x: mw_acceleration_loghalo(x, vc=vcirc, rsoft=rsoft))

    return {
        "history_file": history_file,
        "times": np.array(times),
        "rho_cg": np.array(rho_cg),
        "rho_local_nn": np.array(rho_local_nn),
        "axes": np.array(axes),
        "vel_axes": np.array(vel_axes),
        "n_active": np.array(n_active),
        "launch_times": launch_times,
        "launch_times_abs": launch_times_abs,
        "packet_masses": packet_masses,
        "tracer_counts": tracer_counts,
        "snapshots": snapshots,
        "metadata": {
            "history_file": history_file,
            "n_packets": int(len(packets)),
            "total_packet_mass": float(np.sum(packet_masses)),
            "median_t_cross_t_myr": median_t_cross_t_myr,
            "median_t_cross_l_myr": median_t_cross_l_myr,
            "median_t_cross_eff_myr": median_t_cross_eff_myr,
            "orbital_period_myr": orbital_period_myr,
        },
    }


def fit_loglog_slope_window(
    times: np.ndarray,
    values: np.ndarray,
    t_min_myr: float,
    t_max_myr: float,
) -> float:
    t_myr = to_myr(times)
    mask = (
        np.isfinite(values) &
        (values > 0) &
        np.isfinite(t_myr) &
        (t_myr > t_min_myr) &
        (t_myr < t_max_myr)
    )
    if np.sum(mask) < 4:
        return np.nan
    x = np.log(t_myr[mask])
    y = np.log(values[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def half_time_myr(times: np.ndarray, values: np.ndarray, fraction: float = 0.5) -> float:
    finite = np.where(np.isfinite(values) & (values > 0))[0]
    if len(finite) == 0:
        return np.nan
    i0 = finite[0]
    target = fraction * values[i0]
    below = np.where((np.arange(len(values)) >= i0) & np.isfinite(values) & (values <= target))[0]
    if len(below) == 0:
        return np.nan
    return float(to_myr(times[below[0]]))


def summarize_run(
    results: Dict[str, np.ndarray],
    t_min_slope_myr: float = 0.5,
    t_max_slope_myr: float = 5.0,
) -> Dict[str, float]:
    times = results["times"]
    times_myr = to_myr(times)
    rho_cg = results["rho_cg"]
    rho_local = results["rho_local_nn"]
    axes = results["axes"]
    meta = results["metadata"]

    dyn_t_min_myr, dyn_t_max_myr = choose_dynamic_slope_window(
        times_myr,
        meta,
        base_t_min_myr=t_min_slope_myr,
        base_t_max_myr=t_max_slope_myr,
    )

    out: Dict[str, float] = {
        "n_packets": int(meta["n_packets"]),
        "total_packet_mass": float(meta["total_packet_mass"]),
        "t_end_myr": float(times_myr[-1]) if len(times_myr) else np.nan,
        "t_min_slope_myr_input": float(t_min_slope_myr),
        "t_max_slope_myr_input": float(t_max_slope_myr),
        "t_min_slope_myr": float(dyn_t_min_myr),
        "t_max_slope_myr": float(dyn_t_max_myr),
        "median_t_cross_t_myr": float(meta.get("median_t_cross_t_myr", np.nan)),
        "median_t_cross_l_myr": float(meta.get("median_t_cross_l_myr", np.nan)),
        "median_t_cross_eff_myr": float(meta.get("median_t_cross_eff_myr", np.nan)),
        "orbital_period_myr": float(meta.get("orbital_period_myr", np.nan)),
    }

    finite_cg = np.where(np.isfinite(rho_cg) & (rho_cg > 0))[0]
    finite_local = np.where(np.isfinite(rho_local) & (rho_local > 0))[0]

    out["rho_cg_initial"] = float(rho_cg[finite_cg[0]]) if len(finite_cg) else np.nan
    out["rho_cg_final"] = float(rho_cg[finite_cg[-1]]) if len(finite_cg) else np.nan
    out["rho_local_initial"] = float(rho_local[finite_local[0]]) if len(finite_local) else np.nan
    out["rho_local_final"] = float(rho_local[finite_local[-1]]) if len(finite_local) else np.nan
    out["t_first_cg_myr"] = float(times_myr[finite_cg[0]]) if len(finite_cg) else np.nan
    out["t_last_cg_myr"] = float(times_myr[finite_cg[-1]]) if len(finite_cg) else np.nan
    out["t_half_cg_myr"] = half_time_myr(times, rho_cg, fraction=0.5)
    out["t_tenth_cg_myr"] = half_time_myr(times, rho_cg, fraction=0.1)
    out["t_half_local_myr"] = half_time_myr(times, rho_local, fraction=0.5)
    out["t_tenth_local_myr"] = half_time_myr(times, rho_local, fraction=0.1)

    out["cg_slope_window"] = fit_loglog_slope_window(times, rho_cg, dyn_t_min_myr, dyn_t_max_myr)
    out["local_slope_window"] = fit_loglog_slope_window(times, rho_local, dyn_t_min_myr, dyn_t_max_myr)

    if len(finite_cg):
        out["rho_cg_retention"] = float(rho_cg[finite_cg[-1]] / rho_cg[finite_cg[0]])
    else:
        out["rho_cg_retention"] = np.nan
    if len(finite_local):
        out["rho_local_retention"] = float(rho_local[finite_local[-1]] / rho_local[finite_local[0]])
    else:
        out["rho_local_retention"] = np.nan

    if len(axes) and np.all(np.isfinite(axes[-1])) and axes[-1, 2] > 0:
        out["final_axis_major_pc"] = float(axes[-1, 0])
        out["final_axis_intermediate_pc"] = float(axes[-1, 1])
        out["final_axis_minor_pc"] = float(axes[-1, 2])
        out["final_axis_ratio"] = float(axes[-1, 0] / axes[-1, 2])
        out["final_axis_flatness"] = float(axes[-1, 1] / axes[-1, 2]) if axes[-1, 2] > 0 else np.nan
    else:
        out["final_axis_major_pc"] = np.nan
        out["final_axis_intermediate_pc"] = np.nan
        out["final_axis_minor_pc"] = np.nan
        out["final_axis_ratio"] = np.nan
        out["final_axis_flatness"] = np.nan

    if len(axes) and np.all(np.isfinite(axes[0])) and np.all(np.isfinite(axes[-1])):
        out["major_axis_growth"] = float(axes[-1, 0] / axes[0, 0]) if axes[0, 0] > 0 else np.nan
        out["minor_axis_growth"] = float(axes[-1, 2] / axes[0, 2]) if axes[0, 2] > 0 else np.nan
    else:
        out["major_axis_growth"] = np.nan
        out["minor_axis_growth"] = np.nan

    slope_cg = out["cg_slope_window"]
    slope_local = out["local_slope_window"]
    out["cg_vs_t3_offset"] = float(slope_cg + 3.0) if np.isfinite(slope_cg) else np.nan
    out["local_vs_t3_offset"] = float(slope_local + 3.0) if np.isfinite(slope_local) else np.nan
    out["is_shallower_than_t3_cg"] = int(np.isfinite(slope_cg) and slope_cg > -2.5)
    out["is_shallower_than_t3_local"] = int(np.isfinite(slope_local) and slope_local > -2.5)
    out["is_filamentary"] = int(np.isfinite(out["final_axis_ratio"]) and out["final_axis_ratio"] > 10.0)
    out["local_survives_better"] = int(
        np.isfinite(out["rho_local_retention"]) and
        np.isfinite(out["rho_cg_retention"]) and
        out["rho_local_retention"] > out["rho_cg_retention"]
    )
    return out



def weighted_center_of_mass(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return the mass-weighted center of mass of a tracer cloud."""
    pos = np.asarray(pos, dtype=float)
    masses = np.asarray(masses, dtype=float)
    mask = np.all(np.isfinite(pos), axis=1) & np.isfinite(masses) & (masses > 0.0)
    if np.sum(mask) == 0:
        return np.zeros(3, dtype=float)
    return np.average(pos[mask], axis=0, weights=masses[mask])


def principal_axis_frame(pos: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate positions into the stream principal-axis frame.

    The returned coordinates are centered on the mass-weighted center of mass.
    The X axis is the major principal axis of the tracer distribution, the Y axis
    is the intermediate axis, and the Z axis is the minor axis. This is the most
    useful frame for visualizing stream stretching.
    """
    pos = np.asarray(pos, dtype=float)
    masses = np.asarray(masses, dtype=float)
    mask = np.all(np.isfinite(pos), axis=1) & np.isfinite(masses) & (masses > 0.0)
    pos = pos[mask]
    masses = masses[mask]
    if len(pos) < 3 or np.sum(masses) <= 0.0:
        return np.empty((0, 3)), np.eye(3), np.zeros(3)

    cm = np.average(pos, axis=0, weights=masses)
    x = pos - cm
    try:
        cov = np.cov(x.T, aweights=masses)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order]
        # Enforce a right-handed basis for reproducible plots.
        if np.linalg.det(vecs) < 0.0:
            vecs[:, -1] *= -1.0
    except Exception:
        vecs = np.eye(3)
    coords = x @ vecs
    return coords, vecs, cm


def snapshot_projected_density_xy(
    snapshot: Dict[str, np.ndarray],
    bins: int = 180,
    extent_percentile: float = 99.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build a projected surface-density map in the stream principal-axis frame.

    The density is projected along the minor principal axis Z and displayed in
    the X-Y plane, with X aligned with the stream major axis. The returned map
    has units Msun/pc^2.
    """
    if "masses" not in snapshot:
        raise KeyError(
            "Snapshot does not contain tracer masses. Re-run stream.py after the "
            "snapshot mass-saving update."
        )

    pos = np.asarray(snapshot["pos"], dtype=float)
    masses = np.asarray(snapshot["masses"], dtype=float)
    mask = np.all(np.isfinite(pos), axis=1) & np.isfinite(masses) & (masses > 0.0)
    pos = pos[mask]
    masses = masses[mask]
    if len(pos) < 3:
        raise ValueError("Not enough finite tracers in snapshot to build a density map.")

    coords, basis, cm = principal_axis_frame(pos, masses)
    x = coords[:, 0]
    y = coords[:, 1]

    # Robust symmetric plotting window, avoiding rare far-out tracers dominating the view.
    q = float(extent_percentile)
    xlim = np.nanpercentile(np.abs(x), q)
    ylim = np.nanpercentile(np.abs(y), q)
    if not np.isfinite(xlim) or xlim <= 0.0:
        xlim = np.nanmax(np.abs(x)) if np.any(np.isfinite(x)) else 1.0
    if not np.isfinite(ylim) or ylim <= 0.0:
        ylim = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0
    xlim = max(float(xlim), 1.0e-12)
    ylim = max(float(ylim), 1.0e-12)

    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=int(bins),
        range=[[-xlim, xlim], [-ylim, ylim]],
        weights=masses,
    )
    dx = float(np.diff(xedges)[0])
    dy = float(np.diff(yedges)[0])
    sigma_xy = H.T / max(dx * dy, 1.0e-300)
    xcent = 0.5 * (xedges[1:] + xedges[:-1])
    ycent = 0.5 * (yedges[1:] + yedges[:-1])

    meta = {
        "time_myr": sec_to_myr(float(snapshot["t"]) * PCTOKM) if "t" in snapshot else np.nan,
        "n_tracers": int(len(pos)),
        "total_mass": float(np.sum(masses)),
        "xlim_pc": float(xlim),
        "ylim_pc": float(ylim),
        "center_x_pc": float(cm[0]),
        "center_y_pc": float(cm[1]),
        "center_z_pc": float(cm[2]),
    }
    return xcent, ycent, sigma_xy, meta


def plot_stream_density_contour(
    results: Dict[str, np.ndarray],
    output: str,
    snapshot_index: int = -1,
    bins: int = 180,
    extent_percentile: float = 99.5,
    title: Optional[str] = None,
    levels: int = 22,
):
    """
    Plot a projected density contour in the stream center-of-mass frame.

    The coordinates are rotated to the principal-axis frame of the selected
    snapshot. The X axis follows the major axis of the tracer distribution, so
    stream stretching appears along X. The plotted density is the surface density
    projected along the minor principal axis Z.
    """
    snapshots = results.get("snapshots", [])
    if not snapshots:
        raise RuntimeError("No snapshots stored in results; cannot make density contour.")
    snap = snapshots[snapshot_index]
    x, y, sigma_xy, meta = snapshot_projected_density_xy(
        snap,
        bins=bins,
        extent_percentile=extent_percentile,
    )

    positive = sigma_xy[np.isfinite(sigma_xy) & (sigma_xy > 0.0)]
    if len(positive) == 0:
        raise RuntimeError("Projected density map contains no positive cells.")
    vmin = float(np.nanpercentile(positive, 5.0))
    vmax = float(np.nanmax(positive))
    if not np.isfinite(vmin) or vmin <= 0.0 or vmin >= vmax:
        vmin = float(np.nanmin(positive))

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    zplot = np.where(sigma_xy > 0.0, sigma_xy, np.nan)
    im = ax.contourf(
        x,
        y,
        zplot,
        levels=levels,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xlabel(r"$X_{\rm stream}$ [pc]", fontsize=14)
    ax.set_ylabel(r"$Y_{\rm stream}$ [pc]", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_aspect("equal", adjustable="box")
    if title is None:
        title = rf"$t={meta['time_myr']:.2f}\,\mathrm{{Myr}}$"
    ax.set_title(title, fontsize=13)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Projected density [$M_\odot\,{\rm pc}^{-2}$]", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return meta


def plot_stream_density_contour_sequence(
    results: Dict[str, np.ndarray],
    output_prefix: str,
    snapshot_indices: Optional[List[int]] = None,
    bins: int = 180,
    extent_percentile: float = 99.5,
):
    """Write a sequence of projected density contour plots for selected snapshots."""
    snapshots = results.get("snapshots", [])
    if not snapshots:
        raise RuntimeError("No snapshots stored in results; cannot make density contour sequence.")
    if snapshot_indices is None:
        if len(snapshots) >= 3:
            snapshot_indices = [0, len(snapshots) // 2, len(snapshots) - 1]
        else:
            snapshot_indices = list(range(len(snapshots)))
    metas = []
    for idx in snapshot_indices:
        # Normalize negative indices only for the filename.
        true_idx = idx if idx >= 0 else len(snapshots) + idx
        out = f"{output_prefix}_density_contour_xy_{true_idx:04d}.png"
        meta = plot_stream_density_contour(
            results,
            out,
            snapshot_index=idx,
            bins=bins,
            extent_percentile=extent_percentile,
        )
        metas.append(meta)
    return metas

def save_run(output_prefix: str, results: Dict[str, np.ndarray], save_snapshots: bool = False):
    """
    Save compact stream diagnostics. By default snapshots are not written because
    they can be large. Set save_snapshots=True when the tracer coordinates are
    needed for later contour-map post-processing.
    """
    payload = dict(
        times=results["times"].astype(np.float32),
        rho_cg=results["rho_cg"].astype(np.float32),
        rho_local_nn=results["rho_local_nn"].astype(np.float32),
        axes=results["axes"].astype(np.float32),
        vel_axes=results["vel_axes"].astype(np.float32),
        n_active=results["n_active"].astype(np.int32),
        launch_times=results["launch_times"].astype(np.float32),
        launch_times_abs=results["launch_times_abs"].astype(np.float32),
        packet_masses=results["packet_masses"].astype(np.float32),
        tracer_counts=results["tracer_counts"].astype(np.int32),
    )

    if save_snapshots:
        snapshots = results.get("snapshots", [])
        payload["snapshot_count"] = np.array(len(snapshots), dtype=np.int32)
        for i, snap in enumerate(snapshots):
            payload[f"snapshot_{i:04d}_t"] = np.array(snap["t"], dtype=np.float32)
            payload[f"snapshot_{i:04d}_pos"] = snap["pos"].astype(np.float32)
            payload[f"snapshot_{i:04d}_vel"] = snap["vel"].astype(np.float32)
            payload[f"snapshot_{i:04d}_masses"] = snap["masses"].astype(np.float32)

    np.savez_compressed(output_prefix + ".npz", **payload)

def plot_results(results: Dict[str, np.ndarray], output_prefix: Optional[str] = None):
    times_myr = to_myr(results["times"])
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    axs[0, 0].plot(times_myr, results["rho_cg"], label=r"$\rho_{\rm cg}$")
    axs[0, 0].plot(times_myr, results["rho_local_nn"], label=r"$\rho_{\rm local}$", alpha=0.8)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("Time [Myr]")
    axs[0, 0].set_ylabel(r"Density [$M_\odot\,\mathrm{pc}^{-3}$]")
    axs[0, 0].set_title("Density evolution")
    axs[0, 0].legend()

    if len(results["axes"]) > 0:
        axs[0, 1].plot(times_myr, results["axes"][:, 0], label="major")
        axs[0, 1].plot(times_myr, results["axes"][:, 1], label="intermediate")
        axs[0, 1].plot(times_myr, results["axes"][:, 2], label="minor")
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_xlabel("Time [Myr]")
        axs[0, 1].set_ylabel("Spatial sigma [pc]")
        axs[0, 1].legend()
        axs[0, 1].set_title("Spatial covariance axes")

        axs[1, 0].plot(times_myr, results["vel_axes"][:, 0], label="major")
        axs[1, 0].plot(times_myr, results["vel_axes"][:, 1], label="intermediate")
        axs[1, 0].plot(times_myr, results["vel_axes"][:, 2], label="minor")
        axs[1, 0].set_yscale("log")
        axs[1, 0].set_xlabel("Time [Myr]")
        axs[1, 0].set_ylabel("Velocity sigma [km/s]")
        axs[1, 0].legend()
        axs[1, 0].set_title("Velocity covariance axes")

    axs[1, 1].plot(times_myr, results["n_active"])
    axs[1, 1].set_xlabel("Time [Myr]")
    axs[1, 1].set_ylabel("Active tracers")
    axs[1, 1].set_title("Tracer count")

    plt.tight_layout()
    if output_prefix is not None:
        plt.savefig(output_prefix + ".png", dpi=160)
    plt.close(fig)


def get_options():
    p = argparse.ArgumentParser()
    p.add_argument("history_file", type=str)
    p.add_argument("--total_tracers", type=int, default=6000)
    p.add_argument("--t_end_myr", type=float, default=10.0)
    p.add_argument("--dt_myr", type=float, default=0.01)
    p.add_argument("--vcirc", type=float, default=220.0)
    p.add_argument("--rsoft", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--t_min_slope_myr", type=float, default=0.5,
                   help="Broad lower prior for slope fitting window; final value is chosen dynamically")
    p.add_argument("--t_max_slope_myr", type=float, default=5.0,
                   help="Broad upper prior for slope fitting window; final value is chosen dynamically")
    p.add_argument("--output_prefix", type=str, default="")
    p.add_argument("--plot_density_contour", action="store_true",
                   help="Write a projected X_stream-Y_stream density contour for the final snapshot")
    p.add_argument("--plot_density_sequence", action="store_true",
                   help="Write projected density contours for initial, middle, and final snapshots")
    p.add_argument("--contour_bins", type=int, default=180)
    p.add_argument("--contour_extent_percentile", type=float, default=99.5)
    p.add_argument("--save_snapshots", action="store_true",
                   help="Save tracer snapshots to the output NPZ file; useful but can be large")
    return p.parse_args()


if __name__ == "__main__":
    opts = get_options()
    prefix = opts.output_prefix or os.path.splitext(opts.history_file)[0] + "_stream"
    res = run_event_stream(
        opts.history_file,
        total_tracers=opts.total_tracers,
        t_end_myr=opts.t_end_myr,
        dt_myr=opts.dt_myr,
        vcirc=opts.vcirc,
        rsoft=opts.rsoft,
        save_every=opts.save_every,
        seed=opts.seed,
    )
    save_run(prefix, res, save_snapshots=opts.save_snapshots)
    plot_results(res, prefix)
    if opts.plot_density_contour:
        plot_stream_density_contour(
            res,
            prefix + "_density_contour_xy.png",
            snapshot_index=-1,
            bins=opts.contour_bins,
            extent_percentile=opts.contour_extent_percentile,
        )
    if opts.plot_density_sequence:
        plot_stream_density_contour_sequence(
            res,
            prefix,
            bins=opts.contour_bins,
            extent_percentile=opts.contour_extent_percentile,
        )
    print(summarize_run(
        res,
        t_min_slope_myr=opts.t_min_slope_myr,
        t_max_slope_myr=opts.t_max_slope_myr,
    ))
