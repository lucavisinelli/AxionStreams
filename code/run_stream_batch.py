#!/usr/bin/env python3
import argparse
import os
import subprocess
from typing import List


def parse_a_values(raw: str) -> List[float]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(x) for x in parts]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default="../MC")
    p.add_argument("--profile", type=str, default="PL_powerlaw")
    p.add_argument("--a_values_kpc", type=str, default="0.5,1.0,2.0,5.0,10.0")
    p.add_argument("--compute_stream_script", type=str, default="Compute_stream.py")
    p.add_argument("--output_root", type=str, default="stream_batch_outputs")
    p.add_argument("--max_files", type=int, default=100)
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
    opts = p.parse_args()

    a_values_kpc = parse_a_values(opts.a_values_kpc)
    os.makedirs(opts.output_root, exist_ok=True)

    for a_kpc in a_values_kpc:
        a_pc = a_kpc * 1000.0
        dirname = f"event_histories_{opts.profile}_a={a_pc:.4f}"
        full_path = os.path.join(opts.base_dir, dirname)

        if not os.path.isdir(full_path):
            print(f"[WARNING] Missing directory: {full_path}")
            continue

        outdir = os.path.join(opts.output_root, f"stream_batch_output_a={a_kpc:.2f}kpc")

        print(f"\n=== Processing a = {a_kpc:.2f} kpc ===")
        print(f"Input : {full_path}")
        print(f"Output: {outdir}")

        cmd = [
            "python3", opts.compute_stream_script, full_path,
            "--outdir", outdir,
            "--max_files", str(opts.max_files),
            "--total_tracers", str(opts.total_tracers),
            "--t_end_myr", str(opts.t_end_myr),
            "--dt_myr", str(opts.dt_myr),
            "--vcirc", str(opts.vcirc),
            "--rsoft", str(opts.rsoft),
            "--save_every", str(opts.save_every),
            "--seed", str(opts.seed),
            "--t_min_slope_myr", str(opts.t_min_slope_myr),
            "--t_max_slope_myr", str(opts.t_max_slope_myr),
        ]
        if opts.save_individual:
            cmd.append("--save_individual")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed at a = {a_kpc:.2f} kpc")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
