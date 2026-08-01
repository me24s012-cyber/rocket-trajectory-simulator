"""
Day 3 example: a two-stage launch vehicle. Demonstrates why real
satellite launchers stage -- dropping the spent first-stage structure
mid-flight lets the second stage accelerate a much smaller mass,
achieving far higher final speed than a single-stage vehicle with the
same total propellant could.

Run with: python examples/multistage_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from rocket_sim.vehicle import build_stage_rockets
from rocket_sim.simulate import run_multistage_ascent


def main():
    payload_mass = 500.0  # kg, e.g. a small satellite

    stage_configs = [
        {  # Stage 1: booster
            "prop_mass": 38_000.0,
            "structural_mass": 4_000.0,
            "Isp": 280.0,
            "burn_time": 120.0,
            "A": 3.0,
            "CD": 0.3,
        },
        {  # Stage 2: upper stage
            "prop_mass": 6_000.0,
            "structural_mass": 800.0,
            "Isp": 320.0,   # vacuum-optimized engines: higher Isp
            "burn_time": 180.0,
            "A": 1.2,       # smaller frontal area for the upper stage
            "CD": 0.25,
        },
    ]

    rockets, separation_masses = build_stage_rockets(stage_configs, payload_mass)

    print("Stage 1:", rockets[0])
    print(f"  Structural mass jettisoned at separation: {separation_masses[0]:.1f} kg")
    print("Stage 2:", rockets[1])
    print(f"  Structural mass jettisoned at separation: {separation_masses[1]:.1f} kg")
    print()

    result = run_multistage_ascent(
        rockets,
        separation_masses,
        kick_speed=50.0,
        kick_angle_deg=89.0,
        coast_time=60.0,
        max_step=0.2,
    )

    t = result["t"]
    v = result["v"]
    gamma = result["gamma_deg"]
    x = result["x"]
    h = result["h"]
    m = result["m"]
    sep_times = result["stage_separation_times"]

    print(f"Stage 1 burnout / separation at t = {sep_times[0]:.1f} s")
    print(f"Stage 2 burnout / separation at t = {sep_times[1]:.1f} s")
    print()
    print(f"Final state (t = {t[-1]:.1f} s):")
    print(f"  Altitude:          {h[-1]/1000:.2f} km")
    print(f"  Speed:             {v[-1]:.1f} m/s")
    print(f"  Flight path angle: {gamma[-1]:.2f} deg")
    print(f"  Downrange:         {x[-1]/1000:.2f} km")
    print(f"  Remaining mass:    {m[-1]:.1f} kg (payload={payload_mass:.1f} kg)")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Two-Stage Launch Vehicle Ascent", fontsize=14, fontweight="bold")

    def mark_stages(ax):
        for i, st in enumerate(sep_times):
            ax.axvline(st, color="gray", linestyle="--", linewidth=1,
                       label="Stage separation" if i == 0 else None)

    axes[0, 0].plot(t, h / 1000, color="tab:blue")
    mark_stages(axes[0, 0])
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[0, 0].set_title("Altitude vs Time")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, v, color="tab:red")
    mark_stages(axes[0, 1])
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("Speed vs Time (note the kink at separation)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, m, color="tab:orange")
    mark_stages(axes[1, 0])
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Vehicle mass (kg)")
    axes[1, 0].set_title("Mass vs Time (drops sharply at separation)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x / 1000, h / 1000, color="tab:purple")
    axes[1, 1].set_xlabel("Downrange distance (km)")
    axes[1, 1].set_ylabel("Altitude (km)")
    axes[1, 1].set_title("Ground Track Profile")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "..", "plots", "multistage_demo.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.show()


if __name__ == "__main__":
    main()
