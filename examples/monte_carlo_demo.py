"""
Day 4 example: Monte Carlo dispersion analysis.

Runs the two-stage demo vehicle many times with randomized Isp,
propellant mass, structural mass, and drag coefficient (representing
realistic manufacturing/performance uncertainty), and visualizes the
resulting spread of outcomes -- rather than trusting a single "nominal"
trajectory, this shows the actual envelope of possible flights.

Run with: python examples/monte_carlo_demo.py

Takes roughly 30-60 seconds for the default 200 trials.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from rocket_sim.monte_carlo import run_dispersion_analysis, format_dispersion_report


def main():
    payload_mass = 500.0
    stage_configs = [
        {
            "prop_mass": 38_000.0, "structural_mass": 4_000.0,
            "Isp": 280.0, "burn_time": 120.0, "A": 3.0, "CD": 0.3,
        },
        {
            "prop_mass": 6_000.0, "structural_mass": 800.0,
            "Isp": 320.0, "burn_time": 180.0, "A": 1.2, "CD": 0.25,
        },
    ]

    n_samples = 200
    print(f"Running {n_samples} Monte Carlo trials "
          f"(Isp +/-2%, mass +/-3%, CD +/-10%, 1-sigma)...")
    t0 = time.time()
    result = run_dispersion_analysis(
        stage_configs, payload_mass, n_samples=n_samples,
        isp_sigma_pct=0.02, mass_sigma_pct=0.03, cd_sigma_pct=0.10,
        seed=42,
    )
    print(f"Done in {time.time()-t0:.1f}s\n")

    print(format_dispersion_report(result))

    successful = [s for s in result["samples"] if not s["failed"]]

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Monte Carlo Dispersion Analysis ({len(successful)} successful trials)",
        fontsize=14, fontweight="bold",
    )

    # Spaghetti plot: every trial's altitude-vs-time trace, semi-transparent.
    for s in successful:
        axes[0, 0].plot(s["trajectory"]["t"], s["trajectory"]["h"], color="tab:blue", alpha=0.05)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[0, 0].set_title("Altitude Envelope (all trials overlaid)")
    axes[0, 0].grid(alpha=0.3)

    altitudes = [s["final_altitude_km"] for s in successful]
    axes[0, 1].hist(altitudes, bins=25, color="tab:blue", edgecolor="white")
    axes[0, 1].set_xlabel("Final altitude (km)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Final Altitude Distribution")
    axes[0, 1].grid(alpha=0.3)

    speeds = [s["final_speed_ms"] for s in successful]
    axes[1, 0].hist(speeds, bins=25, color="tab:red", edgecolor="white")
    axes[1, 0].set_xlabel("Final speed (m/s)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Final Speed Distribution")
    axes[1, 0].grid(alpha=0.3)

    perigees = [s["perigee_altitude_km"] for s in successful if s["perigee_altitude_km"] is not None]
    axes[1, 1].hist(perigees, bins=25, color="tab:purple", edgecolor="white")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1.5, label="Earth's surface")
    axes[1, 1].set_xlabel("Perigee altitude (km)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Perigee Altitude Distribution\n(right of the line = valid orbit)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "plots", "monte_carlo_demo.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.show()


if __name__ == "__main__":
    main()
