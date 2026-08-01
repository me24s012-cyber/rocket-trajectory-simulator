"""
Day 2 example: a full gravity-turn ascent with atmospheric drag,
using loosely realistic parameters for a small-to-medium launch
vehicle (numbers are illustrative, not an exact match to any real
rocket).

This demonstrates the actual "gravity turn" mechanism: the vehicle
starts (nearly) vertical and gradually pitches over toward horizontal
as gravity acts on it -- the same mechanism real satellite launch
vehicles use to trade vertical speed for horizontal (orbital) speed.

Run with: python examples/gravity_turn_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from rocket_sim.vehicle import Rocket
from rocket_sim.simulate import run_gravity_turn_ascent


def main():
    # A single-stage vehicle with illustrative parameters:
    # - Thrust-to-weight ratio at liftoff ~1.5 (Curtis notes real
    #   vehicles are typically 1.3-2.0)
    # - Isp = 300 s (mid-range for a liquid rocket)
    # - Reasonably large frontal area/drag coefficient product
    rocket = Rocket(
        m0=50_000.0,      # kg, wet mass
        mf=8_000.0,       # kg, dry mass after burnout
        Isp=300.0,        # s
        burn_time=150.0,  # s
        A=3.0,            # m^2, frontal area
        CD=0.3,           # dimensionless drag coefficient
    )
    print("Vehicle:", rocket)

    T_liftoff = rocket.thrust(rocket.m0)
    TW_ratio = T_liftoff / (rocket.m0 * 9.80665)
    print(f"Liftoff thrust: {T_liftoff/1000:.1f} kN, "
          f"thrust-to-weight ratio: {TW_ratio:.2f}")
    print()

    # Run the simulation through burnout plus some coast time.
    # Uses a two-phase approach: vertical rise until kick_speed is
    # reached, then a small pitchover kick starts the gravity turn.
    # (A naive single-phase integration starting near-vertical AND
    # near-zero-velocity at the same time is numerically unstable --
    # the flight-path-angle equation has a 1/v term that blows up.)
    result = run_gravity_turn_ascent(
        rocket,
        t_span=(0, rocket.burn_time + 60),
        kick_speed=50.0,      # m/s -- reached after vertical rise
        kick_angle_deg=89.0,  # deg -- small pitchover to seed the turn
        max_step=0.2,
    )
    print(f"Pitchover kick applied at t={result['t_kick']:.2f} s\n")

    t = result["t"]
    v = result["v"]
    gamma = result["gamma_deg"]
    x = result["x"]
    h = result["h"]
    m = result["m"]

    # Report key milestones.
    burnout_idx = np.searchsorted(t, rocket.burn_time)
    print(f"At burnout (t={t[burnout_idx]:.1f} s):")
    print(f"  Altitude:         {h[burnout_idx]/1000:.2f} km")
    print(f"  Speed:            {v[burnout_idx]:.1f} m/s")
    print(f"  Flight path angle: {gamma[burnout_idx]:.2f} deg")
    print(f"  Downrange:        {x[burnout_idx]/1000:.2f} km")
    print()
    print(f"At end of simulation (t={t[-1]:.1f} s):")
    print(f"  Altitude:         {h[-1]/1000:.2f} km")
    print(f"  Speed:            {v[-1]:.1f} m/s")
    print(f"  Flight path angle: {gamma[-1]:.2f} deg")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Gravity Turn Ascent Trajectory", fontsize=14, fontweight="bold")

    axes[0, 0].plot(t, h / 1000, color="tab:blue")
    axes[0, 0].axvline(rocket.burn_time, color="gray", linestyle="--", linewidth=1, label="Burnout")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[0, 0].set_title("Altitude vs Time")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, v, color="tab:red")
    axes[0, 1].axvline(rocket.burn_time, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("Speed vs Time")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, gamma, color="tab:green")
    axes[1, 0].axvline(rocket.burn_time, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Flight path angle (deg)")
    axes[1, 0].set_title("Flight Path Angle vs Time (the gravity turn)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x / 1000, h / 1000, color="tab:purple")
    axes[1, 1].set_xlabel("Downrange distance (km)")
    axes[1, 1].set_ylabel("Altitude (km)")
    axes[1, 1].set_title("Ground Track Profile")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "..", "plots", "gravity_turn_demo.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.show()


if __name__ == "__main__":
    main()
