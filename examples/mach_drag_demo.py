"""
Day 4 example: Mach-dependent drag coefficient.

Compares a vehicle's ascent using a constant drag coefficient (the
older, simpler assumption) against the same vehicle using the
Mach-dependent CD model (rocket_sim.aerodynamics), which captures the
real transonic drag rise as the vehicle passes through the speed of
sound -- a genuine physical effect ("max drag" typically occurs near
Mach 1, not at the vehicle's peak speed).

Run with: python examples/mach_drag_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from rocket_sim.vehicle import Rocket
from rocket_sim.simulate import run_gravity_turn_ascent
from rocket_sim import atmosphere
from rocket_sim import aerodynamics
from rocket_sim.dynamics import R_EARTH


def equations_of_motion_constant_cd(t, y, rocket):
    """Same equations of motion as rocket_sim.dynamics, but using a
    fixed CD instead of the Mach-dependent model -- for comparison."""
    v, gamma, x, h, m = y
    v_safe = max(v, 1e-6)
    g = atmosphere.gravity(h)
    rho = atmosphere.density(h)
    T = rocket.thrust(m)
    D = 0.5 * rho * v_safe**2 * rocket.A * rocket.CD  # constant CD, no Mach dependence
    dv_dt = T / m - D / m - g * np.sin(gamma)
    dgamma_dt = -(1.0 / v_safe) * (g - v_safe**2 / (R_EARTH + h)) * np.cos(gamma)
    dx_dt = (R_EARTH / (R_EARTH + h)) * v_safe * np.cos(gamma)
    dh_dt = v_safe * np.sin(gamma)
    dm_dt = -rocket.mass_flow_rate(m)
    return np.array([dv_dt, dgamma_dt, dx_dt, dh_dt, dm_dt])


def run_constant_cd_ascent(rocket, t_span, kick_speed=50.0, kick_angle_deg=89.0, max_step=0.2):
    """Two-phase ascent (vertical rise + pitchover), identical in
    structure to run_gravity_turn_ascent, but using the fixed-CD
    equations of motion for direct comparison."""
    def hit_kick_speed(t, y):
        return y[0] - kick_speed
    hit_kick_speed.terminal = True
    hit_kick_speed.direction = 1

    sol1 = solve_ivp(
        fun=lambda t, y: equations_of_motion_constant_cd(t, y, rocket),
        t_span=t_span, y0=[1e-3, np.pi / 2, 0.0, 0.0, rocket.m0],
        method="RK45", max_step=max_step, events=hit_kick_speed,
        rtol=1e-8, atol=1e-10,
    )
    t_kick = sol1.t_events[0][0]
    v, _, x, h, m = sol1.y_events[0][0]

    sol2 = solve_ivp(
        fun=lambda t, y: equations_of_motion_constant_cd(t, y, rocket),
        t_span=(t_kick, t_span[1]), y0=[v, np.radians(kick_angle_deg), x, h, m],
        method="RK45", max_step=max_step, rtol=1e-8, atol=1e-10,
    )

    t = np.concatenate([sol1.t, sol2.t[1:]])
    v_arr = np.concatenate([sol1.y[0], sol2.y[0][1:]])
    h_arr = np.concatenate([sol1.y[3], sol2.y[3][1:]])
    return t, v_arr, h_arr


def main():
    rocket = Rocket(
        m0=50_000.0, mf=8_000.0, Isp=300.0, burn_time=150.0, A=3.0, CD=0.3,
    )

    t_span = (0, rocket.burn_time + 20)

    print("Running with constant CD...")
    t_const, v_const, h_const = run_constant_cd_ascent(rocket, t_span)

    print("Running with Mach-dependent CD...")
    result = run_gravity_turn_ascent(rocket, t_span)
    t_mach, v_mach, h_mach = result["t"], result["v"], result["h"]

    # Compute drag force and CD over time for the Mach-dependent case,
    # to show where the transonic drag spike actually occurs.
    mach = aerodynamics.mach_number(np.maximum(v_mach, 1e-6), h_mach)
    cd_eff = aerodynamics.drag_coefficient(mach, rocket.CD)
    rho = atmosphere.density(h_mach)
    drag_force = 0.5 * rho * v_mach**2 * rocket.A * cd_eff

    print(f"\nConstant-CD burnout:      v={v_const[np.searchsorted(t_const, rocket.burn_time)]:.1f} m/s, "
          f"h={h_const[np.searchsorted(t_const, rocket.burn_time)]/1000:.2f} km")
    print(f"Mach-dependent burnout:    v={v_mach[np.searchsorted(t_mach, rocket.burn_time)]:.1f} m/s, "
          f"h={h_mach[np.searchsorted(t_mach, rocket.burn_time)]/1000:.2f} km")

    peak_drag_idx = np.argmax(drag_force)
    print(f"\nPeak drag force: {drag_force[peak_drag_idx]/1000:.1f} kN at t={t_mach[peak_drag_idx]:.1f}s, "
          f"Mach {mach[peak_drag_idx]:.2f}, altitude {h_mach[peak_drag_idx]/1000:.2f} km "
          f"(this is 'max-Q' territory)")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Effect of Mach-Dependent Drag Coefficient", fontsize=14, fontweight="bold")

    mach_range = np.linspace(0, 5, 300)
    axes[0, 0].plot(mach_range, aerodynamics.drag_coefficient(mach_range, rocket.CD), color="tab:red")
    axes[0, 0].axvline(1.0, color="gray", linestyle="--", linewidth=1, label="Mach 1")
    axes[0, 0].set_xlabel("Mach number")
    axes[0, 0].set_ylabel("Drag coefficient CD")
    axes[0, 0].set_title("CD vs Mach (transonic drag rise)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t_mach, mach, color="tab:orange")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Mach 1")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Mach number")
    axes[0, 1].set_title("Mach Number vs Time")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t_mach, drag_force / 1000, color="tab:purple")
    axes[1, 0].axvline(t_mach[peak_drag_idx], color="gray", linestyle="--", linewidth=1, label="Peak drag")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Drag force (kN)")
    axes[1, 0].set_title("Drag Force vs Time (Mach-dependent CD)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t_const, h_const / 1000, color="tab:gray", linestyle="--", label="Constant CD")
    axes[1, 1].plot(t_mach, h_mach / 1000, color="tab:blue", label="Mach-dependent CD")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Altitude (km)")
    axes[1, 1].set_title("Altitude: Constant vs Mach-Dependent CD")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "plots", "mach_drag_demo.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.show()


if __name__ == "__main__":
    main()
