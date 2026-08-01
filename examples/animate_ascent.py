"""
Generates an animated GIF of the two-stage launch vehicle's ascent
trajectory -- a moving marker along the ground-track profile, with the
flight path angle and stage separations visible as the animation plays.

Run with: python examples/animate_ascent.py

Note: this can take 20-60 seconds to render depending on your machine,
since it's writing many animation frames to disk.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rocket_sim.vehicle import build_stage_rockets
from rocket_sim.simulate import run_multistage_ascent
from rocket_sim.visualization import rocket_marker


def main():
    payload_mass = 500.0

    stage_configs = [
        {
            "prop_mass": 38_000.0,
            "structural_mass": 4_000.0,
            "Isp": 280.0,
            "burn_time": 120.0,
            "A": 3.0,
            "CD": 0.3,
        },
        {
            "prop_mass": 6_000.0,
            "structural_mass": 800.0,
            "Isp": 320.0,
            "burn_time": 180.0,
            "A": 1.2,
            "CD": 0.25,
        },
    ]

    rockets, separation_masses = build_stage_rockets(stage_configs, payload_mass)

    result = run_multistage_ascent(
        rockets,
        separation_masses,
        kick_speed=50.0,
        kick_angle_deg=89.0,
        coast_time=60.0,
        max_step=0.5,  # coarser step is fine for animation, keeps frame count reasonable
    )

    t = result["t"]
    x = result["x"] / 1000.0   # km
    h = result["h"] / 1000.0   # km
    v = result["v"]
    gamma = result["gamma_deg"]
    sep_times = result["stage_separation_times"]

    # Downsample to a manageable number of animation frames (e.g. ~150).
    n_frames = 150
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    t_f, x_f, h_f, v_f, gamma_f = t[idx], x[idx], h[idx], v[idx], gamma[idx]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Keep only a very faint dotted preview so axis scaling has context,
    # but don't fully draw the path -- otherwise the red trail (which
    # traces the same path) is invisible on top of it.
    ax.plot(x, h, color="lightgray", linewidth=0.8, linestyle=":", zorder=1, alpha=0.5)

    # Mark stage separation points (drawn faintly, revealed properly once
    # the vehicle trail reaches them -- see update()).
    sep_indices = [np.searchsorted(t, st) for st in sep_times]

    point, = ax.plot([], [], marker=rocket_marker(90), color="tab:red", markersize=22,
                      zorder=4, label="Vehicle", markeredgecolor="darkred", linestyle="None")
    trail, = ax.plot([], [], color="tab:red", linewidth=2.5, zorder=3, label="Path flown")
    sep_markers, = ax.plot([], [], marker="x", color="black", markersize=10,
                            linestyle="None", zorder=3, label="Stage separation")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
                         verticalalignment="top",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("Downrange distance (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Two-Stage Launch Vehicle Ascent (Animated)")
    ax.set_xlim(0, x.max() * 1.05)
    ax.set_ylim(0, h.max() * 1.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    def init():
        point.set_data([], [])
        trail.set_data([], [])
        sep_markers.set_data([], [])
        time_text.set_text("")
        return point, trail, sep_markers, time_text

    def update(frame):
        point.set_data([x_f[frame]], [h_f[frame]])
        point.set_marker(rocket_marker(gamma_f[frame]))
        trail.set_data(x_f[:frame + 1], h_f[:frame + 1])

        # Reveal each separation marker once the vehicle has passed it.
        current_time = t_f[frame]
        passed = [i for i, st in enumerate(sep_times) if current_time >= st]
        if passed:
            sep_x = [x[sep_indices[i]] for i in passed]
            sep_h = [h[sep_indices[i]] for i in passed]
            sep_markers.set_data(sep_x, sep_h)

        time_text.set_text(
            f"t = {t_f[frame]:5.1f} s\n"
            f"altitude = {h_f[frame]:6.1f} km\n"
            f"speed = {v_f[frame]:6.0f} m/s"
        )
        return point, trail, sep_markers, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=60, blit=True,
    )

    out_path = os.path.join(os.path.dirname(__file__), "..", "assets", "ascent_animation.gif")
    anim.save(out_path, writer="pillow", fps=20)
    print(f"Animation saved to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
