"""
Day 4 example: orbital insertion analysis.

After a vehicle finishes burning, it coasts under gravity alone, which
means its motion from that point on follows a Kepler orbit fully
determined by its speed, flight path angle, and altitude at burnout.
This script runs the same two-stage vehicle as multistage_demo.py and
analyzes whether it actually achieved a valid orbit -- or, just as
usefully, explains why it didn't.

Run with: python examples/orbital_insertion_check.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_sim.vehicle import build_stage_rockets
from rocket_sim.simulate import run_multistage_ascent
from rocket_sim.orbit import analyze_orbit, format_orbit_report


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
    rockets, separation_masses = build_stage_rockets(stage_configs, payload_mass)

    # No extra coast needed -- energy and angular momentum (and hence
    # the resulting orbit) are conserved the instant thrust cuts off,
    # so analyzing right at burnout gives the same orbit as analyzing
    # after any amount of unpowered coasting.
    result = run_multistage_ascent(
        rockets, separation_masses,
        kick_speed=50.0, kick_angle_deg=89.0, coast_time=0.0, max_step=0.3,
    )

    v_final = result["v"][-1]
    gamma_final = result["gamma_deg"][-1]
    h_final = result["h"][-1]

    print(f"Burnout state: v = {v_final:.1f} m/s, "
          f"flight path angle = {gamma_final:.2f} deg, "
          f"altitude = {h_final/1000:.1f} km\n")

    analysis = analyze_orbit(v_final, gamma_final, h_final)
    print(format_orbit_report(analysis))

    print()
    if analysis["perigee_altitude"] is not None and analysis["perigee_altitude"] < 0:
        print(
            "Why this happens: this simulator applies a single deliberate\n"
            "pitchover kick and then lets gravity do all further steering\n"
            "(a passive gravity turn) -- it has no active guidance loop.\n"
            "Real launch vehicles continuously adjust their thrust vector\n"
            "throughout ascent (closed-loop guidance) specifically to hit a\n"
            "precise target insertion state (near-zero flight path angle at\n"
            "the desired altitude, with speed matching local circular\n"
            "velocity). Achieving that with just one passive kick is possible\n"
            "but numerically delicate -- try the Streamlit app (app.py) and\n"
            "experiment with stage sizing, burn times, and the kick angle to\n"
            "see how sensitive orbital insertion is to these parameters."
        )


if __name__ == "__main__":
    main()
