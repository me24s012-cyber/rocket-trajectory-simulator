"""
Validation: compare our numerical gravity-turn integrator against the
closed-form solution Curtis derives in Example 11.1 for a vertical
sounding rocket (gamma = 90 deg throughout, no drag, constant g0).

This confirms our ODE integration is trustworthy BEFORE we add the
harder physics (drag, gravity turn, staging).

Run with: python examples/validate_example_11_1.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_sim.vehicle import Rocket, STANDARD_GRAVITY
from rocket_sim.simulate import run_simulation


def closed_form_burnout(rocket):
    """
    Curtis Example 11.1 closed-form burnout velocity and altitude for
    a vertical (gamma=90 deg), no-drag, constant-g0 sounding rocket.

    v_bo = c*ln(m0/mf) - g0*t_bo
    h_bo = (c/mdot)*[mf*ln(mf/m0) + m0 - mf] - 0.5*g0*t_bo^2

    where c = Isp*g0 (effective exhaust velocity).
    """
    c = rocket.Isp * STANDARD_GRAVITY
    g0 = STANDARD_GRAVITY
    m0, mf, mdot = rocket.m0, rocket.mf, rocket.mdot
    t_bo = rocket.burn_time

    v_bo = c * np.log(m0 / mf) - g0 * t_bo
    h_bo = (c / mdot) * (mf * np.log(mf / m0) + m0 - mf) - 0.5 * g0 * t_bo**2

    return v_bo, h_bo


def main():
    # A simple sounding rocket: no drag (A=0 removes drag entirely),
    # vertical launch, moderate mass ratio.
    rocket = Rocket(
        m0=1000.0,   # kg
        mf=400.0,    # kg
        Isp=250.0,   # s
        burn_time=60.0,  # s
        A=0.0,       # m^2 -- zero area means zero drag, matching Curtis's
                     # no-drag assumption exactly
        CD=0.0,
    )

    print("Vehicle:", rocket)
    print()

    # Analytic (closed-form) result
    v_bo_analytic, h_bo_analytic = closed_form_burnout(rocket)

    # Numerical result: integrate with gamma0 = 90 deg exactly (vertical),
    # constant_gravity=True to match Curtis's assumptions, and A=CD=0
    # for zero drag (already set on the vehicle).
    result = run_simulation(
        rocket,
        t_span=(0, rocket.burn_time),
        gamma0_deg=90.0,
        v0=1e-6,
        max_step=0.05,
        constant_gravity=True,
    )

    v_bo_numeric = result["v"][-1]
    h_bo_numeric = result["h"][-1]
    gamma_bo_numeric = result["gamma_deg"][-1]

    print(f"{'':20s}{'Analytic':>15s}{'Numerical':>15s}{'Diff %':>10s}")
    v_diff_pct = 100 * abs(v_bo_numeric - v_bo_analytic) / abs(v_bo_analytic)
    h_diff_pct = 100 * abs(h_bo_numeric - h_bo_analytic) / abs(h_bo_analytic)
    print(f"{'Burnout v (m/s)':20s}{v_bo_analytic:15.4f}{v_bo_numeric:15.4f}{v_diff_pct:10.4f}")
    print(f"{'Burnout h (m)':20s}{h_bo_analytic:15.2f}{h_bo_numeric:15.2f}{h_diff_pct:10.4f}")
    print(f"\nFlight path angle at burnout (should stay ~90 deg for a "
          f"vertical launch): {gamma_bo_numeric:.4f} deg")

    print()
    if v_diff_pct < 0.1 and h_diff_pct < 0.1:
        print("VALIDATION PASSED: numerical integrator matches closed-form "
              "solution to within 0.1%.")
    else:
        print("VALIDATION WARNING: discrepancy exceeds 0.1%, investigate "
              "before proceeding.")


if __name__ == "__main__":
    main()
