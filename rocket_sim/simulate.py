"""
High-level simulation runner: wraps scipy's solve_ivp around the
gravity turn equations of motion, and packages results for plotting.
"""

import numpy as np
from scipy.integrate import solve_ivp
from rocket_sim.dynamics import equations_of_motion


def run_simulation(
    rocket,
    t_span,
    gamma0_deg=89.9,
    v0=1e-3,
    x0=0.0,
    h0=0.0,
    max_step=0.5,
    constant_gravity=False,
):
    """
    Integrate the gravity turn trajectory for a given rocket.

    Parameters
    ----------
    rocket : rocket_sim.vehicle.Rocket
        The vehicle to simulate.
    t_span : tuple (t0, tf)
        Start and end time of the simulation, in seconds.
    gamma0_deg : float
        Initial flight path angle, degrees. Use slightly less than 90
        (e.g. 89.9) to "seed" the gravity turn -- an exact 90 degrees
        is a mathematical equilibrium of the gamma equation and the
        vehicle would go straight up forever.
    v0 : float
        Initial speed, m/s. Must be nonzero to avoid division-by-zero
        in the gamma equation; a small value (e.g. 1 mm/s) is
        physically negligible but numerically safe.
    x0, h0 : float
        Initial downrange distance and altitude, m.
    max_step : float
        Maximum solver step size, s. Keep this small during the
        powered phase so thrust cutoff and rapid gamma changes are
        captured accurately.
    constant_gravity : bool
        Passed through to the dynamics function; True reproduces the
        constant-g assumption used in Curtis's Example 11.1 for
        validation purposes.

    Returns
    -------
    dict
        Dictionary with keys: t, v, gamma_deg, x, h, m, plus the
        raw solve_ivp result under 'raw'.
    """
    y0 = [v0, np.radians(gamma0_deg), x0, h0, rocket.m0]

    sol = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, rocket, constant_gravity),
        t_span=t_span,
        y0=y0,
        method="RK45",
        max_step=max_step,
        dense_output=True,
        rtol=1e-8,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    v, gamma, x, h, m = sol.y

    return {
        "t": sol.t,
        "v": v,
        "gamma_deg": np.degrees(gamma),
        "x": x,
        "h": h,
        "m": m,
        "raw": sol,
    }
