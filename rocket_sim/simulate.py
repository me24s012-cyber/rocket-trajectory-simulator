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


def run_gravity_turn_ascent(
    rocket,
    t_span,
    kick_speed=50.0,
    kick_angle_deg=89.0,
    max_step=0.2,
):
    """
    Simulate a realistic gravity-turn ascent in two phases, avoiding
    the numerical singularity in the flight-path-angle equation that
    occurs when velocity is near zero.

    Phase 1 (vertical rise): gamma is held at exactly 90 degrees
    (straight up) from liftoff until the vehicle reaches `kick_speed`.
    At gamma = 90 deg exactly, cos(gamma) = 0, which analytically
    cancels the 1/v term in the flight-path-angle equation -- so this
    phase is well-behaved even as v starts from ~0.

    Phase 2 (gravity turn): a small "pitchover kick" is applied
    (gamma is nudged from 90 deg to `kick_angle_deg`), mimicking the
    real vernier-thruster/gimbal pitchover maneuver Curtis describes.
    From this point on, gravity alone continues to bend the trajectory
    over toward horizontal -- the actual "gravity turn" mechanism.

    Parameters
    ----------
    rocket : rocket_sim.vehicle.Rocket
    t_span : tuple (t0, tf)
        Overall simulation time span, seconds.
    kick_speed : float
        Speed (m/s) at which the vertical rise phase ends and the
        pitchover kick is applied. Must be reached before burnout.
    kick_angle_deg : float
        Flight path angle (degrees) immediately after the pitchover
        kick. Should be close to but less than 90.
    max_step : float
        Maximum solver step size, s.

    Returns
    -------
    dict
        Same structure as run_simulation's return value, but stitched
        together from both phases into one continuous result. Also
        includes 't_kick', the time at which the pitchover occurred.
    """
    # --- Phase 1: vertical rise, gamma fixed at 90 deg exactly ---
    def hit_kick_speed(t, y):
        return y[0] - kick_speed
    hit_kick_speed.terminal = True
    hit_kick_speed.direction = 1

    y0_phase1 = [1e-3, np.pi / 2, 0.0, 0.0, rocket.m0]

    sol1 = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, rocket),
        t_span=t_span,
        y0=y0_phase1,
        method="RK45",
        max_step=max_step,
        events=hit_kick_speed,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol1.success:
        raise RuntimeError(f"Phase 1 (vertical rise) integration failed: {sol1.message}")
    if len(sol1.t_events[0]) == 0:
        raise RuntimeError(
            f"Vehicle never reached kick_speed={kick_speed} m/s within t_span. "
            "Check thrust-to-weight ratio or increase t_span."
        )

    t_kick = sol1.t_events[0][0]
    v_kick, _gamma_at_kick, x_kick, h_kick, m_kick = sol1.y_events[0][0]

    # --- Phase 2: apply pitchover kick, integrate full gravity turn ---
    y0_phase2 = [v_kick, np.radians(kick_angle_deg), x_kick, h_kick, m_kick]

    sol2 = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, rocket),
        t_span=(t_kick, t_span[1]),
        y0=y0_phase2,
        method="RK45",
        max_step=max_step,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol2.success:
        raise RuntimeError(f"Phase 2 (gravity turn) integration failed: {sol2.message}")

    # Stitch the two phases together (avoid duplicating the kick instant).
    t = np.concatenate([sol1.t, sol2.t[1:]])
    v = np.concatenate([sol1.y[0], sol2.y[0][1:]])
    gamma = np.concatenate([sol1.y[1], sol2.y[1][1:]])
    x = np.concatenate([sol1.y[2], sol2.y[2][1:]])
    h = np.concatenate([sol1.y[3], sol2.y[3][1:]])
    m = np.concatenate([sol1.y[4], sol2.y[4][1:]])

    return {
        "t": t,
        "v": v,
        "gamma_deg": np.degrees(gamma),
        "x": x,
        "h": h,
        "m": m,
        "t_kick": t_kick,
        "raw": (sol1, sol2),
    }
