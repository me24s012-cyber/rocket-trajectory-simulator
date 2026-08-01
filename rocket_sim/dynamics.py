"""
Equations of motion for a launch vehicle on a gravity turn trajectory,
following Curtis, "Orbital Mechanics for Engineering Students," Ch. 11.

State vector y = [v, gamma, x, h, m]
    v     : speed, m/s
    gamma : flight path angle, radians (measured from local horizon;
            gamma = 90 deg at vertical liftoff)
    x     : downrange distance, m
    h     : altitude, m
    m     : vehicle mass, kg

Equations (Curtis 11.6, 11.7, 11.8, and mass depletion):
    dv/dt     = T/m - D/m - g*sin(gamma)
    dgamma/dt = -(1/v) * [g - v^2/(R_E + h)] * cos(gamma)
    dx/dt     = [R_E / (R_E + h)] * v * cos(gamma)
    dh/dt     = v * sin(gamma)
    dm/dt     = -mdot   (while propellant remains)

Drag uses a Mach-dependent coefficient (rocket_sim.aerodynamics): the
vehicle's CD attribute is treated as its subsonic baseline value, scaled
by a representative transonic-drag-rise curve based on the local Mach
number (computed from the ISA speed of sound at the current altitude).
"""

import numpy as np
from rocket_sim import atmosphere
from rocket_sim import aerodynamics
from rocket_sim.atmosphere import R_EARTH


def equations_of_motion(t, y, rocket, constant_gravity=False):
    """
    Compute the state derivatives dy/dt for the gravity turn ODE system.

    Parameters
    ----------
    t : float
        Current time, s (unused directly, but required by solve_ivp's
        function signature).
    y : array-like, shape (5,)
        Current state [v, gamma, x, h, m].
    rocket : rocket_sim.vehicle.Rocket
        The vehicle providing thrust(m), mass_flow_rate(m), A, CD. CD is
        treated as the subsonic baseline drag coefficient; the actual
        drag coefficient used varies with Mach number (see
        rocket_sim.aerodynamics.drag_coefficient).
    constant_gravity : bool, optional
        If True, use a fixed sea-level gravity (9.80665 m/s^2) instead
        of the inverse-square variation with altitude. This is used to
        validate against Curtis's Example 11.1 closed-form solution,
        which explicitly neglects the variation of gravity with altitude.

    Returns
    -------
    ndarray, shape (5,)
        Derivatives [dv/dt, dgamma/dt, dx/dt, dh/dt, dm/dt].
    """
    v, gamma, x, h, m = y

    # Avoid division by zero at liftoff (v starts at ~0).
    v_safe = max(v, 1e-6)

    if constant_gravity:
        g = 9.80665
    else:
        g = atmosphere.gravity(h)

    rho = atmosphere.density(h)

    mach = aerodynamics.mach_number(v_safe, h)
    cd_effective = aerodynamics.drag_coefficient(mach, rocket.CD)

    T = rocket.thrust(m)
    D = 0.5 * rho * v_safe**2 * rocket.A * cd_effective

    dv_dt = T / m - D / m - g * np.sin(gamma)

    dgamma_dt = -(1.0 / v_safe) * (g - v_safe**2 / (R_EARTH + h)) * np.cos(gamma)

    dx_dt = (R_EARTH / (R_EARTH + h)) * v_safe * np.cos(gamma)

    dh_dt = v_safe * np.sin(gamma)

    dm_dt = -rocket.mass_flow_rate(m)

    return np.array([dv_dt, dgamma_dt, dx_dt, dh_dt, dm_dt])
