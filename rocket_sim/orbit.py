"""
Orbital insertion analysis.

Once a launch vehicle stops thrusting, it coasts under gravity alone --
which means, from that instant on, its motion follows a Kepler orbit
(an ellipse, parabola, or hyperbola) fully determined by its specific
orbital energy and specific angular momentum at that moment. This
module computes those quantities from a simulated trajectory's final
(v, gamma, h) state and reports the resulting orbit: whether it's a
valid closed orbit around Earth, a suborbital trajectory that will
re-enter, or an escape trajectory.

This treats the vehicle as a two-body (vehicle + Earth) system from
that point forward, ignoring residual atmospheric drag -- a reasonable
assumption above roughly 100-150 km, where the exponential atmosphere
model used elsewhere in this project is already treated as vacuum.
"""

import numpy as np
from rocket_sim.atmosphere import MU_EARTH, R_EARTH


def analyze_orbit(v, gamma_deg, h, mu=MU_EARTH, r_earth=R_EARTH):
    """
    Compute the orbit resulting from a given speed, flight path angle,
    and altitude, treating that instant as the start of unpowered
    two-body motion.

    Parameters
    ----------
    v : float
        Speed at the analysis point, m/s.
    gamma_deg : float
        Flight path angle at the analysis point, degrees (0 = purely
        horizontal, 90 = purely vertical).
    h : float
        Altitude above the reference surface, m.
    mu : float
        Gravitational parameter, m^3/s^2 (default: Earth).
    r_earth : float
        Reference body radius, m (default: Earth).

    Returns
    -------
    dict
        Keys: r, v_circular, specific_energy, angular_momentum,
        eccentricity, semi_major_axis (None if not elliptical),
        perigee_altitude, apogee_altitude (None if not elliptical),
        period (None if not elliptical), orbit_type (str),
        delta_v_to_circularize (float or None).
    """
    gamma = np.radians(gamma_deg)
    r = r_earth + h

    v_radial = v * np.sin(gamma)       # component along the local vertical
    v_transverse = v * np.cos(gamma)   # component along local horizontal

    specific_energy = 0.5 * v**2 - mu / r
    angular_momentum = r * v_transverse  # h = r * v_perp (planar 2D motion)

    v_circular = np.sqrt(mu / r)

    # Eccentricity from energy + angular momentum (valid for e < 1, = 1, > 1):
    e_squared = 1 + (2 * specific_energy * angular_momentum**2) / mu**2
    eccentricity = np.sqrt(max(e_squared, 0.0))

    result = {
        "r": r,
        "v_circular": v_circular,
        "specific_energy": specific_energy,
        "angular_momentum": angular_momentum,
        "eccentricity": eccentricity,
        "semi_major_axis": None,
        "perigee_altitude": None,
        "apogee_altitude": None,
        "period": None,
        "delta_v_to_circularize": None,
    }

    if specific_energy < 0:
        # Bound (elliptical or circular) orbit.
        a = -mu / (2 * specific_energy)
        r_p = a * (1 - eccentricity)
        r_a = a * (1 + eccentricity)
        period = 2 * np.pi * np.sqrt(a**3 / mu)

        result["semi_major_axis"] = a
        result["perigee_altitude"] = r_p - r_earth
        result["apogee_altitude"] = r_a - r_earth
        result["period"] = period

        if r_p < r_earth:
            result["orbit_type"] = (
                "Suborbital -- perigee is below the surface, so this "
                "trajectory will re-enter/impact before completing an orbit."
            )
        elif eccentricity < 0.01:
            result["orbit_type"] = "Circular orbit achieved."
        else:
            result["orbit_type"] = "Elliptical orbit achieved."
            # Delta-v needed at apogee to circularize (raise perigee to
            # match apogee) via a single impulsive burn -- the standard
            # circularization maneuver at the high point of an ellipse.
            v_apogee = np.sqrt(mu * (2 / r_a - 1 / a))
            v_circ_at_apogee = np.sqrt(mu / r_a)
            result["delta_v_to_circularize"] = v_circ_at_apogee - v_apogee
    elif specific_energy == 0:
        result["orbit_type"] = "Parabolic (marginal escape) trajectory."
    else:
        result["orbit_type"] = "Hyperbolic (escape) trajectory -- exceeds Earth escape velocity."

    return result


def format_orbit_report(analysis):
    """Format an analyze_orbit() result dict as a human-readable string."""
    lines = []
    lines.append(f"Orbit type: {analysis['orbit_type']}")
    lines.append(f"Eccentricity: {analysis['eccentricity']:.4f}")
    lines.append(f"Local circular orbital velocity: {analysis['v_circular']:.1f} m/s")

    if analysis["semi_major_axis"] is not None:
        lines.append(f"Semi-major axis: {analysis['semi_major_axis']/1000:.1f} km")
        lines.append(f"Perigee altitude: {analysis['perigee_altitude']/1000:.1f} km")
        lines.append(f"Apogee altitude: {analysis['apogee_altitude']/1000:.1f} km")
        lines.append(f"Orbital period: {analysis['period']/60:.1f} minutes")

    if analysis["delta_v_to_circularize"] is not None:
        lines.append(
            f"Delta-v to circularize at apogee: "
            f"{analysis['delta_v_to_circularize']:.1f} m/s"
        )

    return "\n".join(lines)
