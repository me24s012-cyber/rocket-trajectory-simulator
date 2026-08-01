"""
Mach-dependent drag coefficient model.

Real drag coefficients are not constant with speed: as a vehicle
approaches the speed of sound, shock waves begin to form and wave drag
causes CD to rise sharply (the classic "transonic drag rise"), peaking
somewhere around Mach 1.0-1.1, before gradually decreasing at
supersonic speeds as the shock structure stabilizes.

This module provides a representative CD(Mach) SHAPE for a slender,
axisymmetric body (typical of a rocket/missile airframe), expressed as
a multiplier on a user-supplied subsonic baseline CD. It is NOT
wind-tunnel data for any specific vehicle -- real CD-Mach curves are
determined experimentally or via CFD for a given geometry -- but it
captures the correct qualitative and roughly-correct quantitative
behavior (a multi-fold rise through transonic, then supersonic
relaxation), which is the dominant physical effect missing from a
constant-CD assumption.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from rocket_sim.atmosphere import speed_of_sound

# Control points (Mach, CD multiplier relative to the subsonic baseline)
# describing the classic transonic drag-rise shape for a slender body:
#   - flat and close to 1.0 through the subsonic regime
#   - sharp rise through transonic (shock formation), peaking near Mach 1.05
#   - gradual relaxation through supersonic speeds toward a lower
#     asymptotic multiplier as the shock structure stabilizes
_MACH_POINTS = np.array([0.0, 0.6, 0.8, 0.9, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0])
_MULTIPLIER_POINTS = np.array([1.00, 1.00, 1.05, 1.25, 1.60, 1.80, 1.65, 1.35, 1.05, 0.80, 0.65, 0.60])

# PchipInterpolator: monotone piecewise cubic Hermite interpolation --
# avoids the overshoot/oscillation a plain cubic spline could introduce
# between control points, which matters here since the curve shape
# (not just the control points) carries physical meaning.
_cd_multiplier_curve = PchipInterpolator(_MACH_POINTS, _MULTIPLIER_POINTS, extrapolate=False)

_MACH_MIN = float(_MACH_POINTS[0])
_MACH_MAX = float(_MACH_POINTS[-1])


def cd_multiplier(mach):
    """
    Return the CD multiplier (relative to a subsonic baseline) for a
    given Mach number, following the representative transonic drag-rise
    curve. Mach numbers below 0 are clipped to 0; above _MACH_MAX, the
    curve's final (asymptotic) value is held constant.
    """
    mach = np.asarray(mach, dtype=float)
    mach_clipped = np.clip(mach, _MACH_MIN, _MACH_MAX)
    return _cd_multiplier_curve(mach_clipped)


def mach_number(v, h):
    """
    Compute the Mach number for a given speed v (m/s) at altitude h (m),
    using the ISA-derived local speed of sound.
    """
    a = speed_of_sound(h)
    return v / a


def drag_coefficient(mach, cd0):
    """
    Return the effective drag coefficient at a given Mach number, as
    cd0 (the vehicle's subsonic baseline CD) scaled by the representative
    transonic/supersonic multiplier curve.

    Parameters
    ----------
    mach : float or array-like
        Mach number.
    cd0 : float
        Subsonic baseline drag coefficient for the vehicle (this is
        the same CD value used elsewhere in this project as a single
        constant -- here it becomes the subsonic reference point that
        the Mach-dependent curve scales from).
    """
    return cd0 * cd_multiplier(mach)


if __name__ == "__main__":
    # Quick sanity check / illustration when running this file directly.
    print(f"{'Mach':>6} | {'CD multiplier':>14} | {'CD (cd0=0.3)':>13}")
    for m in [0.0, 0.5, 0.8, 0.9, 1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0]:
        mult = float(cd_multiplier(m))
        print(f"{m:6.2f} | {mult:14.3f} | {0.3*mult:13.3f}")
