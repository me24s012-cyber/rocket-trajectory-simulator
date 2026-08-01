"""
Simple exponential atmosphere model (a common approximation to the
International Standard Atmosphere) for use in drag calculations.

Reference values (sea-level density, scale height) are standard
atmospheric constants used in aerospace preliminary design.
"""

import numpy as np

# Sea-level density, kg/m^3
RHO_0 = 1.225

# Atmospheric scale height, m
# (Height over which density falls by a factor of 1/e.
#  ~8500 m is a representative approximation for Earth's lower atmosphere.)
SCALE_HEIGHT = 8500.0

# Earth's gravitational parameter (GM_Earth), m^3/s^2 (Curtis Table A.2:
# mu_Earth = 398,600 km^3/s^2, converted to SI here).
MU_EARTH = 398_600e9

# Earth's mean equatorial radius, m.
R_EARTH = 6_378_000.0


def density(h):
    """
    Return atmospheric density at altitude h (meters) using an
    exponential decay approximation:

        rho(h) = rho_0 * exp(-h / H)

    Parameters
    ----------
    h : float or array-like
        Altitude above sea level, in meters. Negative altitudes are
        clipped to 0 to avoid unphysical density growth below sea level.

    Returns
    -------
    float or ndarray
        Atmospheric density in kg/m^3. Returns 0 above ~150 km, since
        the exponential model is not valid that high and the true
        atmosphere is negligibly thin there anyway.
    """
    h = np.asarray(h, dtype=float)
    h_clipped = np.clip(h, 0.0, None)

    rho = RHO_0 * np.exp(-h_clipped / SCALE_HEIGHT)

    # Beyond ~150 km the exponential model breaks down; treat as vacuum.
    rho = np.where(h_clipped > 150_000.0, 0.0, rho)

    return rho


def gravity(h, mu_earth=MU_EARTH, r_earth=R_EARTH):
    """
    Return local gravitational acceleration at altitude h using the
    inverse-square law (Curtis Eq. 1.8 equivalent):

        g(h) = mu_earth / (r_earth + h)^2

    Parameters
    ----------
    h : float or array-like
        Altitude above sea level, in meters.
    mu_earth : float
        Earth's gravitational parameter, m^3/s^2 (default 398,600 km^3/s^2
        converted to SI).
    r_earth : float
        Earth's mean radius, in meters.

    Returns
    -------
    float or ndarray
        Local gravitational acceleration, m/s^2.
    """
    h = np.asarray(h, dtype=float)
    return mu_earth / (r_earth + h) ** 2


if __name__ == "__main__":
    # Quick sanity check when running this file directly.
    test_altitudes = [0, 1000, 5000, 10_000, 50_000, 100_000, 200_000]
    print(f"{'Altitude (m)':>12} | {'Density (kg/m^3)':>18} | {'g (m/s^2)':>10}")
    for h in test_altitudes:
        print(f"{h:12.0f} | {density(h):18.6f} | {gravity(h):10.4f}")
