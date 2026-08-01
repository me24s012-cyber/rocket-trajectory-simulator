"""
International Standard Atmosphere (ISA) model, used for drag
calculations.

Implements the actual layered ISA model (1976 US Standard Atmosphere /
ICAO ISA): temperature varies linearly with altitude within each of
several layers (the troposphere, tropopause, stratosphere, etc.), with
different lapse rates per layer, and pressure/density are then derived
from temperature via the barometric formula and the ideal gas law.
This replaces a cruder single-exponential approximation used earlier
in this project (still available as density_exponential_approx() for
comparison) with the standard reference model actually used in
aerospace engineering.

Valid from sea level to 86 km geometric altitude (the standard ISA's
defined range); above that, density is extrapolated with a continued
exponential decay from the 86 km conditions down to zero by 150 km,
since the true thermosphere behaves very differently from the ISA's
layered troposphere/stratosphere/mesosphere model and is out of scope
for this project's drag calculations (which are negligible up there
regardless of the exact model used).
"""

import numpy as np

# ---------------------------------------------------------------
# Physical constants used by the ISA model
# ---------------------------------------------------------------
G0 = 9.80665          # standard gravity, m/s^2 (used as a model constant, not local g)
R_STAR = 8.31432       # universal gas constant as defined in the ISA standard, J/(mol*K)
M_AIR = 0.0289644      # molar mass of dry air, kg/mol
T0_SEA_LEVEL = 288.15  # sea-level standard temperature, K
P0_SEA_LEVEL = 101325.0  # sea-level standard pressure, Pa

# ISA layer definitions: (base geometric altitude in m, lapse rate in K/m).
# The lapse rate is the rate at which temperature DECREASES with altitude
# within that layer; a lapse rate of 0 means an isothermal layer.
# Source: ICAO/1976 US Standard Atmosphere, valid 0-86 km.
_ISA_LAYERS = [
    (0.0,      -0.0065),  # Troposphere
    (11000.0,   0.0),     # Tropopause (isothermal)
    (20000.0,   0.0010),  # Stratosphere I
    (32000.0,   0.0028),  # Stratosphere II
    (47000.0,   0.0),     # Stratopause (isothermal)
    (51000.0,  -0.0028),  # Mesosphere I
    (71000.0,  -0.0020),  # Mesosphere II
    (86000.0,   None),    # upper bound of the standard model
]


def _compute_layer_base_conditions():
    """
    Precompute temperature and pressure at the base of every ISA layer
    by integrating the barometric formula upward from sea level,
    layer by layer. This guarantees internal consistency (each layer's
    base conditions are exactly what the previous layer's formula
    predicts at its own top), rather than relying on separately
    tabulated constants that could be transcribed inconsistently.
    """
    bases = []
    T_b = T0_SEA_LEVEL
    P_b = P0_SEA_LEVEL
    for i in range(len(_ISA_LAYERS) - 1):
        h_b, L_b = _ISA_LAYERS[i]
        h_next, _ = _ISA_LAYERS[i + 1]
        bases.append((h_b, T_b, P_b, L_b))

        # Temperature and pressure at the TOP of this layer (= base of next):
        delta_h = h_next - h_b
        T_top = T_b + L_b * delta_h
        if abs(L_b) < 1e-12:
            # Isothermal layer: P = P_b * exp(-g0*M*delta_h / (R*T_b))
            P_top = P_b * np.exp(-G0 * M_AIR * delta_h / (R_STAR * T_b))
        else:
            # Linear lapse layer:
            # P = P_b * (T/T_b)^(-g0*M / (R*L))
            P_top = P_b * (T_top / T_b) ** (-G0 * M_AIR / (R_STAR * L_b))

        T_b, P_b = T_top, P_top

    return bases


_LAYER_BASES = _compute_layer_base_conditions()  # computed once at import time


def _isa_temperature_pressure_scalar(h):
    """Temperature (K) and pressure (Pa) at a single altitude h (m),
    0 <= h <= 86000, via the standard ISA layer formulas."""
    h = max(h, 0.0)
    h = min(h, 86000.0)

    # Find which layer h falls into.
    layer_idx = 0
    for i, (h_b, T_b, P_b, L_b) in enumerate(_LAYER_BASES):
        if h >= h_b:
            layer_idx = i
        else:
            break
    h_b, T_b, P_b, L_b = _LAYER_BASES[layer_idx]

    delta_h = h - h_b
    T = T_b + L_b * delta_h
    if abs(L_b) < 1e-12:
        P = P_b * np.exp(-G0 * M_AIR * delta_h / (R_STAR * T_b))
    else:
        P = P_b * (T / T_b) ** (-G0 * M_AIR / (R_STAR * L_b))

    return T, P


_isa_vectorized = np.vectorize(_isa_temperature_pressure_scalar, otypes=[float, float])


def temperature(h):
    """
    Atmospheric temperature at altitude h (meters), K, per the ISA
    layered model. Valid 0-86 km; altitudes outside that range are
    clipped to the nearest boundary.
    """
    T, _ = _isa_vectorized(np.asarray(h, dtype=float))
    return T


def pressure(h):
    """
    Atmospheric pressure at altitude h (meters), Pa, per the ISA
    layered model. Valid 0-86 km; altitudes outside that range are
    clipped to the nearest boundary.
    """
    _, P = _isa_vectorized(np.asarray(h, dtype=float))
    return P


# Density and speed of sound at the 86 km ISA boundary, used to blend
# smoothly into a simple exponential falloff above that altitude.
_T_86KM, _P_86KM = _isa_temperature_pressure_scalar(86000.0)
_RHO_86KM = _P_86KM * M_AIR / (R_STAR * _T_86KM)
_UPPER_SCALE_HEIGHT = 6000.0  # representative thermosphere scale height, m


def density(h):
    """
    Atmospheric density at altitude h (meters), kg/m^3.

    Uses the full ISA layered model (temperature lapse rates through
    the troposphere/stratosphere/mesosphere, with density derived via
    the ideal gas law: rho = P*M / (R*T)) for 0-86 km, and a continued
    exponential decay above 86 km down to zero by 150 km, since the
    true thermosphere behaves very differently and drag is negligible
    there regardless of the exact model.

    Parameters
    ----------
    h : float or array-like
        Altitude above sea level, in meters. Negative altitudes are
        clipped to 0.

    Returns
    -------
    float or ndarray
        Atmospheric density, kg/m^3.
    """
    h = np.asarray(h, dtype=float)
    h_clipped = np.clip(h, 0.0, None)

    T, P = _isa_vectorized(np.minimum(h_clipped, 86000.0))
    rho_isa = P * M_AIR / (R_STAR * T)

    # Above 86 km: continue an exponential falloff from the ISA boundary
    # density, reaching zero by 150 km (matches the previous model's
    # upper cutoff so nothing above that altitude ever contributes drag).
    rho_upper = _RHO_86KM * np.exp(-(h_clipped - 86000.0) / _UPPER_SCALE_HEIGHT)
    rho = np.where(h_clipped <= 86000.0, rho_isa, rho_upper)
    rho = np.where(h_clipped > 150_000.0, 0.0, rho)

    return rho


def speed_of_sound(h):
    """
    Local speed of sound at altitude h (meters), m/s, computed from
    the ISA temperature: a = sqrt(gamma_air * R_specific * T), with
    gamma_air = 1.4 (diatomic ideal gas) and R_specific = R*/M_air.
    Useful for Mach number calculations.
    """
    GAMMA_AIR = 1.4
    R_specific = R_STAR / M_AIR
    T = temperature(np.clip(np.asarray(h, dtype=float), 0.0, 86000.0))
    return np.sqrt(GAMMA_AIR * R_specific * T)


def density_exponential_approx(h, rho_0=1.225, scale_height=8500.0):
    """
    The simple single-exponential atmosphere approximation used
    earlier in this project, kept here for comparison against the
    full ISA model: rho(h) = rho_0 * exp(-h / H).
    """
    h = np.asarray(h, dtype=float)
    h_clipped = np.clip(h, 0.0, None)
    rho = rho_0 * np.exp(-h_clipped / scale_height)
    return np.where(h_clipped > 150_000.0, 0.0, rho)


# Backwards-compatible aliases for the old module-level constants.
RHO_0 = 1.225
SCALE_HEIGHT = 8500.0

# Earth's gravitational parameter (GM_Earth), m^3/s^2 (Curtis Table A.2:
# mu_Earth = 398,600 km^3/s^2, converted to SI here).
MU_EARTH = 398_600e9

# Earth's mean equatorial radius, m.
R_EARTH = 6_378_000.0


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
    # Quick sanity check when running this file directly, comparing
    # against published ISA reference table values.
    test_altitudes = [0, 1000, 5000, 11000, 20000, 32000, 47000, 51000, 71000, 86000, 100000, 200000]
    print(f"{'Altitude (m)':>12} | {'T (K)':>8} | {'ISA rho (kg/m^3)':>18} | {'Exp. approx rho':>16} | {'g (m/s^2)':>10}")
    for h in test_altitudes:
        T = temperature(min(h, 86000))
        rho_isa = density(h)
        rho_exp = density_exponential_approx(h)
        print(f"{h:12.0f} | {float(T):8.2f} | {float(rho_isa):18.8f} | {float(rho_exp):16.8f} | {gravity(h):10.4f}")
