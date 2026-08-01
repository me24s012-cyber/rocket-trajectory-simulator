"""
Optimal staging (Curtis, "Orbital Mechanics for Engineering Students,"
Section 11.6), solved via the Lagrange multiplier method.

Problem statement: given a required burnout (delta-v) velocity, a
payload mass, and per-stage specific impulse and structural ratio,
find the mass split between stages that minimizes the vehicle's total
initial mass -- i.e., extracts the required delta-v for the least
total propellant + structure.

Key quantities (per stage i):
    Isp_i     : specific impulse, s
    c_i       : effective exhaust velocity = Isp_i * g0, m/s
    epsilon_i : structural ratio = mE_i / (mE_i + mp_i)
                (empty/dry mass of that stage divided by that stage's
                own dry+propellant mass -- NOT including payload above it)
    n_i       : optimal mass ratio for stage i (solved for)

This is a "field-free space" analysis (Curtis Section 11.5-11.6): no
drag or gravity losses are modeled here, so the resulting vbo is an
idealized total delta-v capability, not a literal atmospheric ascent
trajectory. It answers "how should I split my mass budget across
stages" independent of the trajectory-shaping questions the rest of
this project's simulator answers.
"""

import numpy as np
from scipy.optimize import brentq

STANDARD_GRAVITY = 9.80665  # m/s^2


def _mass_ratio(eta, c_i, epsilon_i):
    """Optimal mass ratio for a stage, given the Lagrange multiplier eta
    (Curtis Eq. 11.87): n_i = (c_i*eta - 1) / (c_i*epsilon_i*eta)."""
    return (c_i * eta - 1) / (c_i * epsilon_i * eta)


def _constraint(eta, c_list, epsilon_list, vbo_required):
    """sum_i [c_i * ln(n_i(eta))] - vbo_required (Curtis Eq. 11.84/11.86,
    unexpanded form). This is the equation solved for eta."""
    total = 0.0
    for c_i, eps_i in zip(c_list, epsilon_list):
        n_i = _mass_ratio(eta, c_i, eps_i)
        if n_i <= 0:
            return -1e18  # steer the root finder away from invalid region
        total += c_i * np.log(n_i)
    return total - vbo_required


def solve_optimal_staging(vbo_required, payload_mass, isp_list, epsilon_list,
                           g0=STANDARD_GRAVITY):
    """
    Solve for the optimal mass split across N stages to achieve a
    required burnout velocity for a given payload, using Curtis's
    Lagrange multiplier formulation (Section 11.6).

    Parameters
    ----------
    vbo_required : float
        Required total burnout (delta-v) velocity, m/s.
    payload_mass : float
        Payload mass, kg.
    isp_list : list of float
        Specific impulse of each stage, s, ordered bottom-up (index 0
        = first stage to burn/ignite).
    epsilon_list : list of float
        Structural ratio of each stage (mE_i / (mE_i + mp_i)), same
        order as isp_list. Must be strictly between 0 and 1.
    g0 : float
        Standard gravity used in the Isp-to-exhaust-velocity conversion.

    Returns
    -------
    dict
        'eta': the solved Lagrange multiplier.
        'stages': list of per-stage dicts (bottom-up), each with
            n (mass ratio), step_mass, structural_mass, propellant_mass,
            payload_ratio.
        'total_mass': total vehicle mass (all stages + payload), kg.
        'payload_fraction': payload_mass / total_mass.

    Raises
    ------
    ValueError
        If the required vbo is unattainable with the given Isp/structural
        ratios (i.e., exceeds the limit as eta -> infinity), or if any
        resulting mass ratio is not physically valid (n_i > 1).
    """
    if len(isp_list) != len(epsilon_list):
        raise ValueError("isp_list and epsilon_list must be the same length.")
    for eps in epsilon_list:
        if not (0 < eps < 1):
            raise ValueError(f"structural ratio {eps} must be strictly between 0 and 1.")

    c_list = [isp * g0 for isp in isp_list]

    # eta must exceed 1/min(c_i) for every n_i to be positive (numerator
    # and denominator both positive). Search for a bracketing upper bound.
    eta_lower = (1.0 / min(c_list)) * (1 + 1e-9)
    eta_upper = eta_lower * 10

    # As eta -> infinity, n_i -> 1/epsilon_i for every stage, so vbo
    # approaches its supremum sum(c_i * ln(1/epsilon_i)). If that's
    # still below vbo_required, no solution exists.
    vbo_supremum = sum(c * np.log(1.0 / eps) for c, eps in zip(c_list, epsilon_list))
    if vbo_required >= vbo_supremum:
        raise ValueError(
            f"Required vbo ({vbo_required/1000:.2f} km/s) is unattainable with the "
            f"given Isp/structural ratios even in the limit of infinite mass "
            f"(supremum is {vbo_supremum/1000:.2f} km/s). Increase Isp, reduce "
            f"structural ratios, or add more stages."
        )

    max_expand = 200
    while _constraint(eta_upper, c_list, epsilon_list, vbo_required) < 0 and max_expand > 0:
        eta_upper *= 2
        max_expand -= 1

    eta = brentq(
        lambda eta_: _constraint(eta_, c_list, epsilon_list, vbo_required),
        eta_lower, eta_upper, xtol=1e-14, rtol=1e-14, maxiter=200,
    )

    n_list = [_mass_ratio(eta, c, eps) for c, eps in zip(c_list, epsilon_list)]
    for i, n_i in enumerate(n_list):
        if n_i <= 1:
            raise ValueError(
                f"Stage {i+1}'s optimal mass ratio ({n_i:.3f}) is not > 1, "
                "which is not physically valid -- check inputs."
            )

    # Step masses, computed top-down (Curtis Eq. 11.88): start from the
    # topmost stage (closest to payload, last index) and work down to
    # the bottom (first-to-ignite, index 0).
    N = len(isp_list)
    step_masses = [0.0] * N
    cumulative = payload_mass  # mass "above" the current stage
    for i in reversed(range(N)):
        n_i = n_list[i]
        eps_i = epsilon_list[i]
        m_i = (n_i - 1) / (1 - n_i * eps_i) * cumulative
        step_masses[i] = m_i
        cumulative += m_i

    stages = []
    cumulative = payload_mass
    for i in range(N):
        m_i = step_masses[i]
        mE_i = epsilon_list[i] * m_i
        mp_i = m_i - mE_i
        stages.append({
            "n": n_list[i],
            "Isp": isp_list[i],
            "epsilon": epsilon_list[i],
            "step_mass": m_i,
            "structural_mass": mE_i,
            "propellant_mass": mp_i,
        })

    total_mass = payload_mass + sum(step_masses)
    # Fill in each stage's payload ratio (mass above it / that stage's own mass)
    cumulative_above = payload_mass
    for i in reversed(range(N)):
        stages[i]["payload_ratio"] = cumulative_above / stages[i]["step_mass"]
        cumulative_above += stages[i]["step_mass"]

    return {
        "eta": eta,
        "stages": stages,
        "total_mass": total_mass,
        "payload_fraction": payload_mass / total_mass,
    }


def format_staging_report(result, payload_mass):
    """Format a solve_optimal_staging() result as a human-readable string."""
    lines = [f"Lagrange multiplier (eta): {result['eta']:.6f}", ""]
    for i, s in enumerate(result["stages"]):
        lines.append(
            f"Stage {i+1}: n={s['n']:.3f}  step_mass={s['step_mass']:,.1f} kg  "
            f"structural={s['structural_mass']:,.1f} kg  "
            f"propellant={s['propellant_mass']:,.1f} kg"
        )
    lines.append("")
    lines.append(f"Payload mass: {payload_mass:,.1f} kg")
    lines.append(f"Total vehicle mass: {result['total_mass']:,.1f} kg")
    lines.append(f"Payload fraction: {result['payload_fraction']:.4f}")
    return "\n".join(lines)
