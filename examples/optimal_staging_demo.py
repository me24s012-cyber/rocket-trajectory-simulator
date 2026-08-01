"""
Day 4 example: optimal staging.

Demonstrates the benefit of Curtis's Lagrange-multiplier optimal
staging (Section 11.6) by comparing it against a naive baseline that
simply splits the required delta-v equally across stages regardless
of each stage's Isp/structural ratio -- showing how much total vehicle
mass (and therefore cost/complexity) is saved by properly accounting
for each stage's individual performance characteristics.

Run with: python examples/optimal_staging_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from rocket_sim.optimal_staging import solve_optimal_staging, format_staging_report, STANDARD_GRAVITY


def naive_equal_split_mass(vbo_required, payload_mass, isp_list, epsilon_list, g0=STANDARD_GRAVITY):
    """
    Baseline for comparison: split the required delta-v equally across
    all stages (vbo_required / N each), rather than optimizing the
    split. Computes the resulting total vehicle mass using each
    stage's own Isp and structural ratio, top-down, the same way
    solve_optimal_staging() does -- the only difference is how the
    delta-v budget is divided between stages.
    """
    N = len(isp_list)
    c_list = [isp * g0 for isp in isp_list]
    vbo_each = vbo_required / N

    # Required mass ratio per stage for its equal share of delta-v:
    # vbo_each = c_i * ln(n_i)  =>  n_i = exp(vbo_each / c_i)
    n_list = [np.exp(vbo_each / c) for c in c_list]

    step_masses = [0.0] * N
    cumulative = payload_mass
    for i in reversed(range(N)):
        n_i = n_list[i]
        eps_i = epsilon_list[i]
        if n_i * eps_i >= 1:
            return None  # infeasible split for this stage's structural ratio
        m_i = (n_i - 1) / (1 - n_i * eps_i) * cumulative
        step_masses[i] = m_i
        cumulative += m_i

    return payload_mass + sum(step_masses)


def main():
    payload_mass = 5000.0
    vbo_required = 10_000.0  # m/s

    isp_list = [400.0, 350.0, 300.0]
    epsilon_list = [0.10, 0.15, 0.20]

    print("=== Optimal staging (Lagrange multiplier) ===")
    result = solve_optimal_staging(vbo_required, payload_mass, isp_list, epsilon_list)
    print(format_staging_report(result, payload_mass))

    naive_mass = naive_equal_split_mass(vbo_required, payload_mass, isp_list, epsilon_list)

    print("\n=== Comparison: optimal vs. naive equal delta-v split ===")
    print(f"Optimal staging total mass:  {result['total_mass']:,.1f} kg")
    if naive_mass is not None:
        print(f"Naive equal-split total mass: {naive_mass:,.1f} kg")
        savings_pct = 100 * (naive_mass - result["total_mass"]) / naive_mass
        print(f"Mass savings from optimization: {savings_pct:.1f}%")
    else:
        print("Naive equal-split approach is infeasible for this vehicle "
              "(a stage's structural ratio makes an equal delta-v share "
              "impossible) -- this by itself demonstrates why optimal "
              "staging matters: it correctly gives faster-burning, "
              "lower-structural-ratio stages more of the delta-v budget "
              "instead of blindly splitting it evenly.")


if __name__ == "__main__":
    main()
