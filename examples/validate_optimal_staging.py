"""
Validation: reproduce Curtis's Example 11.5 (a three-stage optimal
staging problem with known published numerical answers) and confirm
our solver matches.

Run with: python examples/validate_optimal_staging.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_sim.optimal_staging import solve_optimal_staging, format_staging_report


def main():
    payload_mass = 5000.0     # kg
    vbo_required = 10_000.0   # m/s (10 km/s, per the textbook example)

    isp_list = [400.0, 350.0, 300.0]     # s
    epsilon_list = [0.10, 0.15, 0.20]

    result = solve_optimal_staging(vbo_required, payload_mass, isp_list, epsilon_list)
    print(format_staging_report(result, payload_mass))

    # Textbook published answers (Curtis Example 11.5):
    expected = {
        "eta": 0.4668,
        "n": [4.541, 2.507, 1.361],
        "step_mass": [165_700, 18_070, 2_477],
        "structural_mass": [16_570, 2_710, 495.4],
        "propellant_mass": [149_100, 15_360, 1_982],
        "total_mass": 191_200,
    }

    print("\n--- Comparison against Curtis Example 11.5 ---")
    print(f"{'Quantity':<28}{'Textbook':>14}{'This solver':>16}{'Diff %':>10}")

    def compare(label, textbook_val, computed_val):
        diff_pct = 100 * abs(computed_val - textbook_val) / abs(textbook_val)
        print(f"{label:<28}{textbook_val:>14.3f}{computed_val:>16.3f}{diff_pct:>10.3f}")
        return diff_pct

    max_diff = 0.0
    # Curtis works in km/s throughout, so his eta has units of 1/(km/s);
    # our solver works in SI (m/s), so eta has units of 1/(m/s) -- a
    # factor-of-1000 unit difference, not a computational error.
    eta_in_book_units = result["eta"] * 1000
    max_diff = max(max_diff, compare("eta (unit-adjusted)", expected["eta"], eta_in_book_units))
    for i in range(3):
        max_diff = max(max_diff, compare(f"n{i+1}", expected["n"][i], result["stages"][i]["n"]))
    for i in range(3):
        max_diff = max(max_diff, compare(
            f"step_mass{i+1} (kg)", expected["step_mass"][i], result["stages"][i]["step_mass"]
        ))
    for i in range(3):
        max_diff = max(max_diff, compare(
            f"structural_mass{i+1} (kg)", expected["structural_mass"][i], result["stages"][i]["structural_mass"]
        ))
    for i in range(3):
        max_diff = max(max_diff, compare(
            f"propellant_mass{i+1} (kg)", expected["propellant_mass"][i], result["stages"][i]["propellant_mass"]
        ))
    max_diff = max(max_diff, compare("total_mass (kg)", expected["total_mass"], result["total_mass"]))

    print()
    if max_diff < 0.5:
        print(f"VALIDATION PASSED: largest deviation from the textbook's published "
              f"answer is {max_diff:.3f}%.")
    else:
        print(f"VALIDATION WARNING: largest deviation is {max_diff:.3f}%, investigate.")


if __name__ == "__main__":
    main()
