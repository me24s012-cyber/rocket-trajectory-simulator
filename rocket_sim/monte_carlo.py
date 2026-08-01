"""
Monte Carlo dispersion analysis.

Real hardware never performs exactly as specified: manufacturing
tolerances mean actual Isp, propellant mass, structural mass, and drag
coefficient all vary somewhat from their nominal design values on any
given flight. Rather than trusting a single "nominal" trajectory, this
module runs the same ascent many times with these parameters randomly
perturbed (each according to a specified uncertainty), and reports the
resulting SPREAD of outcomes -- burnout altitude/speed, final orbit
perigee, etc. -- which is standard practice in real aerospace
reliability/robustness analysis.
"""

import numpy as np
from rocket_sim.vehicle import build_stage_rockets
from rocket_sim.simulate import run_multistage_ascent
from rocket_sim.orbit import analyze_orbit


def run_dispersion_analysis(
    stage_configs,
    payload_mass,
    n_samples=200,
    isp_sigma_pct=0.02,
    mass_sigma_pct=0.03,
    cd_sigma_pct=0.10,
    kick_speed=50.0,
    kick_angle_deg=89.0,
    coast_time=60.0,
    max_step=1.0,
    seed=None,
):
    """
    Run a Monte Carlo dispersion analysis over a multi-stage vehicle,
    randomly perturbing each stage's Isp, propellant mass, structural
    mass, and drag coefficient according to independent normal
    distributions, and collecting the resulting spread of ascent outcomes.

    Parameters
    ----------
    stage_configs : list of dict
        Nominal stage configurations, same format as
        rocket_sim.vehicle.build_stage_rockets() expects.
    payload_mass : float
        Payload mass, kg (not perturbed -- assumed known exactly).
    n_samples : int
        Number of Monte Carlo trials to run.
    isp_sigma_pct : float
        Standard deviation of each stage's Isp, as a fraction of its
        nominal value (e.g., 0.02 = 2% 1-sigma uncertainty).
    mass_sigma_pct : float
        Standard deviation of each stage's propellant AND structural
        mass, as a fraction of nominal (independent per stage/quantity).
    cd_sigma_pct : float
        Standard deviation of each stage's drag coefficient, as a
        fraction of nominal. Larger than the mass/Isp uncertainties
        by default, since CD is typically the least well-characterized
        parameter without wind-tunnel or flight data.
    kick_speed, kick_angle_deg, coast_time, max_step :
        Passed through to run_multistage_ascent() for every trial
        (not perturbed).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        'samples': list of per-trial result dicts, each with keys
            'final_altitude_km', 'final_speed_ms', 'final_downrange_km',
            'perigee_altitude_km' (None if hyperbolic), 'eccentricity',
            'trajectory' (t, h, v arrays, for plotting an envelope),
            or 'failed': True with an 'error' message if that trial
            could not be simulated (e.g., thrust-to-weight < 1 from an
            unlucky combination of perturbations).
        'n_failed': number of trials that failed to simulate.
        'summary': dict of statistics (mean, std, p5, p50, p95) for
            each successfully-simulated output quantity.
    """
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n_samples):
        perturbed_configs = []
        for cfg in stage_configs:
            perturbed_configs.append({
                "prop_mass": max(cfg["prop_mass"] * (1 + rng.normal(0, mass_sigma_pct)), 1.0),
                "structural_mass": max(cfg["structural_mass"] * (1 + rng.normal(0, mass_sigma_pct)), 1.0),
                "Isp": max(cfg["Isp"] * (1 + rng.normal(0, isp_sigma_pct)), 1.0),
                "burn_time": cfg["burn_time"],
                "A": cfg["A"],
                "CD": max(cfg["CD"] * (1 + rng.normal(0, cd_sigma_pct)), 0.01),
            })

        try:
            rockets, separation_masses = build_stage_rockets(perturbed_configs, payload_mass)
            tw_ratio = rockets[0].thrust(rockets[0].m0) / (rockets[0].m0 * 9.80665)
            if tw_ratio < 1.0:
                samples.append({"failed": True, "error": f"T/W ratio {tw_ratio:.2f} < 1 at liftoff"})
                continue

            result = run_multistage_ascent(
                rockets, separation_masses,
                kick_speed=kick_speed, kick_angle_deg=kick_angle_deg,
                coast_time=coast_time, max_step=max_step,
            )
            v_final, gamma_final, h_final = result["v"][-1], result["gamma_deg"][-1], result["h"][-1]
            orbit = analyze_orbit(v_final, gamma_final, h_final)

            samples.append({
                "failed": False,
                "final_altitude_km": h_final / 1000,
                "final_speed_ms": v_final,
                "final_downrange_km": result["x"][-1] / 1000,
                "perigee_altitude_km": (
                    orbit["perigee_altitude"] / 1000 if orbit["perigee_altitude"] is not None else None
                ),
                "eccentricity": orbit["eccentricity"],
                "trajectory": {"t": result["t"], "h": result["h"] / 1000, "v": result["v"]},
            })
        except Exception as e:
            samples.append({"failed": True, "error": str(e)})

    n_failed = sum(1 for s in samples if s["failed"])
    successful = [s for s in samples if not s["failed"]]

    def stats(key):
        values = np.array([s[key] for s in successful if s[key] is not None])
        if len(values) == 0:
            return None
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p5": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    n_valid_orbit = sum(
        1 for s in successful
        if s["perigee_altitude_km"] is not None and s["perigee_altitude_km"] > 0
    )

    summary = {
        "final_altitude_km": stats("final_altitude_km"),
        "final_speed_ms": stats("final_speed_ms"),
        "final_downrange_km": stats("final_downrange_km"),
        "eccentricity": stats("eccentricity"),
        "n_successful": len(successful),
        "n_failed": n_failed,
        "valid_orbit_probability": n_valid_orbit / len(successful) if successful else 0.0,
    }

    return {"samples": samples, "n_failed": n_failed, "summary": summary}


def format_dispersion_report(result):
    """Format a run_dispersion_analysis() result as a human-readable string."""
    s = result["summary"]
    lines = []
    lines.append(f"Successful trials: {s['n_successful']}  (failed: {s['n_failed']})")
    lines.append("")

    def fmt_stat(label, stat, unit=""):
        if stat is None:
            return f"{label}: no data"
        return (f"{label}: mean={stat['mean']:.2f}{unit}  std={stat['std']:.2f}{unit}  "
                f"[P5={stat['p5']:.2f}, P50={stat['p50']:.2f}, P95={stat['p95']:.2f}]{unit}")

    lines.append(fmt_stat("Final altitude", s["final_altitude_km"], " km"))
    lines.append(fmt_stat("Final speed", s["final_speed_ms"], " m/s"))
    lines.append(fmt_stat("Final downrange", s["final_downrange_km"], " km"))
    lines.append(fmt_stat("Eccentricity", s["eccentricity"]))
    lines.append("")
    lines.append(f"Probability of achieving a valid orbit (perigee > 0): "
                 f"{s['valid_orbit_probability']*100:.1f}%")
    return "\n".join(lines)
