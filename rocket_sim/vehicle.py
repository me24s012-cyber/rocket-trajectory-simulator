"""
Rocket vehicle definition: mass, propulsion, and aerodynamic parameters
for a single stage. Multi-stage support will be added on Day 3.
"""

STANDARD_GRAVITY = 9.80665  # m/s^2, used in the thrust/Isp relationship


class Rocket:
    """
    A single-stage rocket vehicle.

    Parameters
    ----------
    m0 : float
        Initial (wet) mass, kg — includes propellant.
    mf : float
        Final (dry/burnout) mass, kg — mass after all propellant for
        this stage is consumed.
    Isp : float
        Specific impulse, seconds. Typical chemical rockets: 200-300 s
        (solid), 250-450 s (liquid).
    burn_time : float
        Total burn duration, seconds. Used to compute the (assumed
        constant) propellant mass flow rate.
    A : float
        Reference frontal area, m^2 (for drag calculation).
    CD : float
        Drag coefficient (dimensionless). A simple constant is used
        here; real vehicles have CD varying with Mach number.
    """

    def __init__(self, m0, mf, Isp, burn_time, A, CD):
        if mf >= m0:
            raise ValueError("Final mass mf must be less than initial mass m0.")
        if burn_time <= 0:
            raise ValueError("burn_time must be positive.")

        self.m0 = m0
        self.mf = mf
        self.Isp = Isp
        self.burn_time = burn_time
        self.A = A
        self.CD = CD

        # Constant propellant mass flow rate (kg/s), assuming a
        # constant burn rate over burn_time (Curtis Eq. 11.11 setup).
        self.mdot = (m0 - mf) / burn_time

    @property
    def mass_ratio(self):
        """Initial-to-final mass ratio, n = m0/mf (Curtis Eq. 11.24)."""
        return self.m0 / self.mf

    def thrust(self, m):
        """
        Thrust magnitude, N (Curtis Eq. 11.22, rearranged):
            T = Isp * g0 * mdot

        Thrust is constant while propellant remains (m > mf), and zero
        once the stage has burned out.
        """
        if m > self.mf:
            return self.Isp * STANDARD_GRAVITY * self.mdot
        return 0.0

    def mass_flow_rate(self, m):
        """Current propellant mass flow rate, kg/s (0 once burned out)."""
        if m > self.mf:
            return self.mdot
        return 0.0

    def __repr__(self):
        return (
            f"Rocket(m0={self.m0:.1f} kg, mf={self.mf:.1f} kg, "
            f"Isp={self.Isp:.1f} s, burn_time={self.burn_time:.1f} s, "
            f"mass_ratio={self.mass_ratio:.2f})"
        )


def build_stage_rockets(stage_configs, payload_mass):
    """
    Build a list of per-stage Rocket objects for a multi-stage vehicle.

    Each stage is defined by its own propellant mass and structural
    (dry casing) mass. The propellant is what's burned; the structural
    mass is jettisoned (dropped) once that stage's propellant is spent.
    The payload mass rides on top of every stage until the final one.

    Parameters
    ----------
    stage_configs : list of dict
        Each dict must have keys: 'prop_mass', 'structural_mass',
        'Isp', 'burn_time', 'A', 'CD'. Stages are ordered bottom-up
        (index 0 = first stage to ignite/burn, e.g. the booster).
    payload_mass : float
        Mass, kg, that stays attached through all stages (e.g. the
        satellite/upper spacecraft).

    Returns
    -------
    rockets : list of Rocket
        One Rocket per stage, with m0/mf correctly reflecting the
        payload plus all not-yet-jettisoned stages above/below it.
    separation_masses : list of float
        Structural mass jettisoned immediately after each stage's
        propellant is spent (same order as `rockets`).
    """
    n = len(stage_configs)
    stage_total_masses = [
        cfg["prop_mass"] + cfg["structural_mass"] for cfg in stage_configs
    ]

    rockets = []
    separation_masses = []
    for i, cfg in enumerate(stage_configs):
        # Mass still attached at ignition of stage i: payload + this
        # stage + all stages above it that haven't ignited yet.
        m0 = payload_mass + sum(stage_total_masses[i:])
        mf = m0 - cfg["prop_mass"]  # after burn, before jettisoning casing

        rockets.append(
            Rocket(
                m0=m0,
                mf=mf,
                Isp=cfg["Isp"],
                burn_time=cfg["burn_time"],
                A=cfg["A"],
                CD=cfg["CD"],
            )
        )
        separation_masses.append(cfg["structural_mass"])

    return rockets, separation_masses
