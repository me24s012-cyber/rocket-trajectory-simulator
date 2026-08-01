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
