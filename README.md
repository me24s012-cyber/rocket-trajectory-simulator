# Rocket Trajectory Simulator

A numerical simulation of a launch vehicle's ascent trajectory, including the
gravity turn maneuver, atmospheric drag, variable mass depletion, and
multi-stage separation.

## Status
🚧 Work in progress — building day by day. See progress below.

## Physics model

The simulator integrates the equations of motion for a launch vehicle
following a gravity turn trajectory, based on the formulation in Curtis,
*Orbital Mechanics for Engineering Students* (Ch. 11):

- **Speed:** `dv/dt = T/m - D/m - g*sin(γ)`
- **Flight path angle:** `dγ/dt = -(1/v)*[g - v²/(R_E+h)]*cos(γ)`
- **Downrange distance:** `dx/dt = [R_E/(R_E+h)]*v*cos(γ)`
- **Altitude:** `dh/dt = v*sin(γ)`
- **Mass depletion:** `dm/dt = -ṁ_e`
- **Thrust:** `T = I_sp * g0 * ṁ_e`
- **Drag:** `D = 0.5 * ρ(h) * v² * A * C_D`

where `ρ(h)` comes from an exponential ISA atmosphere approximation.

## Project structure
```
rocket_sim/
├── atmosphere.py   # ISA density model
├── vehicle.py      # Rocket class: mass, thrust, Isp, staging
├── dynamics.py      # equations of motion (ODE system)
└── simulate.py      # solve_ivp integration + event detection
examples/            # example runs (e.g. Falcon-9-like vehicle)
notebooks/           # exploratory analysis & plots
```

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Validation
Early results are checked against Curtis's Example 11.1 (closed-form solution
for a vertical sounding rocket, no drag) to confirm the numerical integration
is correct before adding drag and gravity-turn dynamics.

## References
Curtis, H.D. *Orbital Mechanics for Engineering Students*, Chapter 11: Rocket
Vehicle Dynamics.
