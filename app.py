"""
Interactive web UI for the rocket trajectory simulator, built with
Streamlit. Lets the user configure a 1-3 stage launch vehicle and see
the resulting gravity-turn ascent trajectory.

Run locally with:
    streamlit run app.py

Can also be deployed for free on Streamlit Community Cloud
(share.streamlit.io) by pointing it at this file in your GitHub repo.
"""

import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from rocket_sim.vehicle import build_stage_rockets
from rocket_sim.simulate import run_multistage_ascent
from rocket_sim.visualization import rocket_marker
from rocket_sim.orbit import analyze_orbit, format_orbit_report

st.set_page_config(page_title="Rocket Trajectory Simulator", page_icon="🚀", layout="wide")

st.title("🚀 Rocket Trajectory Simulator")
st.markdown(
    "Interactive gravity-turn ascent simulation, built on the equations of "
    "motion from Curtis's *Orbital Mechanics for Engineering Students* "
    "(Ch. 11). Configure your vehicle in the sidebar and run the simulation."
)

# ---------------- Sidebar: vehicle configuration ----------------
st.sidebar.header("Vehicle Configuration")

payload_mass = st.sidebar.number_input(
    "Payload mass (kg)", min_value=1.0, value=500.0, step=50.0,
    help="Mass that rides through every stage, e.g. a satellite."
)

n_stages = st.sidebar.selectbox("Number of stages", options=[1, 2, 3], index=1)

stage_configs = []
for i in range(n_stages):
    with st.sidebar.expander(f"Stage {i + 1}", expanded=(i == 0)):
        prop_mass = st.number_input(
            f"Propellant mass (kg)", min_value=1.0,
            value=[38000.0, 6000.0, 2000.0][i] if i < 3 else 5000.0,
            step=500.0, key=f"prop_{i}",
        )
        structural_mass = st.number_input(
            f"Structural (dry) mass (kg)", min_value=1.0,
            value=[4000.0, 800.0, 300.0][i] if i < 3 else 500.0,
            step=100.0, key=f"struct_{i}",
        )
        isp = st.number_input(
            f"Specific impulse Isp (s)", min_value=50.0,
            value=[280.0, 320.0, 340.0][i] if i < 3 else 300.0,
            step=10.0, key=f"isp_{i}",
        )
        burn_time = st.number_input(
            f"Burn time (s)", min_value=1.0,
            value=[120.0, 180.0, 120.0][i] if i < 3 else 100.0,
            step=10.0, key=f"burn_{i}",
        )
        area = st.number_input(
            f"Frontal area A (m^2)", min_value=0.0,
            value=[3.0, 1.2, 0.8][i] if i < 3 else 1.0,
            step=0.1, key=f"area_{i}",
        )
        cd = st.number_input(
            f"Drag coefficient CD", min_value=0.0,
            value=[0.3, 0.25, 0.2][i] if i < 3 else 0.25,
            step=0.05, key=f"cd_{i}",
        )
        stage_configs.append({
            "prop_mass": prop_mass,
            "structural_mass": structural_mass,
            "Isp": isp,
            "burn_time": burn_time,
            "A": area,
            "CD": cd,
        })

st.sidebar.header("Ascent Profile")
kick_speed = st.sidebar.slider(
    "Pitchover kick speed (m/s)", min_value=10.0, max_value=200.0, value=50.0,
    help="Speed at which the vehicle ends its vertical rise and begins the gravity turn."
)
kick_angle = st.sidebar.slider(
    "Pitchover kick angle (deg)", min_value=80.0, max_value=89.9, value=89.0,
    help="Flight path angle immediately after the pitchover kick (90 = still vertical)."
)
coast_time = st.sidebar.slider(
    "Post-burnout coast time (s)", min_value=0.0, max_value=300.0, value=60.0
)

run = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ---------------- Run simulation and cache results in session_state ----------------
# Streamlit reruns this whole script on every button click (including the
# "Play animation" button below). Without session_state, clicking that
# button would lose the results computed by "Run Simulation" -- which is
# exactly the "page resets" bug this fixes: results are only recomputed
# when the user actually clicks "Run Simulation" again, and otherwise
# persist across reruns.
if run:
    try:
        rockets, separation_masses = build_stage_rockets(stage_configs, payload_mass)
        total_mass = rockets[0].m0
        liftoff_thrust = rockets[0].thrust(rockets[0].m0)
        tw_ratio = liftoff_thrust / (total_mass * 9.80665)

        if tw_ratio < 1.0:
            st.session_state["result"] = None
            st.session_state["tw_error"] = tw_ratio
        else:
            with st.spinner("Integrating equations of motion..."):
                result = run_multistage_ascent(
                    rockets, separation_masses,
                    kick_speed=kick_speed, kick_angle_deg=kick_angle,
                    coast_time=coast_time, max_step=0.3,
                )
            st.session_state["result"] = result
            st.session_state["rockets"] = rockets
            st.session_state["separation_masses"] = separation_masses
            st.session_state["tw_ratio"] = tw_ratio
            st.session_state["tw_error"] = None
    except Exception as e:
        st.session_state["result"] = None
        st.session_state["run_error"] = str(e)

# ---------------- Main panel: display cached results (if any) ----------------
if st.session_state.get("tw_error"):
    st.error(
        f"⚠️ Thrust-to-weight ratio at liftoff is {st.session_state['tw_error']:.2f}, "
        "which is less than 1 -- this vehicle cannot lift off the ground. "
        "Increase thrust (more propellant / shorter burn time) or reduce mass."
    )
elif st.session_state.get("run_error"):
    st.error(f"Simulation failed: {st.session_state['run_error']}")
elif st.session_state.get("result") is not None:
    result = st.session_state["result"]
    rockets = st.session_state["rockets"]
    separation_masses = st.session_state["separation_masses"]
    tw_ratio = st.session_state["tw_ratio"]

    t = result["t"]
    v = result["v"]
    gamma = result["gamma_deg"]
    x = result["x"] / 1000.0
    h = result["h"] / 1000.0
    m = result["m"]
    sep_times = result["stage_separation_times"]

    st.success("Simulation complete.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liftoff T/W ratio", f"{tw_ratio:.2f}")
    col2.metric("Max altitude", f"{h.max():,.1f} km")
    col3.metric("Max speed", f"{v.max():,.0f} m/s")
    col4.metric("Final downrange", f"{x[-1]:,.1f} km")

    # ---------------- Orbital insertion analysis ----------------
    st.subheader("🛰️ Orbital Insertion Analysis")
    st.caption(
        "Analyzed at the end of the simulation, treating the vehicle as coasting "
        "under gravity alone (a two-body Kepler orbit) from that point on."
    )
    orbit_analysis = analyze_orbit(v[-1], gamma[-1], h[-1] * 1000)  # h back to meters

    ocol1, ocol2, ocol3 = st.columns(3)
    ocol1.metric("Eccentricity", f"{orbit_analysis['eccentricity']:.3f}")
    ocol2.metric("Local circular velocity", f"{orbit_analysis['v_circular']:,.0f} m/s")
    if orbit_analysis["perigee_altitude"] is not None:
        peri_km = orbit_analysis["perigee_altitude"] / 1000
        ocol3.metric("Perigee altitude", f"{peri_km:,.1f} km")

    if orbit_analysis["perigee_altitude"] is not None and orbit_analysis["perigee_altitude"] < 0:
        st.warning(
            f"🔻 {orbit_analysis['orbit_type']} Apogee altitude: "
            f"{orbit_analysis['apogee_altitude']/1000:,.1f} km."
        )
    elif "Hyperbolic" in orbit_analysis["orbit_type"] or "Parabolic" in orbit_analysis["orbit_type"]:
        st.warning(f"🚀 {orbit_analysis['orbit_type']}")
    else:
        st.success(
            f"✅ {orbit_analysis['orbit_type']} "
            f"Perigee: {orbit_analysis['perigee_altitude']/1000:,.1f} km, "
            f"Apogee: {orbit_analysis['apogee_altitude']/1000:,.1f} km, "
            f"Period: {orbit_analysis['period']/60:,.1f} min."
        )
        if orbit_analysis["delta_v_to_circularize"]:
            st.info(
                f"To circularize at apogee, a burn of "
                f"{orbit_analysis['delta_v_to_circularize']:,.1f} m/s would be needed."
            )

    with st.expander("Why might this be suborbital or hyperbolic instead of a clean orbit?"):
        st.markdown(
            "This simulator applies **one** deliberate pitchover kick and then lets "
            "gravity do all further steering (a passive gravity turn) — there's no "
            "active guidance loop continuously correcting the trajectory. Real launch "
            "vehicles constantly adjust their thrust vector throughout ascent to hit a "
            "precise target insertion state. Hitting that exactly with a single open-loop "
            "kick is possible but numerically delicate — try adjusting stage propellant "
            "masses, burn times, and the pitchover kick angle/speed in the sidebar to see "
            "how sensitive orbital insertion is to these choices."
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    def mark_stages(ax):
        for i, st_time in enumerate(sep_times):
            ax.axvline(st_time, color="gray", linestyle="--", linewidth=1,
                       label="Stage separation" if i == 0 else None)

    axes[0, 0].plot(t, h, color="tab:blue")
    mark_stages(axes[0, 0])
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[0, 0].set_title("Altitude vs Time")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, v, color="tab:red")
    mark_stages(axes[0, 1])
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("Speed vs Time")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, gamma, color="tab:green")
    mark_stages(axes[1, 0])
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Flight path angle (deg)")
    axes[1, 0].set_title("Flight Path Angle vs Time")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, h, color="tab:purple")
    axes[1, 1].set_xlabel("Downrange distance (km)")
    axes[1, 1].set_ylabel("Altitude (km)")
    axes[1, 1].set_title("Ground Track Profile")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- Animated flight (in-app) ----------------
    st.subheader("🎬 Animated ascent")
    animate = st.button("▶ Play animation", use_container_width=True)

    if animate:
        n_frames = 80
        anim_idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
        t_f, x_f, h_f, v_f, gamma_f = (
            t[anim_idx], x[anim_idx], h[anim_idx], v[anim_idx], gamma[anim_idx]
        )
        sep_indices = [np.searchsorted(t, st_time) for st_time in sep_times]

        anim_fig, anim_ax = plt.subplots(figsize=(9, 5.5))
        placeholder = st.empty()

        for frame in range(n_frames):
            anim_ax.clear()
            anim_ax.plot(x, h, color="lightgray", linewidth=0.8, linestyle=":", alpha=0.5)
            anim_ax.plot(x_f[:frame + 1], h_f[:frame + 1], color="tab:red", linewidth=2.5)
            anim_ax.plot(
                [x_f[frame]], [h_f[frame]],
                marker=rocket_marker(gamma_f[frame]), color="tab:red",
                markersize=20, markeredgecolor="darkred", linestyle="None",
            )
            passed = [i for i, st_time in enumerate(sep_times) if t_f[frame] >= st_time]
            for i in passed:
                anim_ax.plot(x[sep_indices[i]], h[sep_indices[i]],
                              marker="x", color="black", markersize=10)

            anim_ax.set_xlabel("Downrange distance (km)")
            anim_ax.set_ylabel("Altitude (km)")
            anim_ax.set_xlim(0, max(x.max() * 1.05, 1))
            anim_ax.set_ylim(0, max(h.max() * 1.1, 1))
            anim_ax.set_title(
                f"t = {t_f[frame]:.1f}s   alt = {h_f[frame]:.1f} km   "
                f"speed = {v_f[frame]:.0f} m/s"
            )
            anim_ax.grid(alpha=0.3)

            placeholder.pyplot(anim_fig)
            time.sleep(0.05)

        plt.close(anim_fig)

    with st.expander("Stage details"):
        for i, r in enumerate(rockets):
            st.write(f"**Stage {i + 1}:** {r}")
            st.write(f"Structural mass jettisoned at separation: {separation_masses[i]:.1f} kg")

else:
    st.info("Configure your vehicle in the sidebar, then click **Run Simulation**.")
