# Colab-ready EV longitudinal model (no files saved)
# Vehicle: Tesla Model 3 Long Range AWD (2019–2020)
# Tasks: robust 0–100 calibration, Combined drive cycle, Mild/Average/Cold sims, 4 plots, summary, assumptions (colored)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to enable HTML rendering (for Colab/Jupyter). Falls back to ANSI in terminals like VS Code.
try:
    from IPython.display import display, HTML
    _IN_NOTEBOOK = True
except Exception:
    _IN_NOTEBOOK = False

# -----------------------
# Config
# -----------------------
USE_CUSTOM_CYCLE = False     # True -> provide custom_t_s & custom_speed_kmh below
DT = 0.05                    # s (keep 0.05–0.1)

# If using a custom cycle, define:
# custom_t_s = np.array([...], dtype=float)
# custom_speed_kmh = np.array([...], dtype=float)

# -----------------------
# Vehicle (EV Database: Model 3 LR AWD 2019–2020)
# -----------------------
USABLE_KWH      = 73.5       # kWh
P_MAX           = 324e3      # W (aggregate)
MASS_UNLADEN    = 1931.0     # kg
TOP_SPEED_KMH   = 233.0      # km/h
ZERO_TO_100_S   = 4.6        # s target
DRIVER_PAYLOAD  = 75.0       # kg
MASS            = MASS_UNLADEN + DRIVER_PAYLOAD

# Modeling assumptions (mark in red in report)
G               = 9.81
CDA             = 0.50       # m^2 (effective Cd*A)
CRR_BASE        = 0.010
ETA_DRIVELINE   = 0.90
REGEN_EFF       = 0.60
REGEN_POWER_CAP = 100e3      # W

# -----------------------
# Helpers
# -----------------------
def air_density(temp_c=15.0, pressure_pa=101325.0):
    R = 287.05
    T = temp_c + 273.15
    return pressure_pa / (R * T)

def resistive_force(v, rho, crr):
    return crr*MASS*G + 0.5*rho*CDA*v*v

def max_drive_force(v, mu):
    p_wheel = ETA_DRIVELINE * P_MAX
    f_power = 1e12 if v < 0.1 else p_wheel / v
    f_traction = mu * MASS * G
    return min(f_power, f_traction)

def simulate_full_throttle(mu, temp_c=15.0, crr=CRR_BASE, t_max=30.0, dt=DT):
    """
    IMPORTANT: fixed time window (don’t stop on speed) so we always cross 100 km/h,
    then interpolate.
    """
    rho = air_density(temp_c)
    n = int(t_max / dt)
    t = np.zeros(n+1)
    v = np.zeros(n+1)
    a = np.zeros(n+1)
    for i in range(n):
        f_drive = max_drive_force(v[i], mu)
        f_res   = resistive_force(v[i], rho, crr)
        a[i]    = max(0.0, (f_drive - f_res) / MASS)  # no braking at WOT
        v[i+1]  = v[i] + a[i]*dt
        t[i+1]  = t[i] + dt
    a[-1] = a[-2]
    return t, v, a

def t_at_speed(times, speeds, target_kmh):
    v_target = target_kmh / 3.6
    speeds = np.asarray(speeds); times = np.asarray(times)
    idx = np.where(speeds >= v_target)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return times[0]
    v0, v1 = speeds[i-1], speeds[i]
    t0, t1 = times[i-1], times[i]
    return t0 + (v_target - v0) * (t1 - t0) / max(v1 - v0, 1e-12)

def calibrate_mu(target_t=ZERO_TO_100_S, dt=DT):
    low, high = 0.4, 1.2
    for _ in range(40):
        mid = 0.5*(low+high)
        t_acc, v_acc, _ = simulate_full_throttle(mid, t_max=30.0, dt=dt)
        t100 = t_at_speed(t_acc, v_acc, 100.0)
        if t100 is None:
            t_acc, v_acc, _ = simulate_full_throttle(mid, t_max=60.0, dt=dt)
            t100 = t_at_speed(t_acc, v_acc, 100.0)
        if t100 is None:
            low = mid; continue
        if t100 > target_t:
            low = mid
        else:
            high = mid
    return 0.5*(low+high)

# -----------------------
# Drive cycle
# -----------------------
def build_synthetic_combined_cycle(dt=DT):
    """
    Documented "Combined" cycle:
      Urban: 10x [0→50 km/h @~0.9 m/s², 60 s @50, decel @-0.9 to 0, dwell 10 s]
      Highway: 0→110 @1.0 m/s², 600 s @110, decel @-0.6 to 0
    """
    t = [0.0]; v = [0.0]
    def urban_phase(target_kmh, accel=0.9, decel=-0.9, cruise_s=60.0, dwell_s=10.0, repeats=10):
        for _ in range(repeats):
            v1 = target_kmh/3.6
            # accel
            t_acc = max(0.0, (v1 - v[-1])/accel)
            for _ in range(int(max(1, t_acc/dt))):
                t.append(t[-1]+dt); v.append(min(v1, v[-1]+accel*dt))
            # cruise
            for _ in range(int(cruise_s/dt)):
                t.append(t[-1]+dt); v.append(v1)
            # decel to 0
            t_dec = max(0.0, (0.0 - v[-1])/decel)
            for _ in range(int(max(1, t_dec/dt))):
                t.append(t[-1]+dt); v.append(max(0.0, v[-1]+decel*dt))
            # dwell
            for _ in range(int(dwell_s/dt)):
                t.append(t[-1]+dt); v.append(0.0)

    urban_phase(50.0, 0.9, -0.9, 60.0, 10.0, 10)

    # highway
    target = 110.0/3.6
    a = 1.0
    t_acc = (target - v[-1])/a
    for _ in range(int(max(1, t_acc/dt))):
        t.append(t[-1]+dt); v.append(min(target, v[-1]+a*dt))
    for _ in range(int(600.0/dt)):
        t.append(t[-1]+dt); v.append(target)
    a = -0.6
    t_dec = (0.0 - v[-1])/a
    for _ in range(int(max(1, t_dec/dt))):
        t.append(t[-1]+dt); v.append(max(0.0, v[-1]+a*dt))

    return np.array(t), np.array(v)

def get_cycle():
    if USE_CUSTOM_CYCLE:
        assert 'custom_t_s' in globals() and 'custom_speed_kmh' in globals(), \
            "Define custom_t_s and custom_speed_kmh"
        t = np.asarray(custom_t_s, dtype=float)
        v = np.asarray(custom_speed_kmh, dtype=float)/3.6
        return t, v, "Custom cycle"
    else:
        t, v = build_synthetic_combined_cycle()
        return t, v, "Synthetic Combined (City + Highway)"

# -----------------------
# Energy on cycle
# -----------------------
def simulate_cycle_energy(t, v, temp_c=12.0, crr=CRR_BASE, accessories_kw=1.5):
    rho = air_density(temp_c)
    e_propulsive = 0.0
    e_regen = 0.0
    for i in range(1, len(t)):
        dt_i = t[i]-t[i-1]
        v_mid = 0.5*(v[i]+v[i-1])
        f_res = resistive_force(max(v_mid,0.0), rho, crr)
        a_dem = (v[i]-v[i-1]) / max(dt_i, 1e-9)
        f_wheel = f_res + MASS*a_dem
        p_wheel = f_wheel * max(v_mid,0.0)
        if p_wheel >= 0:
            p_batt = p_wheel / max(ETA_DRIVELINE, 1e-3)
            e_propulsive += p_batt * dt_i
        else:
            p_regen_mech = -p_wheel
            p_regen_elec = min(p_regen_mech*REGEN_EFF, REGEN_POWER_CAP)
            e_regen += p_regen_elec * dt_i

    e_access = accessories_kw*1e3*(t[-1]-t[0])
    distance_km = np.trapezoid(v, t)/1000.0  # modern NumPy integration
    e_batt_Wh = (e_propulsive - e_regen + e_access)/3600.0
    wh_per_km = e_batt_Wh / max(distance_km, 1e-9)
    return {
        "E_propulsive_Wh": e_propulsive/3600.0,
        "E_regen_Wh": e_regen/3600.0,
        "E_accessories_Wh": e_access/3600.0,
        "Wh_per_km": wh_per_km,
        "distance_km": distance_km,
        "duration_min": (t[-1]-t[0])/60.0
    }

def calibrate_accessories_for_avg(t, v, target_whpkm=162.0, temp_c=12.0, crr=CRR_BASE):
    low, high = 0.0, 5.0
    for _ in range(35):
        mid = 0.5*(low+high)
        res = simulate_cycle_energy(t, v, temp_c=temp_c, crr=crr, accessories_kw=mid)
        if res["Wh_per_km"] < target_whpkm:
            low = mid
        else:
            high = mid
    return 0.5*(low+high)

def cumulative_energy_profile(t, v, temp_c, crr, accessories_kw):
    rho = air_density(temp_c)
    e_batt = 0.0
    E = [0.0]; D = [0.0]
    for i in range(1, len(t)):
        dt_i = t[i]-t[i-1]
        v_mid = 0.5*(v[i]+v[i-1])
        f_res = resistive_force(max(v_mid,0.0), rho, crr)
        a_dem = (v[i]-v[i-1])/max(dt_i,1e-9)
        f_wheel = f_res + MASS*a_dem
        p_wheel = f_wheel * max(v_mid,0.0)
        if p_wheel >= 0:
            p_batt = p_wheel/max(ETA_DRIVELINE,1e-3)
            e_batt += p_batt*dt_i
        else:
            p_regen_mech = -p_wheel
            p_regen_elec = min(p_regen_mech*REGEN_EFF, REGEN_POWER_CAP)
            e_batt -= p_regen_elec*dt_i
        e_batt += accessories_kw*1e3*dt_i
        E.append(e_batt/3600.0)
        D.append(D[-1] + v_mid*dt_i/1000.0)
    return np.array(D), np.array(E)

def range_km(whpkm): return USABLE_KWH*1000.0 / whpkm

# -----------------------
# Run
# -----------------------
# 1) Calibrate μ robustly (fixed window + interpolation)
mu_cal = calibrate_mu(ZERO_TO_100_S, dt=DT)
t_acc, v_acc, a_acc = simulate_full_throttle(mu_cal, t_max=30.0, dt=DT)
t100 = t_at_speed(t_acc, v_acc, 100.0)
if t100 is None:
    t_acc, v_acc, a_acc = simulate_full_throttle(mu_cal, t_max=60.0, dt=DT)
    t100 = t_at_speed(t_acc, v_acc, 100.0)

print(f"Calibrated traction μ = {mu_cal:.3f}")
print(f"0–100 km/h achieved ≈ {t100:.2f} s (target {ZERO_TO_100_S:.1f} s)")

# Plot 1: speed vs time (full throttle)
plt.figure(); plt.plot(t_acc, v_acc*3.6)
plt.xlabel("Time (s)"); plt.ylabel("Speed (km/h)")
plt.title(f"Full-throttle acceleration (0–100 ≈ {t100:.2f} s)")
plt.grid(True); plt.show()

# Plot 2: acceleration vs speed
plt.figure(); plt.plot(v_acc*3.6, a_acc)
plt.xlabel("Speed (km/h)"); plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration vs Speed (Full Throttle)")
plt.grid(True); plt.show()

# 2) Drive cycle
t_cycle, v_cycle, cycle_name = get_cycle()
plt.figure(); plt.plot(t_cycle, v_cycle*3.6)
plt.xlabel("Time (s)"); plt.ylabel("Speed (km/h)")
plt.title(f"Drive Cycle Speed Trace — {cycle_name}")
plt.grid(True); plt.show()

# 3) Three scenarios
temp_mild, temp_avg, temp_cold = 23.0, 12.0, -10.0
crr_mild, crr_avg, crr_cold = CRR_BASE*0.98, CRR_BASE*1.00, CRR_BASE*1.15

acc_kw_avg = calibrate_accessories_for_avg(t_cycle, v_cycle, target_whpkm=162.0, temp_c=temp_avg, crr=crr_avg)
acc_kw_mild = max(0.2, acc_kw_avg - 1.0)
acc_kw_cold = acc_kw_avg + 2.0

res_mild = simulate_cycle_energy(t_cycle, v_cycle, temp_c=temp_mild, crr=crr_mild, accessories_kw=acc_kw_mild)
res_avg  = simulate_cycle_energy(t_cycle, v_cycle, temp_c=temp_avg,  crr=crr_avg,  accessories_kw=acc_kw_avg)
res_cold = simulate_cycle_energy(t_cycle, v_cycle, temp_c=temp_cold, crr=crr_cold, accessories_kw=acc_kw_cold)

summary = pd.DataFrame([
    {"Scenario":"Mild (23°C)","Temp (°C)":temp_mild,"Accessories (kW)":round(acc_kw_mild,3),
     "Crr":round(crr_mild,4),"Wh/km":round(res_mild["Wh_per_km"],1),
     "Range (km)":round(range_km(res_mild["Wh_per_km"]),0),
     "Distance (km)":round(res_mild["distance_km"],2),"Duration (min)":round(res_mild["duration_min"],1),
     "E_prop (kWh)":round(res_mild["E_propulsive_Wh"]/1000.0,2),"E_regen (kWh)":round(res_mild["E_regen_Wh"]/1000.0,2),
     "E_access (kWh)":round(res_mild["E_accessories_Wh"]/1000.0,2)},
    {"Scenario":"Average (~12°C)","Temp (°C)":temp_avg,"Accessories (kW)":round(acc_kw_avg,3),
     "Crr":round(crr_avg,4),"Wh/km":round(res_avg["Wh_per_km"],1),
     "Range (km)":round(range_km(res_avg["Wh_per_km"]),0),
     "Distance (km)":round(res_avg["distance_km"],2),"Duration (min)":round(res_avg["duration_min"],1),
     "E_prop (kWh)":round(res_avg["E_propulsive_Wh"]/1000.0,2),"E_regen (kWh)":round(res_avg["E_regen_Wh"]/1000.0,2),
     "E_access (kWh)":round(res_avg["E_accessories_Wh"]/1000.0,2)},
    {"Scenario":"Cold (-10°C)","Temp (°C)":temp_cold,"Accessories (kW)":round(acc_kw_cold,3),
     "Crr":round(crr_cold,4),"Wh/km":round(res_cold["Wh_per_km"],1),
     "Range (km)":round(range_km(res_cold["Wh_per_km"]),0),
     "Distance (km)":round(res_cold["distance_km"],2),"Duration (min)":round(res_cold["duration_min"],1),
     "E_prop (kWh)":round(res_cold["E_propulsive_Wh"]/1000.0,2),"E_regen (kWh)":round(res_cold["E_regen_Wh"]/1000.0,2),
     "E_access (kWh)":round(res_cold["E_accessories_Wh"]/1000.0,2)}
])
print("\n=== Energy & Range Summary ===")
print(summary.to_string(index=False))

# Plot 4: cumulative energy (Average)
D_km, E_Wh = cumulative_energy_profile(t_cycle, v_cycle, temp_avg, crr_avg, acc_kw_avg)
plt.figure(); plt.plot(D_km, E_Wh)
plt.xlabel("Distance (km)"); plt.ylabel("Cumulative Battery Energy (Wh)")
plt.title("Average scenario: Cumulative Energy vs Distance")
plt.grid(True); plt.show()

# -----------------------
# Assumptions (colored output in both Notebook & Terminal)
# -----------------------
# HTML (for Colab/Jupyter) + ANSI (for VS Code terminal)
def _assumptions_html():
    def red(s): return f"<span style='color:#c1121f; font-weight:700'>{s}</span>"
    return f"""
<div style="font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
            white-space: pre-wrap; line-height:1.35; font-size:14px">
<b>Tesla Model 3 LR AWD (2019–2020) — Simulation Assumptions</b>

Data (EV Database): usable {USABLE_KWH:.1f} kWh; peak power {P_MAX/1000:.0f} kW; mass {MASS_UNLADEN:.0f} kg (+{DRIVER_PAYLOAD:.0f} kg payload);
top speed {TOP_SPEED_KMH:.0f} km/h; 0–100 target {ZERO_TO_100_S:.1f} s.

Model: longitudinal point-mass; drag+rolling; flat, no wind; driveline η={ETA_DRIVELINE:.2f};
regen eff={REGEN_EFF:.2f}, cap={REGEN_POWER_CAP/1000:.0f} kW;
Cd·A={red(f'{CDA:.2f} m² (assumed)')}; Crr_base={red(f'{CRR_BASE:.3f} (assumed)')}.

Calibrations:
- Traction coefficient μ tuned so 0–100 ≈ {t100:.2f} s ⇒ μ≈{mu_cal:.3f}.
- "Average" accessory load auto-tuned so combined-cycle consumption ≈ 162 Wh/km (EV-DB real-range basis).

Environment setups:
- Mild 23°C ({red('Crr×0.98 assumption')}), Average 12°C ({red('Crr baseline')}),
  Cold −10°C ({red('Crr×1.15 assumption')}), air density via ideal gas.
- HVAC (kW): Mild {acc_kw_mild:.2f}, Average {acc_kw_avg:.2f}, Cold {acc_kw_cold:.2f}.

Drive profile used:
- Urban 10× [0→50 km/h accel ~0.9 m/s², 60 s @50, decel −0.9 m/s² to 0, dwell 10 s]
- Highway [0→110 km/h @1.0 m/s², 600 s @110, decel −0.6 m/s² to 0]
{red('Note: Synthetic, documented stand-in; not an official WLTP/UDDS trace.')}

Key items to {red('mark in red in your report')}:
- {red('CdA, Crr, driveline η, regen efficiency/cap, HVAC loads are assumptions.')}
- {red('Synthetic drive cycle used for modeling (not official).')}
- {red('AWD torque split and detailed motor/inverter maps abstracted by aggregate power.')}
- {red('Battery thermal beyond HVAC not modeled; flat road, no wind.')}

Repro tips:
- Fixed time step DT = {DT:.2f} s; full-throttle window + interpolation to compute 0–100.
- Average case calibrated to ~162 Wh/km; Mild/Cold adjust ρ, Crr, HVAC.
</div>
"""

def _assumptions_ansi():
    RED = "\033[1;31m"; RESET = "\033[0m"
    return (
f"""Tesla Model 3 LR AWD (2019–2020) — Simulation Assumptions

Data (EV Database): usable {USABLE_KWH:.1f} kWh; peak power {P_MAX/1000:.0f} kW; mass {MASS_UNLADEN:.0f} kg (+{DRIVER_PAYLOAD:.0f} kg payload);
top speed {TOP_SPEED_KMH:.0f} km/h; 0–100 target {ZERO_TO_100_S:.1f} s.

Model: longitudinal point-mass; drag+rolling; flat, no wind; driveline η={ETA_DRIVELINE:.2f};
regen eff={REGEN_EFF:.2f}, cap={REGEN_POWER_CAP/1000:.0f} kW;
{RED}Cd·A={CDA:.2f} m² (assumed){RESET}; {RED}Crr_base={CRR_BASE:.3f} (assumed){RESET}.

Calibrations:
- μ tuned so 0–100 ≈ {t100:.2f} s ⇒ μ≈{mu_cal:.3f}.
- "Average" accessory load auto-tuned so combined-cycle consumption ≈ 162 Wh/km (EV-DB real-range basis).

Environment setups:
- Mild 23°C ({RED}Crr×0.98 assumption{RESET}), Average 12°C (baseline),
  Cold −10°C ({RED}Crr×1.15 assumption{RESET}); air density via ideal gas.
- HVAC (kW): Mild {acc_kw_mild:.2f}, Avg {acc_kw_avg:.2f}, Cold {acc_kw_cold:.2f}.

Drive profile used:
- Urban 10× [0→50 km/h accel ~0.9 m/s², 60 s @50, decel −0.9 m/s² to 0, dwell 10 s]
- Highway [0→110 km/h @1.0 m/s², 600 s @110, decel −0.6 m/s² to 0]
{RED}Synthetic documented trace; not WLTP/UDDS.{RESET}

Caveats:
- {RED}CdA, Crr, driveline η, regen efficiency/cap, HVAC loads are assumptions.{RESET}
- {RED}AWD torque split & detailed motor/inverter maps abstracted by aggregate power.{RESET}
- {RED}Battery thermal beyond HVAC not modeled; flat road, no wind.{RESET}

Repro tips:
- DT = {DT:.2f} s; 0–100 via interpolation; Average ~162 Wh/km; Mild/Cold adjust ρ, Crr, HVAC.
"""
    )

# Display assumptions with color in the appropriate environment
if _IN_NOTEBOOK:
    display(HTML(_assumptions_html()))
else:
    print(_assumptions_ansi())
