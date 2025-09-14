EV Longitudinal Model — Tesla Model 3 LR AWD (2019–2020)

A lightweight, physics-based EV simulator for coursework and quick studies. It models full-throttle acceleration, a documented combined drive cycle (city + highway), and energy/range under three ambient conditions (Mild / Average / Cold). Built to be defensible (clear assumptions + calibration) and portable (Colab or VS Code, single file, no external data files).

What this does (at a glance)

0–100 km/h calibration to datasheet (4.6 s) using a traction coefficient μ search.

Drive cycle generator: reproducible “Combined” cycle (10× urban stop-go + highway cruise).

Energy & range estimation with aero/rolling/inertia, regen, driveline efficiency, and HVAC/accessory loads.

Three scenarios: Mild (23 °C), Average (12 °C), Cold (−10 °C), with temp-dependent air density and rolling resistance scaling.

Plots: acceleration curve, a(v), cycle speed trace, cumulative energy vs distance.

Assumptions block rendered with color in notebooks/terminals for easy reporting.

Why it’s defensible for a project

Anchored to two real-world checks:

0–100 km/h time (spec).

Average consumption tuned to ~162 Wh/km (EV-DB “real range” basis).

All assumptions are explicit (Cd·A, Crr, η, regen, HVAC, cycle shape).

Synthetic cycle is documented and reproducible; can be swapped for WLTP/UDDS with one flag.

Quick start
Option A — Google Colab

Open Colab and paste the Colab-ready EV longitudinal model script into a single cell.

Run. You’ll get four figures, a summary table, and a colorized assumptions block.

Option B — VS Code / Terminal

pip install numpy pandas matplotlib

Save the script as ev_model.py and run:

python ev_model.py


Terminal output includes ANSI-colored assumptions.

Key parameters (edit in script)

DT: time step (default 0.05 s).

USE_CUSTOM_CYCLE: False → synthetic cycle; set True and provide arrays custom_t_s, custom_speed_kmh to run official traces.

Vehicle constants from the datasheet: usable energy 73.5 kWh, peak power 324 kW, mass 1931 kg (+ payload), top speed 233 km/h.

Outputs you’ll see

0–100 km/h ≈ 4.6 s (calibrated μ printed).

Summary table per scenario: Wh/km, range (km), trip distance & duration, propulsive/regen/accessory energy.

Figures:

Speed vs time (WOT)

Acceleration vs speed

Cycle speed trace

Cumulative energy vs distance (Average)

Assumptions & calibration (high level)

Point-mass longitudinal dynamics; flat road, no wind.

Resistive forces: F = m·a + Crr·m·g + 0.5·ρ·CdA·v².

Driveline efficiency η and regen efficiency are aggregate constants.

HVAC/accessories modeled as constant power; auto-tuned in “Average” to match ~162 Wh/km.

Temperature affects air density; Crr scaled mildly with temperature.

Limitations: no grade/wind, no motor/inverter efficiency maps, no battery thermal model beyond HVAC.

Repo structure (suggested)
.
├─ ev_model.py              # single-file simulator (or Colab cell content)
├─ README.md                # this file
└─ /figures (optional)      # if you choose to save plots locally

Roadmap / nice-to-haves

WLTP/UDDS/HWFET loaders (CSV with t_s,speed_kmh).

Sensitivity analysis (±10% Cd·A / Crr / HVAC) with small charts.

Simple grade/wind inputs.

Non-constant driveline/regen efficiency maps.

License 

sample 

MIT (or your preferred license).

Citation / acknowledgment

Model inputs derived from the Tesla Model 3 LR AWD (2019–2020) public datasheet values and EV-Database real-world consumption anchor. Include appropriate citations in your report.
