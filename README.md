# EV Longitudinal Model 

A Colab/Jupyter-friendly Python script that simulates **longitudinal dynamics** for a **Tesla Model 3 LR AWD (2019–2020)**, calibrates **0–100 km/h**, runs a **combined drive cycle (city + highway)** or your **custom cycle**, and estimates **energy use & range** across **Mild / Average / Cold** scenarios — **without saving files**.

---

## Features
- **0–100 km/h calibration:** tunes traction coefficient **μ** to hit **≈ 4.6 s** target.  
- **Drive cycles:** built-in **Synthetic Combined (City + Highway)** or plug in your own (`USE_CUSTOM_CYCLE=True`).  
- **Energy & range:** computes **propulsive**, **regen**, **accessories**, **Wh/km**, and **range (km)** using **73.5 kWh usable**.  
- **Accessory auto-tune:** adjusts accessory load so **Average** case ≈ **162 Wh/km** baseline.  
- **Plots (x4):** 
  1) Speed vs Time (full throttle)  
  2) Acceleration vs Speed (WOT)  
  3) Drive Cycle Speed Trace  
  4) Cumulative Energy vs Distance  
- **Readable summary:** prints a tidy pandas table for Mild / Average / Cold.

---

## Model & Key Assumptions
- Point-mass longitudinal model (flat road, no wind) with **aero drag + rolling resistance**.  
- Aggregate driveline with **efficiency (η)**, **regen efficiency/cap**, **Cd·A**, **Crr**.  
- **Temperatures:** Mild 23 °C, Average 12 °C, Cold −10 °C (affects air density and Crr).  
- The script highlights assumptions (e.g., **CdA, Crr, η, regen, HVAC loads**) **in red** in notebook/terminal output.

> Vehicle refs in code: `USABLE_KWH=73.5`, `P_MAX=324e3 W`, `MASS≈2006 kg (incl. driver)`, `ZERO_TO_100_S=4.6`.

---

## Quick Start
**Requirements:** Python 3.10+, NumPy, Pandas, Matplotlib.

```bash
pip install numpy pandas matplotlib
