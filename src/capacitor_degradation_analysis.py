"""
Capacitor Fatigue Post-Analysis
================================
Accepts CSV files produced by either the old acquisition script (7 columns)
or the new one (12/13 columns). Detects format automatically.

Degradation analysis — corrected for PSU filter capacitor pollution:

  METHOD 2 — Single-Exponential Discharge Tau  ← PRIMARY FATIGUE METRIC
    · V(t) = V0 · exp(−t/τ) fit to the clean [5%-50%] tail of each discharge.
    · τ / R_discharge → C_test (the fatigue signal).
    · R_discharge measured empirically from V/I during clean decay.
    · PSU relay hold-off plateau (~0.9 s at discharge onset) is skipped
      by finding the first sample where dV/dt < −5 V/s.

  ADDITIONAL METRICS — discharge energy and V@2s. Tau-independent,
    no curve fitting, cross-validate Method 2.

  CHARGE-PHASE DIAGNOSTICS (NEW) — three more tau-independent, no-fit
    metrics extracted from the charge phase, all valid at 100 Hz:
      · I_leak_mA          — steady-state leakage current on the charge
                              plateau. Independent aging mechanism from
                              capacitance/tau (oxide degradation vs.
                              electrolyte drying). Fully reliable at any
                              format — it's a steady-state DC reading, not
                              affected by the shunt-undersampling issue
                              below.
      · efficiency_pct     — E_discharge / E_charge round-trip energy
                              efficiency. Falling efficiency = rising
                              dissipative losses (ESR + leakage). CAUTION:
                              for 'minimal' format files this inherits the
                              SAME shunt-undersampling problem that made
                              Method 1 unreliable (see below) — the ramp
                              segment of E_charge is undercounted. Treat as
                              supplementary for minimal-format data; fully
                              reliable for 'full' format.
      · time_to_plateau_s  — duration of the charge ramp. Uses only the
                              voltage channel (timing, not current
                              magnitude), so it does NOT share Method 1's
                              shunt problem. Independent charge-side
                              cross-check on the discharge-derived C_test.

  ESR PROXY  (1 kHz files only) — τ_initial vs τ_tail. Skipped below
    900 Hz; ADC quantisation noise dominates at 100 Hz.

  METHOD 1 (Charge Phase Integral) is defined below but DISABLED by
  default in this pipeline — see method1_charge_integral() docstring.
  RUL linear-extrapolation is also disabled — see notes in print_report
  and _plot_combined.

Usage:
    python capacitor_degradation_analysis.py  [path/to/daq_data.csv | path/to/folder]

    If no argument is given, defaults to this script's own directory and
    runs in folder mode (processes every daq_data_*.csv found there).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import trapezoid
from scipy.optimize import curve_fit
from scipy.stats import linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
R_SHUNT           = 220.0      # Ω  (series current-sense resistor)
R_OUTPUT          = 32_000.0   # Ω  (voltage-divider bleed)
ATTENUATION       = 3.2        # CH0 hardware attenuation factor
C_NOMINAL_uF      = 1000.0     # μF  rated capacitance of DUT

# NOTE: R_OUTPUT / ATTENUATION are carried over from the DAQ stage purely
# for documentation. The acquisition script already applies both when it
# writes capacitor_voltage / capacitor_current to the CSV, so this analysis
# script never re-applies them — they are not used in any calculation below.

# Discharge fitting
RELAY_HOLDOFF_DVDT = -5.0      # V/s  threshold to detect end of relay hold-off
DISCHARGE_V_FLOOR  = 0.10      # fraction of V0 below which fit is abandoned

# Charge ramp / plateau detection (used by method1_charge_integral [legacy,
# unused] AND by method_charge_metrics [new, active])
RAMP_DV_THRESH    = 0.05       # V/sample  dV per sample to flag ramp start
PLATEAU_FRAC      = 0.985      # fraction of V_max that defines plateau start

# Leakage-current window: skip this many seconds at the start of the plateau
# to let dielectric absorption settle before reading steady-state leakage.
LEAK_SETTLE_S     = 0.5        # s

# ESR window
ESR_ONSET_SAMPLES = 5          # first N samples after ramp start (legacy)

# PSU set-point voltage (used in ESR calculation)
TARGET_VOLTAGE    = 25.0       # V   must match acquisition script

# ESR measurement window at 1 kHz only
ESR_ONSET_SAMPLES_1KHZ = 10   # samples (= 10 ms at 1 kHz)

# Charge/discharge profile overlay panels: plot every Nth cycle instead of
# all of them, so individual lines stay distinguishable and the colorbar
# progression (early -> late) is actually readable.
PROFILE_PLOT_STRIDE = 100

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME — light/white background
# ─────────────────────────────────────────────────────────────────────────────
_BG    = "#ffffff"   # figure background
_PANEL = "#f6f8fa"   # axes background
_GRID  = "#d0d7de"   # gridlines / spines
_TEXT  = "#1f2328"   # titles, suptitle, legend text
_GREY  = "#57606a"   # axis labels, ticks, secondary annotations
_BLUE  = "#0969da"
_GREEN = "#1a7f37"
_YELL  = "#9a6700"
_RED   = "#cf222e"


# ─────────────────────────────────────────────────────────────────────────────
# INPUT  ← set to a single CSV file path OR a folder path containing CSVs.
#          Folder mode processes every daq_data_*.csv inside and produces
#          per-file reports plus one combined summary across all files.
# ─────────────────────────────────────────────────────────────────────────────
# CSV_INPUT = "/path/to/daq_data_folder"   # or a single daq_data_*.csv file


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER  (handles old 7-col and new 12/13-col formats + sentinel quirk)
# ─────────────────────────────────────────────────────────────────────────────
def _find_sentinel_row(filepath: Path) -> int | None:
    """
    The old acquisition script wraps the DATA-START sentinel in quotes, which
    prevents pandas comment='#' from filtering it.  Scan the raw file and
    return the 0-based line index of that row so it can be passed to skiprows.
    Returns None if the sentinel is not found.
    """
    with filepath.open() as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip().strip('"')
            if stripped.startswith("# === DATA START"):
                return idx
    return None


def load_csv(filepath: Path) -> tuple[pd.DataFrame, dict, str]:
    """
    Load a DAQ CSV.  Returns (df, metadata_dict, format_string).

    Two file shapes are accepted:
      'minimal'  — 7 raw columns (time_s, channel0_voltage, capacitor_voltage,
                   shunt_voltage, capacitor_current, psu_mode, cycles).
                   This is the lean acquisition format.  Derived columns
                   (delta_V, dv_dt, physical_kcl_current) are computed here
                   so the rest of the analysis pipeline sees a uniform schema.
      'full'     — 12/13 columns including all derived quantities pre-computed
                   by the acquisition script.
    """
    # Parse metadata comments
    meta: dict = {}
    with filepath.open() as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("#"):
                break
            parts = s.lstrip("# ").split(",", 1)
            if len(parts) == 2:
                meta[parts[0].strip()] = parts[1].strip()

    sentinel_row = _find_sentinel_row(filepath)
    skip = [sentinel_row] if sentinel_row is not None else []

    df = pd.read_csv(filepath, comment="#", skiprows=skip)

    # Normalise cycle column name
    if "cycles" in df.columns:
        df = df.rename(columns={"cycles": "cycle_number"})

    # Detect format by the presence of pre-computed derived columns
    full_cols = {"delta_V", "dv_dt", "ideal_capacitor_current",
                 "true_capacitor_current", "physical_kcl_current",
                 "leakage_current", "true_capacitor_capacitance"}
    fmt = "full" if full_cols.issubset(df.columns) else "minimal"

    # Minimal format: rename raw columns and reconstruct derived quantities
    if fmt == "minimal":
        df = df.rename(columns={
            "shunt_voltage"     : "v_shunt",
            "capacitor_current" : "physical_kcl_current",
            "channel0_voltage"  : "voltage_V_ch0",
        })
        dt = df["time_s"].diff().median()
        df["delta_V"] = df["capacitor_voltage"].diff().fillna(0.0)
        df["dv_dt"]   = df["delta_V"] / dt

    # Infer sampling rate
    dt_median = df["time_s"].diff().median()
    df.attrs["dt"]          = dt_median
    df.attrs["sample_rate"] = round(1.0 / dt_median)
    df.attrs["fmt"]         = fmt

    print(f"  Format      : {fmt}  ({len(df.columns)} columns)")
    print(f"  Rows        : {len(df):,}")
    print(f"  Sample rate : {df.attrs['sample_rate']} Hz  (dt={dt_median*1000:.2f} ms)")
    print(f"  Cycles      : {df['cycle_number'].min()} → {df['cycle_number'].max()}")
    print(f"  Duration    : {df['time_s'].max():.1f} s")

    return df, meta, fmt


# ─────────────────────────────────────────────────────────────────────────────
# R_DISCHARGE  —  empirical from V / |I| during clean discharge decay
# ─────────────────────────────────────────────────────────────────────────────
def measure_R_discharge(df: pd.DataFrame) -> float:
    """
    Aggregate V/|I| across all discharge cycles, restricting to the clean
    RC-decay window (past relay hold-off, above noise floor).
    Uses physical_kcl_current which is the KCL-measured cap current.
    """
    all_R: list[float] = []
    for cyc in df["cycle_number"].unique():
        di = df[(df["cycle_number"] == cyc) &
                (df["psu_mode"] == "discharge")].reset_index(drop=True)
        if len(di) < 30:
            continue
        di["dv"] = di["capacitor_voltage"].diff().fillna(0.0)
        fi = di.index[di["dv"] / df.attrs["dt"] < RELAY_HOLDOFF_DVDT]
        if len(fi) == 0:
            continue
        active = di.iloc[fi[0]:].copy()
        # Restrict to clean window: V > 2 V and |I| > 0.5 mA (above noise)
        mask = (active["capacitor_voltage"] > 2.0) & \
               (active["physical_kcl_current"].abs() > 5e-4)
        vals = (active.loc[mask, "capacitor_voltage"] /
                active.loc[mask, "physical_kcl_current"].abs())
        all_R.extend(vals.tolist())

    if not all_R:
        print("  WARNING: R_discharge estimation failed; using fallback 1339 Ω")
        return 1339.0
    R = float(np.median(all_R))
    print(f"  R_discharge : {R:.1f} Ω  (median of {len(all_R):,} samples)")
    return R


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — CHARGE PHASE INTEGRAL  (LEGACY — NOT CALLED BY DEFAULT)
# ─────────────────────────────────────────────────────────────────────────────
def method1_charge_integral(df: pd.DataFrame) -> pd.DataFrame:
    """
    C_total = Q_ramp / ΔV_ramp   where Q = ∫ I_kcl dt over voltage-ramp window.

    NOT CALLED in the default pipeline below (_process_single_file). On
    minimal-format files the shunt sees ~0.4 mA while 1000 uF actually
    charges through 24 V in 6 s at ~4 mA average — the PSU delivers most of
    the initial charge directly, bypassing the shunt, so C_total comes out
    as ~75-85 uF noise (confirmed in your June 2 report). Kept here for
    'full' format files where the shunt reliably captures charge current;
    re-enable by calling this in _process_single_file and adding a panel
    back into plot_results if/when you're analysing new-format data.

    IMPORTANT: this same shunt-undersampling problem is why efficiency_pct
    (see compute_efficiency below) is flagged as low-reliability for
    minimal-format files too — it integrates current over the same ramp
    window that breaks here.

    Reliability flag:
      - 'full'    format : RELIABLE  (shunt accurately captures charging current)
      - 'minimal' format : LOW       (see above)
    """
    fmt      = df.attrs.get("fmt", "unknown")
    reliable = fmt == "full"
    dt       = df.attrs["dt"]
    records  = []

    charge_cycles = sorted(
        df.loc[df["psu_mode"] == "charge", "cycle_number"].unique()
    )

    for cyc in charge_cycles:
        ch = df[(df["cycle_number"] == cyc) &
                (df["psu_mode"] == "charge")].reset_index(drop=True)
        if len(ch) < 10:
            continue

        # Ramp window: first sample with dV > RAMP_DV_THRESH to plateau
        dv_series   = ch["capacitor_voltage"].diff().fillna(0.0)
        rising_idx  = ch.index[dv_series > RAMP_DV_THRESH]
        if len(rising_idx) == 0:
            continue
        rs = rising_idx[0]

        V_max     = ch["capacitor_voltage"].max()
        plateau   = ch.index[ch["capacitor_voltage"] >= PLATEAU_FRAC * V_max]
        re        = plateau[0] if len(plateau) > 0 else len(ch) - 1
        if re <= rs:
            continue

        ramp    = ch.iloc[rs : re + 1]
        delta_V = ramp["capacitor_voltage"].iloc[-1] - ramp["capacitor_voltage"].iloc[0]
        if delta_V <= 0:
            continue

        Q_ramp  = trapezoid(ramp["physical_kcl_current"], ramp["time_s"])
        C_uF    = Q_ramp / delta_V * 1e6
        dt_ramp = ramp["time_s"].iloc[-1] - ramp["time_s"].iloc[0]

        records.append({
            "cycle"      : cyc,
            "Q_mC"       : Q_ramp * 1e3,
            "delta_V_V"  : delta_V,
            "C_total_uF" : C_uF,
            "ramp_dur_s" : dt_ramp,
            "ramp_rows"  : len(ramp),
            "reliable"   : reliable,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — SINGLE-EXPONENTIAL DISCHARGE TAU  (PRIMARY)
# ─────────────────────────────────────────────────────────────────────────────
def _single_exp(t, V0, tau):
    return V0 * np.exp(-t / tau)


def method2_single_tau(df: pd.DataFrame, R_discharge: float) -> pd.DataFrame:
    """
    Fit V(t) = V0 · exp(−t/τ) to the TAIL of each discharge curve.

    Why tail only [5 %–50 %] of V0:
      The first ~20–30 % of discharge is contaminated by the relay hold-off
      plateau and the fast inductive/dielectric transient immediately after
      the relay opens.  Fitting through those distorts τ upward and inflates
      RMSE to ~300 mV.  Restricting the window to [5 %–50 %] of V0 sits
      entirely inside the clean RC decay, giving RMSE ≈ 70–120 mV and a
      monotone τ that tracks C_test faithfully.

    C_test = τ / R_discharge  (single component — no PSU filter cap present,
    confirmed empirically: blocky charge/discharge with no cap = no filter cap).
    """
    dt      = df.attrs["dt"]
    records = []

    dis_cycles = sorted(
        df.loc[df["psu_mode"] == "discharge", "cycle_number"].unique()
    )

    for cyc in dis_cycles:
        di = df[(df["cycle_number"] == cyc) &
                (df["psu_mode"] == "discharge")].reset_index(drop=True)
        if len(di) < 50:
            continue

        # Locate end of relay hold-off (first sample where dV/dt < threshold)
        di["_dvdt"] = di["capacitor_voltage"].diff().fillna(0.0) / dt
        falling     = di.index[di["_dvdt"] < RELAY_HOLDOFF_DVDT]
        if len(falling) == 0:
            continue
        V0_approx = di["capacitor_voltage"].iloc[falling[0]]

        # Tail window: 5 % to 50 % of V0_approx — clean single-RC decay
        tail = di[(di["capacitor_voltage"] <= 0.50 * V0_approx) &
                  (di["capacitor_voltage"] >  0.05 * V0_approx)].reset_index(drop=True)
        if len(tail) < 20:
            continue

        t_fit = tail["time_s"].values - tail["time_s"].iloc[0]
        V_fit = tail["capacitor_voltage"].values

        try:
            popt, _ = curve_fit(
                _single_exp, t_fit, V_fit,
                p0=[V_fit[0], 1.1],
                bounds=([0, 0.1], [60, 10]),
                maxfev=20_000,
            )
            V0_fit, tau_fit = popt
            residuals = V_fit - _single_exp(t_fit, *popt)
            rmse      = float(np.sqrt(np.mean(residuals ** 2)))
            C_uF      = tau_fit / R_discharge * 1e6

            records.append({
                "cycle"       : cyc,
                "tau_s"       : tau_fit,
                "C_test_uF"   : C_uF,
                "rmse_mV"     : rmse * 1e3,
                "fit_rows"    : len(tail),
                "R_discharge" : R_discharge,
            })
        except Exception:
            continue

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# ADDITIONAL DEGRADATION METRICS  (discharge-side, no fitting)
# ─────────────────────────────────────────────────────────────────────────────
def method_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two tau-independent degradation metrics extracted per discharge cycle.

    1. E_discharge_mJ  — energy delivered during discharge = ∫ V · |I| dt
       Decreasing energy → capacitor storing/releasing less charge → degradation.
       Does not require curve fitting; robust to waveform shape changes.
       Reliable at any format: during discharge there's no PSU bypass path,
       so the shunt sees the full current.

    2. V_at_t2s  — capacitor voltage exactly 2 s after relay hold-off ends.
       Directly reflects τ: V(2) = V0 · exp(−2/τ).  A shorter τ (degraded cap)
       gives a lower V@2 s.  Computed by nearest-sample lookup — no fitting.
       t = 2 s chosen because it sits in the centre of the [5 %–50 %] RC window
       (at τ ≈ 1.1 s, V(2) ≈ 16 % of V0 ≈ 3.9 V — well above the noise floor).
    """
    dt      = df.attrs["dt"]
    records = []

    dis_cycles = sorted(
        df.loc[df["psu_mode"] == "discharge", "cycle_number"].unique()
    )

    for cyc in dis_cycles:
        di = df[(df["cycle_number"] == cyc) &
                (df["psu_mode"] == "discharge")].reset_index(drop=True)
        if len(di) < 50:
            continue

        # Locate relay hold-off end
        di["_dvdt"] = di["capacitor_voltage"].diff().fillna(0.0) / dt
        falling     = di.index[di["_dvdt"] < RELAY_HOLDOFF_DVDT]
        if len(falling) == 0:
            continue
        fall_start = falling[0]

        # ── Discharge energy ──────────────────────────────────────────────────
        active = di.iloc[fall_start:].reset_index(drop=True)
        E_J    = trapezoid(
            active["capacitor_voltage"] * active["physical_kcl_current"].abs(),
            active["time_s"],
        )

        # ── V at t = 2 s after relay opens ───────────────────────────────────
        active["t_rel"] = active["time_s"] - active["time_s"].iloc[0]
        idx_2s = (active["t_rel"] - 2.0).abs().idxmin()
        V_at_t2s = float(active.loc[idx_2s, "capacitor_voltage"])

        records.append({
            "cycle"        : cyc,
            "E_discharge_mJ": E_J * 1e3,
            "V_at_t2s_V"   : V_at_t2s,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGE-PHASE DIAGNOSTICS (NEW)  — leakage current, E_charge, time-to-plateau
# ─────────────────────────────────────────────────────────────────────────────
def method_charge_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three charge-phase diagnostics per cycle. No curve fitting; valid at
    100 Hz. See top-of-file docstring for the full physical reasoning.

    1. I_leak_mA — median current over the back portion of the charge
       plateau (skips the first LEAK_SETTLE_S seconds for dielectric
       absorption to die down). Steady-state DC reading — NOT affected by
       the shunt-undersampling problem that breaks Method 1, because that
       problem is specific to the fast charging transient, not a slow
       steady-state hold.

    2. E_charge_mJ — energy integral over the FULL charge phase (ramp +
       plateau), same trapezoid approach as E_discharge. CAUTION: this DOES
       include the ramp window, so for minimal-format files it inherits
       Method 1's shunt-undersampling problem. Used only to build
       efficiency_pct in compute_efficiency() below — flag accordingly
       there.

    3. time_to_plateau_s — duration from rise-onset to plateau threshold.
       Pure timing measurement on the voltage channel only — does not
       touch the current/shunt measurement chain at all, so it has none of
       Method 1's reliability problem. Independent cross-check on the
       discharge-derived C_test trend.
    """
    dt      = df.attrs["dt"]
    records = []

    charge_cycles = sorted(
        df.loc[df["psu_mode"] == "charge", "cycle_number"].unique()
    )

    for cyc in charge_cycles:
        ch = df[(df["cycle_number"] == cyc) &
                (df["psu_mode"] == "charge")].reset_index(drop=True)
        if len(ch) < 10:
            continue

        dv_series  = ch["capacitor_voltage"].diff().fillna(0.0)
        rising_idx = ch.index[dv_series > RAMP_DV_THRESH]
        if len(rising_idx) == 0:
            continue
        rs = rising_idx[0]

        V_max       = ch["capacitor_voltage"].max()
        plateau_idx = ch.index[ch["capacitor_voltage"] >= PLATEAU_FRAC * V_max]
        if len(plateau_idx) == 0:
            continue
        re = plateau_idx[0]
        if re <= rs:
            continue

        # 3. Time to plateau — voltage-channel timing only
        time_to_plateau = float(ch["time_s"].iloc[re] - ch["time_s"].iloc[rs])

        # 2. E_charge — full charge-phase energy integral (rs to end)
        full_charge = ch.iloc[rs:].reset_index(drop=True)
        E_charge_J  = trapezoid(
            full_charge["capacitor_voltage"] * full_charge["physical_kcl_current"].abs(),
            full_charge["time_s"],
        )

        # 1. Leakage current — back portion of plateau, post-settle
        plateau = ch.iloc[re:].reset_index(drop=True)
        settle_skip = max(1, int(round(LEAK_SETTLE_S / dt)))
        if len(plateau) <= settle_skip + 5:
            continue
        leak_region = plateau.iloc[settle_skip:]
        I_leak_mA = float(leak_region["physical_kcl_current"].abs().median()) * 1e3

        records.append({
            "cycle"             : cyc,
            "I_leak_mA"         : I_leak_mA,
            "E_charge_mJ"       : E_charge_J * 1e3,
            "time_to_plateau_s" : time_to_plateau,
        })

    return pd.DataFrame(records)


def compute_efficiency(m_charge: pd.DataFrame, m_extra: pd.DataFrame) -> pd.DataFrame:
    """
    Round-trip energy efficiency = E_discharge / E_charge, matched by cycle
    number. Falling efficiency = rising dissipative losses (ESR + leakage)
    — the complement to C_test (which tracks storage capacity, not loss).

    RELIABILITY CAUTION: E_charge includes the charge ramp window, which on
    minimal-format files suffers the same shunt-undersampling problem that
    made Method 1 unreliable (PSU bypasses the shunt during the fast
    charging transient). The caller (print_report / plot_results) flags
    this explicitly when fmt == 'minimal'. On 'full' format files this
    metric is fully reliable.
    """
    if m_charge.empty or m_extra.empty:
        return pd.DataFrame()
    merged = pd.merge(
        m_charge[["cycle", "E_charge_mJ"]],
        m_extra[["cycle", "E_discharge_mJ"]],
        on="cycle", how="inner",
    )
    merged = merged[merged["E_charge_mJ"] > 0].reset_index(drop=True)
    if merged.empty:
        return merged
    merged["efficiency_pct"] = merged["E_discharge_mJ"] / merged["E_charge_mJ"] * 100.0
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# ESR PROXY  (1 kHz files only)
# ─────────────────────────────────────────────────────────────────────────────
def method_esr_1khz(df: pd.DataFrame, m2: pd.DataFrame,
                    R_discharge: float) -> pd.DataFrame:
    """
    Compute an ESR proxy per discharge cycle.  Only meaningful at ≥ 900 Hz.

    Physics:
      A capacitor with internal ESR discharges through a two-regime curve:
        • Onset  (first ~10 ms): fast decay dominated by ESR voltage drop.
        • Tail   (long RC):      slow exponential set by C × R_discharge.

      τ_initial = −1 / slope(ln V) over the first 10 ms after relay release.
      τ_tail    = from method2_single_tau [5 %–50 %] window.

      ESR_proxy ≈ R_discharge × (τ_tail − τ_initial) / (τ_initial × τ_tail)
                = extra resistance responsible for the faster initial decay.

    For a pure ideal RC:  τ_initial ≈ τ_tail → ESR_proxy ≈ 0.
    As ESR rises:         τ_initial < τ_tail → ESR_proxy increases.

    At 100 Hz the 10-sample window spans 100 ms which mixes the onset
    and tail regimes, making tau_initial unreliable.  This function
    returns an empty DataFrame for sub-900 Hz files.
    """
    sr = df.attrs.get("sample_rate", 100)
    if sr < 900:
        return pd.DataFrame()

    dt       = df.attrs["dt"]
    records  = []

    # Build a lookup of tau_tail keyed by cycle number from m2
    tau_tail_map = dict(zip(m2["cycle"].values, m2["tau_s"].values))

    dis_cycles = sorted(
        df.loc[df["psu_mode"] == "discharge", "cycle_number"].unique()
    )

    for cyc in dis_cycles:
        if cyc not in tau_tail_map:
            continue
        tau_tail = tau_tail_map[cyc]

        di = df[(df["cycle_number"] == cyc) &
                (df["psu_mode"] == "discharge")].reset_index(drop=True)
        if len(di) < 50:
            continue

        # Locate relay hold-off end
        di["_dvdt"] = di["capacitor_voltage"].diff().fillna(0.0) / dt
        falling     = di.index[di["_dvdt"] < RELAY_HOLDOFF_DVDT]
        if len(falling) == 0:
            continue
        start = falling[0]

        # 10-sample onset window immediately after hold-off ends
        onset = di.iloc[start : start + ESR_ONSET_SAMPLES_1KHZ].reset_index(drop=True)
        if len(onset) < 5:
            continue

        V_on = onset["capacitor_voltage"].values
        if V_on[0] < 1.0:
            continue

        t_on = np.arange(len(onset)) * dt
        log_V = np.log(np.abs(V_on) + 1e-9)

        try:
            coef  = np.polyfit(t_on, log_V, 1)
            slope = coef[0]
            if slope >= -0.01:          # voltage not falling — skip
                continue
            tau_initial = -1.0 / slope
        except Exception:
            continue

        # ESR proxy: excess resistance in the onset regime
        if tau_initial > 0 and tau_tail > 0:
            ESR_proxy = R_discharge * (tau_tail - tau_initial) / (tau_initial * tau_tail)
        else:
            ESR_proxy = np.nan

        records.append({
            "cycle"        : cyc,
            "tau_initial_s": tau_initial,
            "tau_tail_s"   : tau_tail,
            "ESR_proxy_ohm": ESR_proxy,
        })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def _trend_line(x, y):
    """Return slope, intercept, R², p-value from linregress."""
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    sl, ic, r, p, _ = linregress(x, y)
    return float(sl), float(ic), float(r ** 2), float(p)


def print_report(m2: pd.DataFrame, m_extra: pd.DataFrame,
                 m_esr: pd.DataFrame, m_charge: pd.DataFrame,
                 m_eff: pd.DataFrame, R_discharge: float, fmt: str,
                 sample_rate: int = 100) -> None:
    """
    NOTE: Method 1 section and the RUL/EOL-cycle extrapolation block that
    used to live in the Method 2 section have both been removed. The RUL
    extrapolation projected ~7-20x beyond the observed cycle range from a
    single linear fit, which doesn't represent the actual curve shape
    (steep initial drop, then a long flatter tail) — see chat for detail.
    """
    SEP  = "─" * 65
    WIDE = "═" * 65
    print(f"\n{WIDE}")
    print("  CAPACITOR FATIGUE POST-ANALYSIS REPORT")
    print(WIDE)
    print(f"  Format: {fmt}    Sample rate: {sample_rate} Hz")

    # ── Method 2 ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  METHOD 2 — Single-Tau Discharge Fit  ← PRIMARY FATIGUE METRIC")
    print(SEP)
    if not m2.empty:
        print(f"  R_discharge    : {R_discharge:.1f} Ω  (empirical from V/I)")
        print(f"  Fit window     : [5 %–50 %] of V0  (clean RC tail, no transient)")
        print(f"  Fit RMSE range : {m2['rmse_mV'].min():.1f} – {m2['rmse_mV'].max():.1f} mV")
        print()
        print(f"  {'Cycle':>5}  {'τ (s)':>8}  {'C_test (μF)':>12}  {'RMSE (mV)':>10}")
        print(f"  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*10}")
        # Print every ~10 % of cycles to keep output manageable
        step = max(1, len(m2) // 10)
        for _, row in m2.iloc[::step].iterrows():
            print(f"  {int(row['cycle']):>5}  {row['tau_s']:>8.4f}  "
                  f"{row['C_test_uF']:>12.2f}  {row['rmse_mV']:>10.2f}")

        sl_C, ic_C, r2_C, p_C = _trend_line(m2["cycle"].values,
                                              m2["C_test_uF"].values)
        note = "✓ Significant" if p_C < 0.05 else "✗ Not yet significant"
        print()
        print(f"  C_test first   : {m2['C_test_uF'].iloc[0]:.2f} μF")
        print(f"  C_test last    : {m2['C_test_uF'].iloc[-1]:.2f} μF")
        print(f"  Total drop     : {m2['C_test_uF'].iloc[0]-m2['C_test_uF'].iloc[-1]:.2f} μF"
              f"  ({(m2['C_test_uF'].iloc[0]-m2['C_test_uF'].iloc[-1])/m2['C_test_uF'].iloc[0]*100:.2f} %)")
        print(f"  Linear slope   : {sl_C:+.4f} μF/cycle   R²={r2_C:.4f}   p={p_C:.2e}")
        print(f"  Trend          : {note}  (α = 0.05)")

    # ── Additional metrics ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ADDITIONAL METRICS — Discharge Energy & V@2 s")
    print(SEP)
    if not m_extra.empty:
        sl_E, ic_E, r2_E, p_E = _trend_line(m_extra["cycle"].values,
                                              m_extra["E_discharge_mJ"].values)
        sl_V, ic_V, r2_V, p_V = _trend_line(m_extra["cycle"].values,
                                              m_extra["V_at_t2s_V"].values)
        print(f"  Discharge energy  first : {m_extra['E_discharge_mJ'].iloc[0]:.3f} mJ")
        print(f"  Discharge energy  last  : {m_extra['E_discharge_mJ'].iloc[-1]:.3f} mJ")
        print(f"  Energy slope            : {sl_E:+.5f} mJ/cycle   R²={r2_E:.4f}   p={p_E:.2e}")
        print()
        print(f"  V @ t=2 s  first        : {m_extra['V_at_t2s_V'].iloc[0]:.3f} V")
        print(f"  V @ t=2 s  last         : {m_extra['V_at_t2s_V'].iloc[-1]:.3f} V")
        print(f"  V@2s slope              : {sl_V:+.5f} V/cycle   R²={r2_V:.4f}   p={p_V:.2e}")

    # ── Leakage current (NEW) ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  LEAKAGE CURRENT  — charge-plateau steady-state (NEW)")
    print(SEP)
    if not m_charge.empty:
        sl_I, ic_I, r2_I, p_I = _trend_line(m_charge["cycle"].values,
                                              m_charge["I_leak_mA"].values)
        print(f"  I_leak first   : {m_charge['I_leak_mA'].iloc[0]:.4f} mA")
        print(f"  I_leak last    : {m_charge['I_leak_mA'].iloc[-1]:.4f} mA")
        print(f"  Linear slope   : {sl_I:+.6f} mA/cycle   R²={r2_I:.4f}   p={p_I:.2e}")
        note_I = ("✓ Rising leakage" if (not np.isnan(sl_I) and sl_I > 0 and p_I < 0.05)
                  else "— No significant rise yet")
        print(f"  Verdict        : {note_I}")
    else:
        print("  (not enough clean plateau samples to compute)")

    # ── Round-trip efficiency (NEW, flagged for minimal format) ──────────────
    print(f"\n{SEP}")
    print("  ROUND-TRIP EFFICIENCY  — E_discharge / E_charge (NEW)")
    print(SEP)
    if fmt == "minimal":
        print("  ⚠  LOW RELIABILITY for minimal-format files.")
        print("     E_charge integrates the same charge-ramp current that the shunt")
        print("     undersamples (see Method 1) — efficiency numbers below inherit")
        print("     that error. Treat as supplementary; reliable on 'full' format.")
    if not m_eff.empty:
        sl_e, ic_e, r2_e, p_e = _trend_line(m_eff["cycle"].values,
                                              m_eff["efficiency_pct"].values)
        print(f"  Efficiency first : {m_eff['efficiency_pct'].iloc[0]:.2f} %")
        print(f"  Efficiency last  : {m_eff['efficiency_pct'].iloc[-1]:.2f} %")
        print(f"  Linear slope     : {sl_e:+.5f} %/cycle   R²={r2_e:.4f}   p={p_e:.2e}")
        note_e = ("✓ Falling efficiency (rising losses)"
                  if (not np.isnan(sl_e) and sl_e < 0 and p_e < 0.05)
                  else "— No significant trend yet")
        print(f"  Verdict          : {note_e}")
    else:
        print("  (could not match charge/discharge cycles)")

    # ── Time-to-plateau (NEW, charge-side cross-check) ────────────────────────
    print(f"\n{SEP}")
    print("  TIME-TO-PLATEAU  — charge-side cross-check on C_test (NEW)")
    print(SEP)
    if not m_charge.empty:
        sl_t, ic_t, r2_t, p_t = _trend_line(m_charge["cycle"].values,
                                              m_charge["time_to_plateau_s"].values)
        print(f"  t_plateau first  : {m_charge['time_to_plateau_s'].iloc[0]:.4f} s")
        print(f"  t_plateau last   : {m_charge['time_to_plateau_s'].iloc[-1]:.4f} s")
        print(f"  Linear slope     : {sl_t:+.6f} s/cycle   R²={r2_t:.4f}   p={p_t:.2e}")
        agree = ("consistent with C_test decline" if (not np.isnan(sl_t) and sl_t < 0)
                 else "does NOT corroborate C_test decline — check discharge-side instrumentation")
        print(f"  Cross-check      : {agree}")
    else:
        print("  (not enough clean ramp samples to compute)")

    # ── ESR section (1 kHz only) ──────────────────────────────────────────────
    if sample_rate >= 900 and not m_esr.empty:
        print(f"\n{SEP}")
        print("  ESR PROXY  — τ_initial vs τ_tail ratio  (1 kHz only)")
        print(SEP)
        sl_esr, ic_esr, r2_esr, p_esr = _trend_line(
            m_esr["cycle"].values, m_esr["ESR_proxy_ohm"].values)
        print(f"  ESR_proxy first  : {m_esr['ESR_proxy_ohm'].iloc[0]:.2f} Ω")
        print(f"  ESR_proxy last   : {m_esr['ESR_proxy_ohm'].iloc[-1]:.2f} Ω")
        print(f"  Linear slope     : {sl_esr:+.4f} Ω/cycle   R²={r2_esr:.4f}   p={p_esr:.2e}")
        print(f"  Onset window     : {ESR_ONSET_SAMPLES_1KHZ} ms  (τ_initial from log-slope)")
        note_esr = "✓ Rising ESR trend" if (not np.isnan(sl_esr) and sl_esr > 0 and p_esr < 0.05) else "— No significant ESR rise yet"
        print(f"  Verdict          : {note_esr}")
    elif sample_rate < 900:
        print(f"\n{SEP}")
        print(f"  ESR PROXY  — not computed (requires >= 1 kHz, file is {sample_rate} Hz)\n")
        print(SEP)

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n{WIDE}")
    print("  RECOMMENDATION")
    print(WIDE)
    print(f"  • Method 2 (τ → C_test) is the primary fatigue metric.")
    print(f"  • Discharge energy and V@2 s cross-validate without curve fitting.")
    print(f"  • Leakage current and time-to-plateau are independent cross-checks —")
    print(f"    leakage probes a different failure mechanism (oxide vs electrolyte);")
    print(f"    time-to-plateau corroborates C_test using a separate measurement chain.")
    print(f"  • Round-trip efficiency is supplementary only on minimal-format files.")
    print(f"  • ESR proxy (1 kHz files only) tracks internal resistance degradation.")
    print(f"  • Flag DUT for inspection if C_test drops >5 % from baseline.")
    print(f"  • All metrics should trend in the same direction; divergence signals a")
    print(f"    measurement artefact rather than true degradation.")
    print(f"{WIDE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def _ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=_GREY, fontsize=8)
    ax.set_ylabel(ylabel, color=_GREY, fontsize=8)
    ax.tick_params(colors=_GREY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.7)


def plot_results(df: pd.DataFrame, m2: pd.DataFrame,
                 m_extra: pd.DataFrame, m_esr: pd.DataFrame,
                 m_charge: pd.DataFrame, m_eff: pd.DataFrame,
                 R_discharge: float, output_path: Path) -> None:
    fig = plt.figure(figsize=(20, 15), facecolor=_BG)
    fig.suptitle(
        "Capacitor Fatigue Post-Analysis  —  Single-Tau Discharge Method",
        fontsize=14, fontweight="bold", color=_TEXT, y=0.98,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

    dt  = df.attrs["dt"]
    fmt = df.attrs.get("fmt", "unknown")

    # ── [0,0]  Charge voltage profiles, every PROFILE_PLOT_STRIDE-th cycle ────
    ax = fig.add_subplot(gs[0, 0])
    cycs_ch = sorted(df.loc[df["psu_mode"] == "charge", "cycle_number"].unique())
    if cycs_ch:
        idxs = list(range(0, len(cycs_ch), PROFILE_PLOT_STRIDE))
        if idxs[-1] != len(cycs_ch) - 1:
            idxs.append(len(cycs_ch) - 1)   # always include the latest cycle
        cycs_ch_sub = [cycs_ch[i] for i in idxs]
        norm_ch = plt.Normalize(min(cycs_ch), max(cycs_ch))
        for cyc in cycs_ch_sub:
            ch = df[(df["cycle_number"] == cyc) & (df["psu_mode"] == "charge")]
            t0 = ch["time_s"].iloc[0]
            ax.plot(ch["time_s"] - t0, ch["capacitor_voltage"],
                    color=plt.cm.plasma(norm_ch(cyc)), lw=1.4, alpha=0.9)
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_ch)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("Cycle #", color=_GREY, fontsize=7)
        cb.ax.yaxis.set_tick_params(color=_GREY, labelsize=6)
    _ax_style(ax, f"Charge Voltage Profiles (every {PROFILE_PLOT_STRIDE} cycles)",
              "Time offset (s)", "V_cap (V)")

    # ── [0,1]  Discharge curves + single-exp tail fits, same stride ──────────
    ax = fig.add_subplot(gs[0, 1])
    cycs_di = sorted(df.loc[df["psu_mode"] == "discharge", "cycle_number"].unique())
    if cycs_di:
        idxs_d = list(range(0, len(cycs_di), PROFILE_PLOT_STRIDE))
        if idxs_d[-1] != len(cycs_di) - 1:
            idxs_d.append(len(cycs_di) - 1)
        cycs_di_sub = [cycs_di[i] for i in idxs_d]
        norm_di = plt.Normalize(min(cycs_di), max(cycs_di))
        for cyc in cycs_di_sub:
            di = df[(df["cycle_number"] == cyc) & (df["psu_mode"] == "discharge")]
            t0 = di["time_s"].iloc[0]
            ax.plot(di["time_s"] - t0, di["capacitor_voltage"],
                    color=plt.cm.cool(norm_di(cyc)), lw=1.1, alpha=0.8)
    # Overlay single-exp fit for first and last cycle
    for row_idx, color, lbl in [(0, _GREEN, "First fit"), (-1, _RED, "Last fit")]:
        if not m2.empty and abs(row_idx) < len(m2):
            r    = m2.iloc[row_idx]
            cyc_r = int(r["cycle"])
            di_r  = df[(df["cycle_number"] == cyc_r) &
                        (df["psu_mode"] == "discharge")].reset_index(drop=True)
            di_r["_dv"] = di_r["capacitor_voltage"].diff().fillna(0) / dt
            fi = di_r.index[di_r["_dv"] < RELAY_HOLDOFF_DVDT]
            if len(fi):
                V0_r = di_r["capacitor_voltage"].iloc[fi[0]]
                t_d  = np.linspace(0, r["tau_s"] * 4, 400)
                ax.plot(t_d, _single_exp(t_d, V0_r * 0.5, r["tau_s"]),
                        "--", color=color, lw=1.8,
                        label=f"{lbl}  tau={r['tau_s']:.3f}s")
    ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
    _ax_style(ax, f"Discharge Curves + Single-Exp Fits (every {PROFILE_PLOT_STRIDE} cycles)",
              "Time offset (s)", "V_cap (V)")

    # ── [0,2]  NEW — Leakage current per cycle ────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if not m_charge.empty:
        cyc_l = m_charge["cycle"].values
        I_v   = m_charge["I_leak_mA"].values
        ax.scatter(cyc_l, I_v, color=_GREEN, s=4, zorder=5, alpha=0.7, label="I_leak")
        sl_l, ic_l, r2_l, _ = _trend_line(cyc_l, I_v)
        if not np.isnan(sl_l):
            x_l = np.array([cyc_l[0], cyc_l[-1]])
            ax.plot(x_l, sl_l * x_l + ic_l, "--", color=_RED, lw=1.8,
                    label=f"{sl_l:+.5f} mA/cycle  R2={r2_l:.3f}")
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
    else:
        ax.text(0.5, 0.5, "Not enough clean\nplateau samples", transform=ax.transAxes,
                fontsize=8, color=_GREY, ha="center", va="center", style="italic")
    _ax_style(ax, "Leakage Current  (charge-plateau steady-state)",
              "Cycle Number", "I_leak (mA)")

    # ── [1,0]  C_test per cycle ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if not m2.empty:
        cyc_v = m2["cycle"].values
        Ct_v  = m2["C_test_uF"].values
        ax.scatter(cyc_v, Ct_v, color=_BLUE, s=4, zorder=5, alpha=0.7, label="C_test")
        sl, ic, r2, _ = _trend_line(cyc_v, Ct_v)
        if not np.isnan(sl):
            x_lr = np.array([cyc_v[0], cyc_v[-1]])
            ax.plot(x_lr, sl * x_lr + ic, "--", color=_RED, lw=1.8,
                    label=f"{sl:+.4f} uF/cycle  R2={r2:.3f}")
        ax.axhline(C_NOMINAL_uF, color=_GREY, lw=0.8, linestyle=":", label="Nominal 1000 uF")
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
    _ax_style(ax, "Method 2 — C_test (tau/R) per Cycle", "Cycle Number", "C_test (uF)")

    # ── [1,1]  Discharge energy per cycle ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    if not m_extra.empty:
        cyc_e = m_extra["cycle"].values
        E_v   = m_extra["E_discharge_mJ"].values
        ax.scatter(cyc_e, E_v, color=_YELL, s=4, zorder=5, alpha=0.7, label="E_discharge")
        sl_e, ic_e, r2_e, _ = _trend_line(cyc_e, E_v)
        if not np.isnan(sl_e):
            x_e = np.array([cyc_e[0], cyc_e[-1]])
            ax.plot(x_e, sl_e * x_e + ic_e, "--", color=_RED, lw=1.8,
                    label=f"{sl_e:+.5f} mJ/cycle  R2={r2_e:.3f}")
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
    _ax_style(ax, "Discharge Energy per Cycle  (integral V*|I|*dt)",
              "Cycle Number", "Energy (mJ)")

    # ── [1,2]  V@t=2s per cycle (tau-free proxy) ──────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if not m_extra.empty:
        V2_v = m_extra["V_at_t2s_V"].values
        ax.scatter(cyc_e, V2_v, color=_GREEN, s=4, zorder=5, alpha=0.7, label="V @ t=2s")
        sl_v, ic_v, r2_v, _ = _trend_line(cyc_e, V2_v)
        if not np.isnan(sl_v):
            ax.plot(x_e, sl_v * x_e + ic_v, "--", color=_RED, lw=1.8,
                    label=f"{sl_v:+.5f} V/cycle  R2={r2_v:.3f}")
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
    ax.text(0.5, 0.06, "Tau-free: V(2s)=V0*exp(-2/tau) — lower V = shorter tau = degraded",
            transform=ax.transAxes, fontsize=6.5, color=_GREEN, ha="center", style="italic")
    _ax_style(ax, "V @ t=2 s After Discharge Onset (Tau-Free Proxy)",
              "Cycle Number", "Voltage (V)")

    # ── [2,0]  NEW — Round-trip efficiency (flagged on minimal format) ──────
    ax = fig.add_subplot(gs[2, 0])
    if not m_eff.empty:
        cyc_e2 = m_eff["cycle"].values
        eff_v  = m_eff["efficiency_pct"].values
        ax.scatter(cyc_e2, eff_v, color=_YELL, s=4, zorder=5, alpha=0.7, label="Efficiency")
        sl_ef, ic_ef, r2_ef, _ = _trend_line(cyc_e2, eff_v)
        if not np.isnan(sl_ef):
            x_ef = np.array([cyc_e2[0], cyc_e2[-1]])
            ax.plot(x_ef, sl_ef * x_ef + ic_ef, "--", color=_RED, lw=1.8,
                    label=f"{sl_ef:+.5f} %/cycle  R2={r2_ef:.3f}")
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
        if fmt == "minimal":
            ax.text(0.5, 0.92, "WARNING: LOW RELIABILITY (E_charge inherits Method-1 shunt issue)",
                    transform=ax.transAxes, fontsize=6.5, color=_YELL,
                    ha="center", style="italic")
    else:
        ax.text(0.5, 0.5, "Could not match\ncharge/discharge cycles", transform=ax.transAxes,
                fontsize=8, color=_GREY, ha="center", va="center", style="italic")
    _ax_style(ax, "Round-Trip Efficiency  (E_discharge / E_charge)",
              "Cycle Number", "Efficiency (%)")

    # ── [2,1]  NEW — Time-to-plateau (charge-side cross-check) ───────────────
    ax = fig.add_subplot(gs[2, 1])
    if not m_charge.empty:
        cyc_t = m_charge["cycle"].values
        t_v   = m_charge["time_to_plateau_s"].values
        ax.scatter(cyc_t, t_v, color=_BLUE, s=4, zorder=5, alpha=0.7, label="t_plateau")
        sl_t2, ic_t2, r2_t2, _ = _trend_line(cyc_t, t_v)
        if not np.isnan(sl_t2):
            x_t2 = np.array([cyc_t[0], cyc_t[-1]])
            ax.plot(x_t2, sl_t2 * x_t2 + ic_t2, "--", color=_RED, lw=1.8,
                    label=f"{sl_t2:+.6f} s/cycle  R2={r2_t2:.3f}")
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
        ax.text(0.5, 0.06, "Voltage-channel only — should fall if C_test decline is real",
                transform=ax.transAxes, fontsize=6.5, color=_GREY, ha="center", style="italic")
    else:
        ax.text(0.5, 0.5, "Not enough clean\nramp samples", transform=ax.transAxes,
                fontsize=8, color=_GREY, ha="center", va="center", style="italic")
    _ax_style(ax, "Time-to-Plateau  (charge-side cross-check)",
              "Cycle Number", "Time (s)")

    # ── [2,2]  ESR proxy (1 kHz) or Fit RMSE (100 Hz) ───────────────────────
    ax = fig.add_subplot(gs[2, 2])
    sr = df.attrs.get("sample_rate", 100)
    if sr >= 900 and not m_esr.empty:
        # 1 kHz: show ESR proxy trend
        cyc_esr = m_esr["cycle"].values
        esr_v   = m_esr["ESR_proxy_ohm"].values
        ax.scatter(cyc_esr, esr_v, color=_RED, s=4, alpha=0.7, label="ESR proxy")
        sl_er, ic_er, r2_er, _ = _trend_line(cyc_esr, esr_v)
        if not np.isnan(sl_er):
            ax.plot(np.array([cyc_esr[0], cyc_esr[-1]]),
                    sl_er * np.array([cyc_esr[0], cyc_esr[-1]]) + ic_er,
                    "--", color=_YELL, lw=1.8,
                    label=f"{sl_er:+.4f} Ω/cycle  R²={r2_er:.3f}")
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
        _ax_style(ax, "ESR Proxy — τ_initial vs τ_tail  (1 kHz)",
                  "Cycle Number", "ESR proxy (Ω)")
    else:
        # 100 Hz: show fit RMSE
        if not m2.empty:
            ax.scatter(m2["cycle"], m2["rmse_mV"],
                       color=_BLUE, s=4, alpha=0.7, label="Single-exp RMSE")
            ax.axhline(m2["rmse_mV"].mean(), color=_YELL, lw=1.2, linestyle="--",
                       label=f"Mean {m2['rmse_mV'].mean():.1f} mV")
            ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
        ax.text(0.5, 0.5, "ESR requires ≥ 1 kHz\n(upgrade sampling rate)",
                transform=ax.transAxes, fontsize=8, color=_GREY,
                ha="center", va="center", style="italic")
        _ax_style(ax, "Method 2 — Fit RMSE  (ESR N/A at 100 Hz)",
                  "Cycle Number", "RMSE (mV)")

    # ── [3,:]  tau_capacitance — single C_test across all cycles ──────────────
    ax = fig.add_subplot(gs[3, :])
    if not m2.empty:
        cyc_v = m2["cycle"].values
        Ct_v  = m2["C_test_uF"].values
        ax.scatter(cyc_v, Ct_v, color=_BLUE, s=5, alpha=0.6, zorder=4, label="C_test  (tau/R)  — DUT")
        sl, ic, r2, _ = _trend_line(cyc_v, Ct_v)
        if not np.isnan(sl):
            x_lr = np.linspace(cyc_v[0], cyc_v[-1], 300)
            ax.plot(x_lr, sl * x_lr + ic, "-", color=_BLUE, lw=2.0, alpha=0.85,
                    label=f"Trend  {sl:+.4f} uF/cycle  R2={r2:.3f}")
        ax.axhline(C_NOMINAL_uF, color=_GREY, lw=0.9, linestyle=":", label="Nominal 1000 uF")
        C_eol_b = (float(Ct_v[0]) if len(Ct_v) else C_NOMINAL_uF) * 0.80
        ax.axhline(C_eol_b, color=_RED, lw=0.9, linestyle="--",
                   label=f"EOL threshold 80%  ({C_eol_b:.0f} uF)")
        ax.fill_between(cyc_v, 0, Ct_v, alpha=0.10, color=_BLUE)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID,
                  loc="upper right", ncol=2)
    _ax_style(ax, "tau_capacitance — C_test from Single-Tau Discharge Fit",
              "Cycle Number", "Capacitance (uF)")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    print(f"  Plot saved  -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def _process_single_file(filepath: Path, global_cycle_offset: int = 0
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                     float, str, int]:
    """
    Run the full analysis pipeline on one CSV file.
    Returns (m2, m_extra, m_esr, R_discharge, fmt, next_global_offset).
    m2 has an extra 'global_cycle' column = cycle + global_cycle_offset.

    m_charge and m_eff (the new charge-phase diagnostics) are computed here
    and fed into print_report/plot_results, but are NOT included in the
    return tuple — they're per-file-report-only for now and aren't yet
    mirrored in the combined multi-file summary plot.

    Method 1 is intentionally NOT computed — see method1_charge_integral().
    """
    df, meta, fmt = load_csv(filepath)
    R_discharge   = measure_R_discharge(df)
    m2            = method2_single_tau(df, R_discharge)
    m_extra       = method_additional_metrics(df)
    m_esr         = method_esr_1khz(df, m2, R_discharge)
    m_charge      = method_charge_metrics(df)
    m_eff         = compute_efficiency(m_charge, m_extra)

    if not m2.empty:
        m2["global_cycle"]       = m2["cycle"]      + global_cycle_offset
        next_offset = int(m2["cycle"].max()) + 1 + global_cycle_offset
    else:
        next_offset = global_cycle_offset

    if not m_extra.empty:
        m_extra["global_cycle"]  = m_extra["cycle"] + global_cycle_offset
    if not m_esr.empty:
        m_esr["global_cycle"]    = m_esr["cycle"]   + global_cycle_offset

    sr = df.attrs.get("sample_rate", 100)
    print_report(m2, m_extra, m_esr, m_charge, m_eff, R_discharge, fmt, sr)

    stem      = filepath.stem
    # Write per-file report to cwd (uploads dir may be read-only)
    plot_path = Path(f"{stem}_fatigue_report.png")
    print(f"  Generating per-file report → {plot_path.name}")
    plot_results(df, m2, m_extra, m_esr, m_charge, m_eff, R_discharge, plot_path)

    return m2, m_extra, m_esr, R_discharge, fmt, next_offset


def _plot_combined(all_m2: pd.DataFrame, all_extra: pd.DataFrame,
                   all_esr: pd.DataFrame, file_boundaries: list[int],
                   output_path: Path) -> None:
    """
    Combined summary plot across ALL files/cycles.
    Shows global degradation trend with file boundaries marked.

    NOTE: the linear RUL extrapolation line that used to run off the right
    edge of the bottom panel has been removed — same reasoning as in
    print_report (extrapolating ~5x past observed data from a fit that
    doesn't match the curve's actual shape). The flat EOL-80% threshold
    line is kept since it's just a fixed reference, not an extrapolation.

    The new charge-phase diagnostics (leakage, efficiency, time-to-plateau)
    are not yet mirrored here — this combined view still covers only
    C_test, discharge energy, V@2s/ESR, same as before.
    """
    fig = plt.figure(figsize=(20, 10), facecolor=_BG)
    fig.suptitle(
        "Combined Fatigue Summary — All Files",
        fontsize=14, fontweight="bold", color=_TEXT, y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    def _mark_boundaries(ax):
        for b in file_boundaries[1:]:   # skip first (= 0)
            ax.axvline(b, color=_GREY, lw=0.7, linestyle=":", alpha=0.6)

    # ── [0,0]  C_test across all files ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    if not all_m2.empty:
        gc = all_m2["global_cycle"].values
        Ct = all_m2["C_test_uF"].values
        ax.scatter(gc, Ct, color=_BLUE, s=3, alpha=0.6, label="C_test")
        sl, ic, r2, _ = _trend_line(gc, Ct)
        if not np.isnan(sl):
            x_lr = np.array([gc[0], gc[-1]])
            ax.plot(x_lr, sl * x_lr + ic, "--", color=_RED, lw=1.8,
                    label=f"{sl:+.4f} uF/cycle  R2={r2:.3f}")
        ax.axhline(C_NOMINAL_uF, color=_GREY, lw=0.8, linestyle=":")
        C_eol = float(Ct[0]) * 0.80
        ax.axhline(C_eol, color=_RED, lw=0.8, linestyle="--",
                   label=f"EOL 80%  ({C_eol:.0f} uF)")
        _mark_boundaries(ax)
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
    _ax_style(ax, "C_test — All Files Combined", "Global Cycle", "C_test (uF)")

    # ── [0,1]  Discharge energy ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if not all_extra.empty:
        gc_e = all_extra["global_cycle"].values
        E_v  = all_extra["E_discharge_mJ"].values
        ax.scatter(gc_e, E_v, color=_YELL, s=3, alpha=0.6, label="E_discharge")
        sl_e, ic_e, r2_e, _ = _trend_line(gc_e, E_v)
        if not np.isnan(sl_e):
            x_e = np.array([gc_e[0], gc_e[-1]])
            ax.plot(x_e, sl_e * x_e + ic_e, "--", color=_RED, lw=1.8,
                    label=f"{sl_e:+.5f} mJ/cycle  R2={r2_e:.3f}")
        _mark_boundaries(ax)
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
    _ax_style(ax, "Discharge Energy — All Files", "Global Cycle", "Energy (mJ)")

    # ── [0,2]  V@2s or ESR if 1kHz data present ──────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if not all_esr.empty:
        gc_esr = all_esr["global_cycle"].values
        esr_v  = all_esr["ESR_proxy_ohm"].values
        ax.scatter(gc_esr, esr_v, color=_RED, s=3, alpha=0.6, label="ESR proxy")
        sl_esr, ic_esr, r2_esr, _ = _trend_line(gc_esr, esr_v)
        if not np.isnan(sl_esr):
            ax.plot(np.array([gc_esr[0], gc_esr[-1]]),
                    sl_esr * np.array([gc_esr[0], gc_esr[-1]]) + ic_esr,
                    "--", color=_YELL, lw=1.8,
                    label=f"{sl_esr:+.4f} Ω/cycle  R2={r2_esr:.3f}")
        _mark_boundaries(ax)
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
        _ax_style(ax, "ESR Proxy — All 1 kHz Files", "Global Cycle", "ESR proxy (Ω)")
    elif not all_extra.empty:
        gc_e = all_extra["global_cycle"].values
        V2_v = all_extra["V_at_t2s_V"].values
        ax.scatter(gc_e, V2_v, color=_GREEN, s=3, alpha=0.6, label="V@t=2s")
        sl_v, ic_v, r2_v, _ = _trend_line(gc_e, V2_v)
        if not np.isnan(sl_v):
            x_e = np.array([gc_e[0], gc_e[-1]])
            ax.plot(x_e, sl_v * x_e + ic_v, "--", color=_RED, lw=1.8,
                    label=f"{sl_v:+.5f} V/cycle  R2={r2_v:.3f}")
        _mark_boundaries(ax)
        ax.legend(fontsize=6, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID)
        _ax_style(ax, "V @ t=2 s — All Files", "Global Cycle", "Voltage (V)")

    # ── [1,:]  tau_capacitance — full combined history ────────────────────────
    ax = fig.add_subplot(gs[1, :])
    if not all_m2.empty:
        gc = all_m2["global_cycle"].values
        Ct = all_m2["C_test_uF"].values
        ax.scatter(gc, Ct, color=_BLUE, s=4, alpha=0.5, zorder=4,
                   label="C_test (tau/R)")
        sl, ic, r2, _ = _trend_line(gc, Ct)
        if not np.isnan(sl):
            x_lr = np.linspace(gc[0], gc[-1], 300)
            ax.plot(x_lr, sl * x_lr + ic, "-", color=_BLUE, lw=2.0, alpha=0.85,
                    label=f"Trend  {sl:+.4f} uF/cycle  R2={r2:.3f}")
        ax.axhline(C_NOMINAL_uF, color=_GREY, lw=0.9, linestyle=":",
                   label="Nominal 1000 uF")
        C_eol_line = float(Ct[0]) * 0.80
        ax.axhline(C_eol_line, color=_RED, lw=0.9, linestyle="--",
                   label=f"EOL 80%  ({C_eol_line:.0f} uF)")
        ax.fill_between(gc, 0, Ct, alpha=0.08, color=_BLUE)
        # Mark file boundaries with labels
        for i, b in enumerate(file_boundaries[1:], 1):
            ax.axvline(b, color=_GREY, lw=0.8, linestyle=":", alpha=0.7)
            ax.text(b + 5, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1000,
                    f"F{i+1}", color=_GREY, fontsize=7, va="top")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, facecolor=_PANEL, labelcolor=_TEXT, edgecolor=_GRID,
                  loc="upper right", ncol=3)
    _ax_style(ax, "tau_capacitance — Combined C_test History (All Files)",
              "Global Cycle Number", "Capacitance (uF)")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    print(f"  Combined summary saved → {output_path}")


def main():
    # Resolve input — CLI arg overrides default.
    # Defaults to the directory where this .py script is located.
    raw_input = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent
    input_path = Path(raw_input)

    if not input_path.exists():
        print(f"ERROR: path not found: {input_path}")
        sys.exit(1)

    # ── Folder mode ───────────────────────────────────────────────────────────
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("daq_data_*.csv"))
        if not csv_files:
            csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"ERROR: no CSV files found in {input_path}")
            sys.exit(1)

        print(f"\nFolder mode: found {len(csv_files)} CSV file(s) in {input_path}")
        print("─" * 60)

        all_m2_list    = []
        all_extra_list = []
        all_esr_list   = []
        file_boundaries = [0]    # global cycle at which each file starts
        global_offset  = 0

        for i, fpath in enumerate(csv_files):
            print(f"\n[{i+1}/{len(csv_files)}]  {fpath.name}")
            print("─" * 50)
            m2, m_extra, m_esr, _, _, next_off = _process_single_file(
                fpath, global_offset
            )
            all_m2_list.append(m2)
            all_extra_list.append(m_extra)
            all_esr_list.append(m_esr)
            file_boundaries.append(next_off)
            global_offset = next_off

        # Concatenate all results
        all_m2    = pd.concat([d for d in all_m2_list    if not d.empty], ignore_index=True)
        all_extra = pd.concat([d for d in all_extra_list if not d.empty], ignore_index=True)
        esr_ne  = [d for d in all_esr_list if not d.empty]
        all_esr = pd.concat(esr_ne, ignore_index=True) if esr_ne else pd.DataFrame()
        # Write combined report to cwd (folder itself may be read-only)
        combined_path = Path("combined_fatigue_summary.png")
        print(f"\nGenerating combined summary ({len(all_m2)} total cycles)...")
        _plot_combined(all_m2, all_extra, all_esr, file_boundaries, combined_path)
        print("\nFolder analysis complete.")

    # ── Single file mode ──────────────────────────────────────────────────────
    else:
        print(f"\nLoading  {input_path}")
        _process_single_file(input_path)


if __name__ == "__main__":
    main()
