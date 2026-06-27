# Capacitor_Charge–Discharge_Cycling_and_Degradation_Analysis

Sensorless capacitor fatigue / Remaining-Useful-Life (RUL) monitoring.
A programmable PSU and NI-DAQ cycle a capacitor through repeated charge/discharge;
a post-analysis pipeline then extracts degradation metrics purely from the logged
voltage/current telemetry — no external ESR meter needed (above ~1 kHz sampling).

| File | Role |
|---|---|
| `src/capacitor_charge_discharge_daq.py` | **Acquisition.** Drives PSU charge/discharge, logs raw V/I at 100 Hz. |
| `src/capacitor_degradation_analysis.py` | **Post-analysis.** Fits decay models, extracts fatigue metrics, plots trends. |

---

## 1. Circuit Topology

```
PSU(+) ──[ R_shunt = 220 Ω ]── Node B ──┬── Capacitor (DUT) ──── PSU(−) / GND
                                         │
                                  Voltage divider / attenuator
                                  (always connected — NOT switched)
                                  ai0 reads the divider tap → ×3.2 → V_cap
                                  Equivalent bleed resistance: R_output = 32 kΩ
```

- **ai0** (differential): capacitor voltage, attenuated ~3.2× so a 25 V signal
  fits inside the DAQ's ±10 V differential input range.
- **ai1** (differential): voltage across the 220 Ω shunt → current via Ohm's law.
- The divider sits permanently in parallel across the capacitor. It's not just a
  sense tap — it's also a small constant leakage/bleed path, which is why it
  shows up as a correction term in the current equation below.

## 2. KCL Current Reconstruction

At Node B:

```
i_shunt = i_cap + i_bleed
i_cap   = (V_shunt / R_shunt) − (V_cap / R_output)
```

- **Charging:** PSU sources current through the shunt → `i_shunt` large and positive.
- **Discharging:** PSU output is off → `i_shunt ≈ 0`, so `i_cap` resolves
  **negative** — the capacitor is now the source, draining back through the
  shunt to ground.

## 3. Charge / Discharge State Machine

One thread (`psu_thread`) polls `MEAS:ALL?` over PyVISA and toggles the PSU output:

| Condition | Action |
|---|---|
| mode = charge **and** V ≥ 25.0 − 0.04 V | `OUTP OFF` → mode = discharge |
| mode = discharge **and** V ≤ 0.0 + 0.05 V | `OUTP ON` → mode = charge |

A `sleep(5)` after each transition lets the PSU's own output relay settle before
the next poll. There's no separate Arduino/relay board in this lean version —
the **PSU's own output switch is the charge/discharge switch**.

`cycle_number` increments once per **discharge → charge** edge (one full cycle =
one charge + one discharge), which avoids the double-count you'd get from
incrementing on every half-cycle.

## 4. DAQ Acquisition Loop

- 100 Hz, 2-channel differential, continuous mode, 100-sample blocks.
- Each sample is timestamped from the DAQ sample index (`sample_count / SAMPLE_RATE`),
  not the OS clock — immune to thread/CPU scheduling jitter.
- Rows buffer in memory (8000 rows ≈ 80 s) and flush together, so disk I/O never
  blocks the tight acquisition loop.

**CSV schema:**
```
time_s, channel0_voltage, capacitor_voltage, shunt_voltage, capacitor_current, psu_mode, cycles
```

## 5. Degradation Analysis — Method Summary

| Method | Formula | What it isolates |
|---|---|---|
| **Primary: τ-fit** | `V(t) = V₀·e^(−t/τ)` on the **[5%–50%]** tail → `C_test = τ / R_discharge` | True dielectric decay, clear of the relay/PSU-off transient |
| R_discharge | empirical `median(V / \|I\|)` over the same clean tail | No datasheet assumption needed |
| Discharge energy | `E = ∫ V·\|I\| dt` | Tau-independent cross-check |
| V @ t = 2 s | `V(2) = V₀·e^(−2/τ)` | Single-point, fit-free sanity check |
| Leakage current | median `\|I\|` on the settled charge plateau | Independent failure mode (oxide vs. electrolyte drying) |
| Time-to-plateau | charge ramp duration | Voltage-only — doesn't touch the current chain at all |
| Round-trip efficiency | `E_discharge / E_charge` | Rising loss (ESR + leakage). **Unreliable on the 100 Hz "minimal" CSV format** — see code docstring |

**Why [5%–50%] and not the whole curve:** the first ~20–30% of discharge still
carries the relay/PSU hold-off plateau and a fast transient sitting on top of the
clean RC decay. Fitting through that drags τ upward and roughly triples the
residual error.

## 6. ESR — Why It Needs ≥ 1 kHz

ESR shows up as an **instantaneous IR step** at the moment the PSU output drops,
sitting on top of the much slower RC decay set by `τ = R_discharge × C`.

```
τ_initial  = −1 / slope(ln V)   over the first 10 samples after the PSU-off edge
τ_tail     = τ from the main [5%–50%] fit
ESR_proxy  = R_discharge × (τ_tail − τ_initial) / (τ_initial × τ_tail)
```

`ESR_proxy → 0` for an ideal capacitor (`τ_initial ≈ τ_tail`); it rises as ESR rises.

- **At 100 Hz**, one sample = 10 ms, so the 10-sample onset window already spans
  100 ms — long enough that you're sampling well into the RC tail, not the
  instantaneous step. The two regimes blend into one number and `τ_initial`
  stops meaning anything.
- **At ≥1 kHz**, the same 10-sample window spans ≤10 ms — short enough to catch
  the step before the RC decay has moved appreciably, so `τ_initial` and
  `τ_tail` separate cleanly.

This is exactly why `method_esr_1khz()` returns nothing below 900 Hz rather than
reporting a misleading number.

## 7. Adapting This for a Plain RC Circuit (no divider, no bleed path)

If your rig is just `PSU → shunt → capacitor → GND`, with the DAQ wired straight
across the capacitor (no attenuating divider), these are the lines to change in
**your copy** of the acquisition script (this repo's copy is left untouched):

| Line / constant | Change to | Why |
|---|---|---|
| `cap_voltage = v_ch0 * 3.2` | `cap_voltage = v_ch0` | No divider → no attenuation factor |
| `i_bleed = cap_voltage / R_OUTPUT` | drop the term, `i_cap = i_shunt` | No parallel bleed path to subtract |
| ai0 channel `min_val` / `max_val` | must bracket your **un-attenuated** capacitor voltage | ⚠️ see caution below |

> ⚠️ **Caution:** the NI USB-6001's differential input only handles ±10 V. If
> your capacitor charges above ~9–10 V and you simply remove the divider, you
> will clip the ADC or risk damaging the input. You still need *some*
> attenuation — just maybe not this exact ratio. Size your own divider for your
> voltage rating and update the `×3.2` multiplier and the channel's
> `min_val`/`max_val` together; don't strip the divider and leave the channel
> range unchanged.

If you keep a divider but its impedance is high enough that bleed current is
negligible at your shunt's sensitivity, set `R_OUTPUT` to something very large
(e.g. `1e9`) instead of deleting the term — keeps the KCL equation general and
avoids surprises if you later add the bleed path back.

## 8. Things That Will Bite You (lessons from building this)

- **PSU SCPI dialect**: `MEAS:ALL?` and the comma-split parsing in `psu_thread`
  assume this exact OWON reply format. Other PSU brands return a different
  field order/count — query `*IDN?` and `MEAS:ALL?` manually before trusting
  the parser.
- **VISA address is hardware-specific**: `OWON_ADDRESS` encodes this PSU's USB
  vendor/product/serial IDs. Run `pyvisa.ResourceManager().list_resources()` to
  find yours.
- **Shared state race**: `psu_mode` is read by both threads. It's already
  behind a lock — if you extend the state machine, keep it that way; the
  edge-triggered cycle counter assumes `psu_mode` doesn't change mid-read.
- **Long-run CSVs get big fast**: 8 hours × 100 Hz × 2 channels ≈ 2.9M rows.
  Don't open these in Excel — the analysis script is the intended reader.
- **Hold-off threshold (`RELAY_HOLDOFF_DVDT = -5.0 V/s`)** assumes a fairly
  fast discharge onset. If your RC time constant is much slower than this
  rig's (~1.1 s), loosen this threshold or the fit window will skip real data.
- **"minimal" vs "full" CSV format**: the analysis script auto-detects which
  columns are present. If you add your own derived columns to the acquisition
  CSV, name them to match the `full_cols` set in `load_csv()`, or the format
  detector will misclassify the file.

## 9. Known Limitations (intentionally left unfixed)

- **No automatic RUL cycle-count extrapolation.** An earlier version linearly
  extrapolated `C_test` out to an 80%-EOL cycle number; it routinely projected
  5–20× past the observed data and didn't match the curve's real shape (steep
  initial drop, then a long flatter tail). It's been removed — the pipeline
  now reports only the linear *slope* (μF/cycle) with an explicit significance
  check (R², p-value), not a predicted cycle count.
- **Thermal relaxation spikes aren't separated from real fatigue.** After rest
  periods, `C_test` shows a brief upward spike as the capacitor cools back to
  ambient before resuming its decay trend — visible at file boundaries in the
  combined plot. The pipeline doesn't currently flag or exclude these; treat
  the first few cycles after a long gap with some skepticism.

## 10. Usage

```bash
pip install -r requirements.txt

# Acquisition (Ctrl+C, or 'q' + Enter, to stop safely)
python src/capacitor_charge_discharge_daq.py

# Analysis — single file or a folder of daq_data_*.csv files
python src/capacitor_degradation_analysis.py path/to/daq_data_folder
```

## License

MIT — see `LICENSE`.
