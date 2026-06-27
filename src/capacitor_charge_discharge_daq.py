import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import nidaqmx
import pyvisa
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# --------------------------------------------------
# HARDWARE CONFIGURATION
# --------------------------------------------------
OWON_ADDRESS   = "USB0::0x5345::0x1235::25270017::INSTR"
TARGET_VOLTAGE = 25.0
LOW_VOLTAGE    = 0.0
CHARGE_CURRENT = 4.0
CHANNEL        = 1

# Hardware Mapping
NIDAQ_DEVICE_CH0 = "Dev1/ai0"  # Attenuated Capacitor Voltage
NIDAQ_DEVICE_CH1 = "Dev1/ai1"  # Shunt Voltage for KCL Current
R_SHUNT  = 220.0
R_OUTPUT = 32000.0

SAMPLE_RATE = 100
BLOCK_SIZE  = 100
DURATION    = 28800           # 6 hours

timestamp_str   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DAQ_OUTPUT_FILE = Path(f"daq_data_{timestamp_str}.csv")

# --------------------------------------------------
# SHARED SYSTEM STATE
# --------------------------------------------------
running    = True
psu_mode   = "charge"
state_lock = threading.Lock()

def get_running() -> bool:
    with state_lock:
        return running

def set_running(value: bool) -> None:
    global running
    with state_lock:
        running = value

def get_psu_mode() -> str:
    with state_lock:
        return psu_mode

def set_psu_mode(value: str) -> None:
    global psu_mode
    with state_lock:
        psu_mode = value

# --------------------------------------------------
# THREAD 1: STOP LISTENER
# --------------------------------------------------
def stop_listener() -> None:
    while get_running():
        try:
            command = input().lower().strip()
        except EOFError:
            return
        if command in ("q", "quit", "stop"):
            set_running(False)
            print("\nStopping all threads...")
            return

# --------------------------------------------------
# THREAD 2: POWER SUPPLY AUTOMATION
# --------------------------------------------------
def psu_thread() -> None:
    rm  = pyvisa.ResourceManager()
    psu = None
    try:
        psu = rm.open_resource(OWON_ADDRESS)
        psu.timeout           = 5000
        psu.read_termination  = "\n"
        psu.write_termination = "\n"

        psu.write("OUTP OFF")
        time.sleep(1)
        print("PSU IDN ->", psu.query("*IDN?").rstrip())
        psu.write(f"INST:NSEL {CHANNEL}")
        psu.write(f"VOLT {TARGET_VOLTAGE:.4f}")
        psu.write(f"CURR {CHARGE_CURRENT:.4f}")
        psu.write("OUTP ON")
        set_psu_mode("charge")
        print("PSU thread started.")

        while get_running():
            try:
                raw     = psu.query("MEAS:ALL?").rstrip()
                vals    = raw.split(",")
                voltage = float(vals[0])
            except Exception:
                continue

            current_mode = get_psu_mode()
            if current_mode == "charge" and voltage >= TARGET_VOLTAGE - 0.04:
                psu.write("OUTP OFF")
                set_psu_mode("discharge")
                print(f"[PSU] -> Switched to DISCHARGE at {voltage:.3f} V")
                time.sleep(5)
            elif current_mode == "discharge" and voltage <= LOW_VOLTAGE + 0.05:
                psu.write("OUTP ON")
                set_psu_mode("charge")
                print(f"[PSU] -> Switched to CHARGE at {voltage:.3f} V")
                time.sleep(5)
            time.sleep(1)
    except Exception as e:
        print(f"[PSU] Error: {e}")
    finally:
        if psu:
            try:
                psu.write("OUTP OFF")
                psu.close()
            except:
                pass
        print("PSU output OFF. Connection closed.")

# --------------------------------------------------
# THREAD 3: NI-DAQ HIGH-SPEED ACQUISITION
# --------------------------------------------------
def nidaq_thread() -> None:
    total_samples     = SAMPLE_RATE * DURATION
    samples_collected = 0

    BUFFER_ROWS = 8000
    row_buffer  = []

    # FIX 1: cycle_number now counts COMPLETE electrochemical cycles.
    # A complete cycle is defined as one full charge + discharge pair.
    # Increment is triggered exclusively on the discharge → charge transition,
    # preventing the previous double-count (one increment per half-cycle).
    cycle_number = 0
    prev_mode    = None

    try:
        with nidaqmx.Task() as ai_task:
            # CH0: Attenuated Capacitor Voltage
            ai_task.ai_channels.add_ai_voltage_chan(
                NIDAQ_DEVICE_CH0,
                terminal_config=TerminalConfiguration.DIFF,
                min_val=-10, max_val=10,
            )
            # CH1: Shunt Voltage for KCL
            ai_task.ai_channels.add_ai_voltage_chan(
                NIDAQ_DEVICE_CH1,
                terminal_config=TerminalConfiguration.DIFF,
                min_val=-10, max_val=10,
            )

            ai_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
            )

            with DAQ_OUTPUT_FILE.open("w", newline="") as handle:
                writer = csv.writer(handle)


                # Column header
                writer.writerow([
                    "time_s",
                    "channel0_voltage",
                    "capacitor_voltage",
                    "shunt_voltage",
                    "capacitor_current",
                    "psu_mode",
                    "cycles",
                ])
                print(f"NI-DAQ started → Logging to {DAQ_OUTPUT_FILE}")

                while get_running() and samples_collected < total_samples:
                    data              = ai_task.read(number_of_samples_per_channel=BLOCK_SIZE)
                    block_start_index = samples_collected

                    for sample_offset in range(BLOCK_SIZE):
                        # Hardware-derived monotonic timestamp — disciplined to
                        # the DAQ crystal clock; immune to OS/CPU scheduling jitter.
                        timestamp    = (block_start_index + sample_offset) / SAMPLE_RATE
                        current_mode = get_psu_mode()

                        # ── Hardware Math & Attenuation ────────────────────────
                        v_ch0       = data[0][sample_offset]
                        v_shunt     = data[1][sample_offset]
                        cap_voltage = v_ch0 * 3.2

                        # ── KCL Current Calculation ────────────────────────────
                        # Nodal analysis at capacitor (+) terminal:
                        #   i_shunt = i_cap + i_bleed
                        #   i_cap   = (V_shunt / R_shunt) - (V_cap / R_output)
                        # During discharge (PSU OFF): i_shunt ≈ 0,
                        # i_cap resolves negative (capacitor sourcing current).
                        i_shunt = v_shunt / R_SHUNT
                        i_bleed = cap_voltage / R_OUTPUT
                        i_cap   = i_shunt - i_bleed

                        # ── State Machine: Complete-Cycle Counter ──────────────
                        # FIX 1 applied here: increment only on the
                        # discharge → charge edge, so cycle_number equals the
                        # number of complete charge-discharge pairs elapsed.
                        if prev_mode == "discharge" and current_mode == "charge":
                            cycle_number += 1

                        prev_mode = current_mode

                        # ── Append to Memory Buffer ────────────────────────────
                        row_buffer.append([
                            f"{timestamp:.6f}",
                            f"{v_ch0:.6f}",
                            f"{cap_voltage:.6f}",
                            f"{v_shunt:.6f}",
                            f"{i_cap:.9f}",
                            current_mode,
                            f"{cycle_number}",
                        ])

                    # ── Batch Write to Disk ────────────────────────────────────
                    if len(row_buffer) >= BUFFER_ROWS:
                        writer.writerows(row_buffer)
                        row_buffer.clear()
                        handle.flush()

                        # Console Status Update (~ every 8 seconds)
                        print(f"Running... t={timestamp:.0f}s | Cycle={cycle_number} | "
                              f"V_cap={cap_voltage:.3f}V | I_cap={i_cap*1000:.3f}mA")

                    samples_collected += BLOCK_SIZE

    except Exception as exc:
        print(f"[DAQ] Critical Error: {exc}")
    finally:
        # Flush any remaining data in the buffer on exit
        if row_buffer:
            with DAQ_OUTPUT_FILE.open("a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(row_buffer)
        set_running(False)
        print(f"NI-DAQ run complete — {samples_collected} samples secured.")

# --------------------------------------------------
# MAIN SYSTEM EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    print("Starting Lean PSU + DAQ Acquisition System...")
    print("Press 'q' + Enter to stop safely.")

    threads = [
        threading.Thread(target=stop_listener, daemon=True),
        threading.Thread(target=psu_thread,    daemon=True),
        threading.Thread(target=nidaq_thread,  daemon=True),
    ]

    for t in threads:
        t.start()
    for t in threads[1:]:
        t.join()

    print("\nSystem shut down cleanly.")
