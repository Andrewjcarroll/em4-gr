import subprocess
import csv
import os
import re
import math
from scipy.optimize import differential_evolution

# === CONFIGURATION ===
PARAM_FILE = "em4_simplified.param.toml"
LOG_DIR = "logs"
LOG_CSV_ALL = os.path.join(LOG_DIR, "optimization_log.csv")
LOG_CSV_COMPLETED = os.path.join(LOG_DIR, "optimization_log_completed.csv")
MPI_CMD = ["mpirun", "-np", "24", "em4Solver", PARAM_FILE]

COEFF_BOUNDS = [(0.0, 0.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
ENABLED_ERRORS = ["*"]
LOG_FLOOR = 1e-12  # avoid log(0)

# === SETUP ===
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "runs"), exist_ok=True)

# === COUNTERS ===
count_completed = 0
count_crashed = 0
run_counter = 0

def update_param_file(coeffs):
    with open(PARAM_FILE, "r") as f:
        lines = f.readlines()
    coeff_str = ",".join(f"{v:.16f}" for v in coeffs)
    new_lines = [f"SOLVER_DERIV_FIRST_COEFFS = [{coeff_str}]\n" if line.strip().startswith("SOLVER_DERIV_FIRST_COEFFS") else line for line in lines]
    with open(PARAM_FILE, "w") as f:
        f.writelines(new_lines)
    print(f"🔧 Updated {PARAM_FILE} with coefficients [{coeff_str}]")

def run_simulation(coeffs, run_id):
    run_dir = os.path.join(LOG_DIR, "runs")
    coeff_str = "_".join(f"{v:+.6f}" for v in coeffs)
    run_log_file = os.path.join(run_dir, f"run_{run_id:04d}_{coeff_str}.txt")

    print(f"🚀 Running simulation... Logging to {run_log_file}")
    with open(run_log_file, "w") as logfile:
        result = subprocess.run(MPI_CMD, stdout=logfile, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, MPI_CMD)

    print("✅ Simulation complete.")
    return run_log_file

# === REGEX PATTERNS ===
rmse_pattern = re.compile(r'\[var\]:\s+(\w+_DIFF)\s+\(min, max, l2, rmse, nrmse, mae\)\s*:\s*\([^,]+,\s*[^,]+,\s*[^,]+,\s*([-\d.eE+]+)')
div_pattern = re.compile(r'\[const\]:\s*(C_DIVE|C_DIVB)\s+\(min, max, l2\)\s*:\s*\([^,]+,\s*[^,]+,\s*([-\d.eE+]+)')
step_pattern = re.compile(r"Current Step:\s+(\d+)\s+Current time:")

def parse_metrics(logfile, verbose=True):
    if not os.path.exists(logfile):
        raise FileNotFoundError(f"{logfile} not found.")

    use_all = "*" in ENABLED_ERRORS
    enabled_set = set(ENABLED_ERRORS)

    with open(logfile, "r") as f:
        lines = f.readlines()

    current_step = None
    current_time = 0.0
    step_errors = {}
    step_times = {}

    for line in lines:
        step_match = step_pattern.search(line)
        if step_match:
            try:
                current_step = int(step_match.group(1))
                time_search = re.search(r"Current time:\s*([-\d.eE+]+)", line)
                if time_search:
                    current_time = float(time_search.group(1))
                step_times[current_step] = current_time
                if current_step not in step_errors:
                    step_errors[current_step] = []
            except ValueError:
                continue

        rmse_match = rmse_pattern.search(line)
        if rmse_match and current_step is not None:
            varname = rmse_match.group(1)
            if use_all or varname in enabled_set:
                try:
                    val = float(rmse_match.group(2))
                    step_errors[current_step].append(abs(val))
                except ValueError:
                    continue

        div_match = div_pattern.search(line)
        if div_match and current_step is not None:
            varname = div_match.group(1)
            if use_all or varname in enabled_set:
                try:
                    val = float(div_match.group(2))
                    step_errors[current_step].append(abs(val))
                except ValueError:
                    continue

    if not step_errors:
        raise RuntimeError("No RMSE or constraint values found in the log.")

    step_metrics = {step: sum(vals) for step, vals in step_errors.items()}
    times = [step_times[s] for s in sorted(step_metrics.keys())]
    max_time = max(times)

    early_steps = [step for step in step_metrics if step_times[step] <= 20]
    late_steps = [step for step in step_metrics if step_times[step] >= (max_time - 2.0)]

    if not early_steps or not late_steps:
        raise RuntimeError("Not enough data in early or late time regions.")

    early_errors = [step_metrics[s] for s in early_steps]
    full_errors = list(step_metrics.values())

    log_avg_early = sum(math.log(e + LOG_FLOOR) for e in early_errors) / len(early_errors)
    log_avg_full = sum(math.log(e + LOG_FLOOR) for e in full_errors) / len(full_errors)

    combined_log_avg = 0.5 * (log_avg_early + log_avg_full)

    if verbose:
        print(f"✅ Parsed {len(step_metrics)} steps")
        print(f"📊 Early Time Avg Log Error: {log_avg_early:.6f}")
        print(f"📊 Full Time Avg Log Error: {log_avg_full:.6f}")
        print(f"📊 Combined Log Error (Objective): {combined_log_avg:.6f}")

    return log_avg_early, log_avg_full, combined_log_avg

def log_result(coeffs, log_early, log_full, completed=True):
    global count_completed, count_crashed

    write_header_all = not os.path.exists(LOG_CSV_ALL)
    with open(LOG_CSV_ALL, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header_all:
            writer.writerow(["Coeff_0", "Coeff_1", "Coeff_2", "Coeff_3", "Coeff_4", "Early_Log_Error", "Full_Log_Error"])
        writer.writerow([*coeffs, log_early, log_full])

    if completed:
        write_header_success = not os.path.exists(LOG_CSV_COMPLETED)
        with open(LOG_CSV_COMPLETED, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header_success:
                writer.writerow(["Coeff_0", "Coeff_1", "Coeff_2", "Coeff_3", "Coeff_4", "Early_Log_Error", "Full_Log_Error"])
            writer.writerow([*coeffs, log_early, log_full])
        count_completed += 1
    else:
        count_crashed += 1

    status = "✅ Completed" if completed else "❌ Crashed"
    coeff_str = ", ".join(f"{v:.6f}" for v in coeffs)
    print(f"{status} — Logged: [{coeff_str}], Early Log Error = {log_early:.6e}, Full Log Error = {log_full:.6e}")

def objective(x):
    global run_counter
    run_counter += 1
    coeffs = x.tolist()
    print(f"\n=== 🧪 Testing Coefficients (Run {run_counter}): {coeffs} ===")
    try:
        update_param_file(coeffs)
        logfile = run_simulation(coeffs, run_counter)
        log_early, log_full, combined_log = parse_metrics(logfile)
        log_result(coeffs, log_early, log_full, completed=True)
        return combined_log

    except subprocess.CalledProcessError as e:
        print(f"❌ Simulation crashed (return code {e.returncode})")
    except RuntimeError as e:
        print(f"⚠ Incomplete simulation data: {e}")
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")

    fallback_log = float("inf")
    log_result(coeffs, fallback_log, fallback_log, completed=False)
    return fallback_log

# === MAIN ===
if __name__ == "__main__":
    print("🧬 Starting optimization of 5 coefficients using Differential Evolution...\n")
    print("📏 Using Coefficient Bounds:")
    for i, (lo, hi) in enumerate(COEFF_BOUNDS):
        print(f"   Coeff_{i}: [{lo}, {hi}]")

    result = differential_evolution(
        func=objective,
        bounds=COEFF_BOUNDS,
        strategy='best1bin',
        maxiter=5000,
        popsize=100,
        disp=True,
        polish=True,
        seed=45
    )

    print("\n✅ Optimization complete!")
    print(f"🔧 Best Coefficients: {[round(c, 6) for c in result.x]}")
    print(f"📉 Minimum Combined Log Avg Error: {result.fun:.6e}")
    print("\n📊 Summary:")
    print(f"   ✅ Completed runs : {count_completed}")
    print(f"   ❌ Crashed runs    : {count_crashed}")
