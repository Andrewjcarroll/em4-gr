import subprocess
import csv
import os
import re
import math
from scipy.optimize import differential_evolution

# === CONFIGURATION ===
PARAM_FILE = "em4_simplified.param.toml"
LOG_FILE = "outputdata.txt"
LOG_DIR = "logs"
LOG_CSV = os.path.join(LOG_DIR, "optimization_log.csv")
MPI_CMD = ["mpirun", "-np", "24", "em4Solver", PARAM_FILE]

COEFF_BOUNDS = [(0.0, 0.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
ENABLED_ERRORS = ["C_DIVB"]  # or use ["*"]

os.makedirs(LOG_DIR, exist_ok=True)

def update_param_file(coeffs):
    with open(PARAM_FILE, "r") as f:
        lines = f.readlines()

    coeff_str = ",".join(f"{v:.16f}" for v in coeffs)
    new_lines = []
    for line in lines:
        if line.strip().startswith("SOLVER_DERIV_FIRST_COEFFS"):
            new_lines.append(f"SOLVER_DERIV_FIRST_COEFFS = [{coeff_str}]\n")
        else:
            new_lines.append(line)

    with open(PARAM_FILE, "w") as f:
        f.writelines(new_lines)
    print(f"🔧 Updated {PARAM_FILE} with coefficients [{coeff_str}]")


def run_simulation(coeffs):
    print(f"🚀 Running simulation... Logging to {LOG_FILE}")
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    with open(LOG_FILE, "w") as logfile:
        result = subprocess.run(MPI_CMD, stdout=logfile, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, MPI_CMD)
    print("✅ Simulation complete.")


# === REGEX ===
rmse_pattern = re.compile(
    r'\[var\]:\s+(\w+_DIFF)\s+\(min, max, l2, rmse, nrmse, mae\)\s*:\s*\([^,]+,\s*[^,]+,\s*[^,]+,\s*([-\d.eE+]+)'
)
div_pattern = re.compile(
    r'\[const\]:\s*(C_DIVE|C_DIVB)\s+\(min, max, l2\)\s*:\s*\([^,]+,\s*[^,]+,\s*([-\d.eE+]+)'
)
step_pattern = re.compile(r"Current Step:\s+(\d+)\s+Current time:")


def parse_logsum_metric(verbose=True):
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(f"{LOG_FILE} not found.")

    use_all = "*" in ENABLED_ERRORS
    enabled_set = set(ENABLED_ERRORS)

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    current_step = None
    step_errors = {}
    ignored_zeros = 0

    for line in lines:
        step_match = step_pattern.search(line)
        if step_match:
            try:
                current_step = int(step_match.group(1))
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
                    if val > 0.0:
                        step_errors[current_step].append(val)
                    else:
                        ignored_zeros += 1
                except ValueError:
                    continue

        div_match = div_pattern.search(line)
        if div_match and current_step is not None:
            varname = div_match.group(1)
            if use_all or varname in enabled_set:
                try:
                    val = float(div_match.group(2))
                    if val > 0.0:
                        step_errors[current_step].append(val)
                    else:
                        ignored_zeros += 1
                except ValueError:
                    continue

    if not step_errors:
        raise RuntimeError("No RMSE or constraint values found in the log.")

    step_metrics = {
        step: sum(math.log(val) for val in vals)
        for step, vals in step_errors.items()
        if vals
    }

    num_steps = len(step_metrics)
    if num_steps == 0:
        raise RuntimeError("All errors were zero — nothing to optimize.")

    average_logsum = sum(step_metrics.values()) / num_steps

    if verbose:
        print(f"✅ Parsed {num_steps} steps")
        if ignored_zeros > 0:
            print(f"⚠️ Ignored {ignored_zeros} zero error values")
        for step, val in sorted(step_metrics.items()):
            print(f"   Step {step:3d}: Log-Sum Error = {val:.6e}")
        print(f"📊 Average Log-Sum Error Across Steps = {average_logsum:.6e}")

    return average_logsum, step_metrics, num_steps


def parse_last_step():
    if not os.path.exists(LOG_FILE):
        return 0

    last_step = 0
    with open(LOG_FILE, "r") as f:
        for line in f:
            match = step_pattern.search(line)
            if match:
                try:
                    last_step = int(match.group(1))
                except ValueError:
                    continue
    return last_step


def log_result(coeffs, metric):
    write_header = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Coeff_0", "Coeff_1", "Coeff_2", "Coeff_3", "LogSum_Avg_Error"])
        writer.writerow([*coeffs, metric])
    coeff_str = ", ".join(f"{v:.6f}" for v in coeffs)
    print(f"🗘️ Logged: [{coeff_str}], LogSum Avg Error = {metric:.6e}")


def objective(x):
    coeffs = x.tolist()
    print(f"\n=== 🧪 Testing Coefficients: {coeffs} ===")
    try:
        update_param_file(coeffs)
        run_simulation(coeffs)
        metric, step_metrics, num_steps = parse_logsum_metric()
        print(f"📊 LogSum Avg Error over {num_steps} steps = {metric:.6e}")
        log_result(coeffs, metric)
        return metric

    except subprocess.CalledProcessError as e:
        print(f"❌ Simulation crashed (return code {e.returncode})")

    except RuntimeError as e:
        print(f"⚠️ Partial log found, but no error terms parsed: {e}")

    except Exception as e:
        print(f"❌ Unexpected exception during simulation: {e}")

    last_step = parse_last_step()
    if last_step > 0:
        print(f"⚠️ Using fallback metric: Simulation crashed at step {last_step}")
        metric = 1e6 - last_step * 1e3
    else:
        print("⚠️ Simulation failed before any time steps were completed.")
        metric = 1e9

    log_result(coeffs, metric)
    return min(metric, 1e9)


# === MAIN ===
if __name__ == "__main__":
    print("🧬 Starting optimization of 4 coefficients using Differential Evolution...\n")

    print("📏 Using Coefficient Bounds:")
    for i, (lo, hi) in enumerate(COEFF_BOUNDS):
        print(f"   Coeff_{i}: [{lo}, {hi}]")

    result = differential_evolution(
        func=objective,
        bounds=COEFF_BOUNDS,
        strategy='best1bin',
        maxiter=500,
        popsize=10,
        disp=True,
        polish=True,
        seed=45
    )

    print("\n✅ Optimization complete!")
    print(f"🔧 Best Coefficients: {[round(c, 6) for c in result.x]}")
    print(f"📉 Minimum Log-Sum Avg Error: {result.fun:.6e}")


