import argparse
import math
import os
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

# what we used!
SOLVER_DERIV_TYPES = [-1, 0]
SOLVER_2ND_DERIV_TYPES = [None]
SOLVER_FILTER_TYPES = [-1, 0, 1]
SOLVER_DERIV_CLOSURE_TYPES = [None]

plt.style.use("seaborn-v0_8-deep")

linestyle_tuple = [
    ("solid", "solid"),
    ("dotted", "dotted"),
    ("dashed", "dashed"),
    ("dashdot", "dashdot"),
    ("dotted", (0, (1, 1))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


SOLVER_DERIV_TYPE_NAMES = [
    "CFD_NONE",
    "CFD_P1_O4",
    "CFD_P1_O6",
    "CFD_Q1_O6_ETA1",
    "CFD_KIM_O4",
    "CFD_HAMR_O4",
    "CFD_JT_O6",
    "EXPLCT_FD_O4",
    "EXPLCT_FD_O6",
    "EXPLCT_FD_O8",
]

SOLVER_2ND_DERIV_TYPE_NAMES = [
    "CFD2ND_NONE",
    "CFD2ND_P2_O4",
    "CFD2ND_P2_O6",
    "CFD2ND_Q2_O6_ETA1",
    "CFD2ND_KIM_O4",
    "CFD2ND_HAMR_O4",
    "CFD2ND_JT_O6",
    "EXPLCT2ND_FD_O4",
    "EXPLCT2ND_FD_O6",
    "EXPLCT2ND_FD_O8",
]

SOLVER_FILTER_TYPE_NAMES = [
    "FILT_NONE",
    "FILT_KO_DISS",
    "FILT_KIM_6",
    "FILT_JT_6",
    "FILT_JT_8",
    "FILT_JT_10",
    "EXPLCT_KO",
]

SOLVER_DERIV_CLOSURE_TYPE_NAMES = [
    "BLOCK_CFD_CLOSURE",
    "BLOCK_CFD_DIRICHLET",
    "BLOCK_CFD_LOPSIDE_CLOSURE",
    "BLOCK_PHYS_BOUNDARY",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_pretty_name_for_measure(measure, is_diff):
    the_key = {"max": "Maximum", "min": "Minimum", "l2": "L2 Norm"}
    text = the_key[measure]

    if is_diff:
        text += " Error"

    return text


def get_deriv_type_name(idx):
    if idx is None:
        return None
    return SOLVER_DERIV_TYPE_NAMES[idx + 1]


def get_sec_deriv_type_name(idx):
    if idx is None:
        return None
    return SOLVER_2ND_DERIV_TYPE_NAMES[idx + 1]


def get_filter_type_names(idx):
    if idx is None:
        return None
    return SOLVER_FILTER_TYPE_NAMES[idx + 1]


def get_closure_type_names(idx):
    if idx is None:
        return None
    return SOLVER_DERIV_CLOSURE_TYPE_NAMES[idx]


def get_full_name(deriv_type, sec_deriv_type, filter_type, deriv_closure_type):
    deriv_name = get_deriv_type_name(deriv_type)
    sec_deriv_name = get_sec_deriv_type_name(sec_deriv_type)
    filter_name = get_filter_type_names(filter_type)
    closure_name = get_closure_type_names(deriv_closure_type)

    full_name = "" if deriv_name is None else deriv_name
    full_name += "" if sec_deriv_name is None else f"-{sec_deriv_name}"
    full_name += "" if filter_name is None else f"-{filter_name}"

    if deriv_type in [-1, 3, 6, 7, 8]:
        pass
    else:
        full_name += "" if closure_name is None else f"-{closure_name}"

    return full_name


def split_results_to_dict(results, is_custom):
    results_output = {}
    idx = 0

    if is_custom:
        for ii, res in enumerate(results):
            name = "DENDRO_RUN_" + str(ii)

            results_output[name] = results[ii]
            results_output[name]["types"] = "unknown"

        return results_output

    for ii, deriv_type in enumerate(SOLVER_DERIV_TYPES):
        for jj, sec_deriv_type in enumerate(SOLVER_2ND_DERIV_TYPES):
            for kk, filter_type in enumerate(SOLVER_FILTER_TYPES):
                for ll, deriv_closure_type in enumerate(SOLVER_DERIV_CLOSURE_TYPES):
                    if deriv_type in [-1, 3, 6, 7, 8] and ll > 0:
                        continue

                    if sec_deriv_type == [-1, 6, 7, 8] and ll > 0:
                        continue

                    the_name = get_full_name(
                        deriv_type, sec_deriv_type, filter_type, deriv_closure_type
                    )

                    print(the_name)

                    try:
                        results_output[the_name] = results[idx]
                    except IndexError:
                        # if we run out of results, then we exit early
                        return results_output
                    results_output[the_name]["types"] = [
                        deriv_type,
                        sec_deriv_type,
                        filter_type,
                        deriv_closure_type,
                    ]
                    idx += 1

    return results_output


def plot_results(results, variable, measure, is_diff, is_custom=False):
    num_types = len(results)

    if num_types > len(linestyle_tuple):
        # calculate the number of times it needs to be repeated...
        division = math.ceil(num_types / len(linestyle_tuple))
        linestyles = [x for ii in range(division) for x in linestyle_tuple]
    else:
        linestyles = linestyle_tuple

    idx = 0
    fig, axs = plt.subplots(2)

    idx = 0
    print("Now plotting results...")
    for name, data in results.items():
        # print(name, data)
        if len(data) == 0 or list(data.keys()) == ["types"]:
            print("    Skipping", name)
            continue
        if name == "CFD_P1_O4-FILT_KIM_6-BLOCK_CFD_LOPSIDE_CLOSURE":
            continue
        _, linestyle = linestyles[idx]

        axs[0].semilogy(
            data["time"][1:],
            data[variable][measure][1:],
            label=name,
            linestyle=linestyle,
            marker="o",
        )

        axs[1].plot(
            data["time"][1:],
            data[variable][measure][1:],
            label=name,
            linestyle=linestyle,
            marker="o",
        )

        # print(name, max(data[variable][measure]))

        idx += 1

    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()

    # pretty name
    measure_name = get_pretty_name_for_measure(measure, is_diff)

    plt.suptitle(
        f"{measure_name} for {variable} Function of Time for Different Parameters"
    )

    return fig


def plot_mesh_size(results, is_custom=False):
    num_types = len(results)

    if num_types > len(linestyle_tuple):
        # calculate the number of times it needs to be repeated...
        division = math.ceil(num_types / len(linestyle_tuple))
        linestyles = [x for ii in range(division) for x in linestyle_tuple]
    else:
        linestyles = linestyle_tuple

    idx = 0
    fig, axs = plt.subplots(2)

    idx = 0
    print("Now plotting results...")
    for name, data in results.items():
        if len(data) == 0 or list(data.keys()) == ["types"]:
            print("    Skipping", name)
            continue
        if name == "CFD_P1_O4-FILT_KIM_6-BLOCK_CFD_LOPSIDE_CLOSURE":
            continue
        _, linestyle = linestyles[idx]

        axs[0].semilogy(
            data["meshSize"]["time"],
            data["meshSize"]["newMesh"],
            label=name,
            linestyle=linestyle,
            marker="o",
        )

        axs[1].plot(
            data["meshSize"]["time"],
            data["meshSize"]["newMesh"],
            label=name,
            linestyle=linestyle,
            marker="o",
        )

        # print(name, max(data[variable][measure]))

        idx += 1

    axs[0].legend()
    axs[0].grid()
    axs[1].legend()
    axs[1].grid()

    # pretty name

    plt.suptitle(f"New Mesh Size as a Function of Simulation Time")

    return fig


def clean_up(curr_results, results):
    for key in curr_results.keys():
        if key == "step" or key == "time" or key == "dt":
            curr_results[key] = np.array(curr_results[key])
        elif key == "meshSize":
            curr_results[key] = curr_results[key]

        else:
            temp = curr_results[key]
            temp["min"] = np.array(temp["min"])
            temp["max"] = np.array(temp["max"])
            temp["l2"] = np.array(temp["l2"])
            curr_results[key] = temp

    results.append(curr_results)
    return results


def dump_results(results, filename):
    json_dump = json.dumps(results, cls=NumpyEncoder)

    with open(filename, "w") as f:
        f.write(json_dump)


def main(output_file_dump, skip_plotting, is_custom):
    # open the file

    with open(output_file_dump, "r") as f:
        lines = f.readlines()

    results = []
    curr_results = {}
    ii = 0
    while ii < len(lines):
        line = lines[ii].strip()

        # check if we're finished with the solver, if we are, then we exit
        if "Solver finished!" in line:
            print("Found the end of a program...  ", lines[ii - 1].strip())
            results = clean_up(curr_results, results)

            curr_results = {}
            ii += 1
            continue

        if "mpirun noticed" in line:
            # this is a crash!
            print("Found a crash:", line)
            results = clean_up(curr_results, results)

            curr_results = {}

        if "slurmstepd: error:" in line:
            print("Found a killed run:", line)
            results = clean_up(curr_results, results)
            curr_results = {}

        if "old mesh" in line and "new mesh" in line and "iter :" not in line:
            parts = line.split("\t")

            step = int(parts[0].split(":")[-1].strip())
            time = float(parts[1].split(":")[-1].strip())
            old_mesh = int(parts[2].split(":")[-1].strip())
            new_mesh = int(parts[3].split(":")[-1].strip())

            meshSizeData = curr_results.get("meshSize", {})
            stepData = meshSizeData.get("step", [])
            stepData.append(step)
            timeData = meshSizeData.get("time", [])
            timeData.append(time)
            oldMeshData = meshSizeData.get("oldMesh", [])
            oldMeshData.append(old_mesh)
            newMeshData = meshSizeData.get("newMesh", [])
            newMeshData.append(new_mesh)
            meshSizeData = {
                "step": stepData,
                "time": timeData,
                "oldMesh": oldMeshData,
                "newMesh": newMeshData,
            }
            curr_results["meshSize"] = meshSizeData

        if line == "[ETS - SOLVER] : SOLVER UPDATE":
            inside_solver_results = True
            # print(line)

            ii += 1
            current_step_line = lines[ii].strip()
            current_step_line = current_step_line.split("\t")

            step = int(current_step_line[1].split(":")[1].strip())
            time = float(current_step_line[3].split(":")[1].strip())
            dt = float(current_step_line[4].split(":")[1].strip())

            steps = curr_results.get("step", [])
            steps.append(step)
            curr_results["step"] = steps

            times = curr_results.get("time", [])
            times.append(time)
            curr_results["time"] = times

            dts = curr_results.get("dt", [])
            dts.append(dt)
            curr_results["dt"] = dts

            # go through the lines until we see "ETS time (max)"
            while True:
                ii += 1
                line = lines[ii].strip()

                if "[var]" in line or "[const]" in line:
                    line_pieces = line.split("\t")

                    var_text = line_pieces[0]
                    var_text = var_text[
                        var_text.find(":") + 1 : var_text.find("(")
                    ].strip()

                    s = line_pieces[-1]
                    min_max_l2 = s[s.find("(") + 1 : s.find(")")]
                    min_max_l2 = np.fromstring(min_max_l2, sep=",")

                    var_results = curr_results.get(var_text, {})
                    var_results_min = var_results.get("min", [])
                    var_results_max = var_results.get("max", [])
                    var_results_l2 = var_results.get("l2", [])

                    var_results_min.append(min_max_l2[0])
                    var_results_max.append(min_max_l2[1])
                    var_results_l2.append(min_max_l2[2])

                    var_results = {
                        "min": var_results_min,
                        "max": var_results_max,
                        "l2": var_results_l2,
                    }
                    curr_results[var_text] = var_results

                if "[ETS]" in line:
                    # print(line)
                    break
                if "ETS time (max)" in line:
                    # print(step)
                    # print(line)
                    break
                # potential MPI break here...
                if "-------------" in line:
                    break
                if "mpirun noticed" in line:
                    print("HEY")
                    break

        ii += 1

    if len(curr_results) != 0:
        print("Got to the end of the file, saving latest set of results")
        results = clean_up(curr_results, results)

    # now convert the results into a dictionary with their proper name

    results_new = split_results_to_dict(results, is_custom)

    print(results_new.keys())

    dump_results(results_new, output_file_dump + ".json")

    if skip_plotting:
        return

    from pprint import pprint

    # pprint(results)
    print()
    print("Found", len(results), "total program runs")

    plot_results(results_new, "U_E0_DIFF", "l2", True)
    plot_results(results_new, "U_E1_DIFF", "l2", True)
    plot_results(results_new, "U_E2_DIFF", "l2", True)

    plot_results(results_new, "U_B0_DIFF", "l2", True)
    plot_results(results_new, "U_B1_DIFF", "l2", True)
    plot_results(results_new, "U_B2_DIFF", "l2", True)

    plot_mesh_size(results_new, is_custom)

    plt.show()

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text_file",
        "-f",
        type=str,
        help="Text file that should be run",
        default="file.txt",
    )

    parser.add_argument(
        "--skip_plotting",
        "-s",
        action="store_true",
        help="Whether or not to skip plotting. This will just create the JSON files and can be used elsewhere",
    )
    parser.add_argument(
        "--is_custom",
        "-c",
        action="store_true",
        help="Set this flag if the output to be parsed is not from the generated runs",
    )

    args = parser.parse_args()

    main(args.text_file, args.skip_plotting, args.is_custom)
