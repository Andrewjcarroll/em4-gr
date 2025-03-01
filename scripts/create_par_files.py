"""create_par_files.py

This file is a helper that takes a base set of parameters and can
generate many more for testing different combinations of parameters.

Simply set what should be explored, and it will create the TOML files
to do so.
"""

import argparse
import os
from pathlib import Path
from pprint import pprint

import toml

# PARAMETERS TO MODIFY:
# empty means to leave them alone!
SOLVER_DERIV_TYPES = [-1, 0]
SOLVER_2ND_DERIV_TYPES = [None]
SOLVER_FILTER_TYPES = [-1, 0, 1]
SOLVER_DERIV_CLOSURE_TYPES = [None]

SOLVER_BATCH_SCRIPT_HEADER = """#!/bin/bash

#SBATCH --time={time}
#SBATCH --ntasks={tasks}
#SBATCH --nodes={nodes}
#SBATCH -A {account}
#SBATCH -p {partition}
#

print_divider_and_name() {{
    echo ""
    echo ""
    echo "==============================="
    echo "Now beginning task for $1!"
    echo "==============================="
    echo ""
    echo ""
}}


"""


SOLVER_INITIALIZATION_START = """
print_divider_and_name "{outfile}"
mpirun -np $SLURM_NTASKS --output-filename {outfile} ./em3Solver {paramfile} 1

"""


def get_true_stem(file: Path):
    curr_stem = Path(file).stem
    return curr_stem if curr_stem == file else get_true_stem(curr_stem)


def main(
    config_file,
    output_folder,
    slurm_tasks,
    slurm_nodes,
    slurm_time,
    slurm_account,
    slurm_partition,
):
    # read in the file
    config_file = Path(config_file)
    config_file_base = get_true_stem(config_file)

    output_folder = Path(output_folder)
    if output_folder.is_file():
        raise Exception("Output folder cannot already be an existing FILE!")
    # make sure we create the output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_toml = toml.load(config_file)

    vtu_output = Path(raw_toml["dsolve::SOLVER_VTU_FILE_PREFIX"])
    vtu_folder = vtu_output.parent
    vtu_filename = vtu_output.name

    profile_output = Path(raw_toml["dsolve::SOLVER_PROFILE_FILE_PREFIX"])
    profile_folder = profile_output.parent
    profile_filename = profile_output.name

    # generate all combinations! So many!
    num_deriv_types = len(SOLVER_DERIV_TYPES)
    num_2nd_deriv_types = len(SOLVER_2ND_DERIV_TYPES)
    num_filter_types = len(SOLVER_FILTER_TYPES)
    num_deriv_closure_types = len(SOLVER_DERIV_CLOSURE_TYPES)
    total_parameters = (
        num_deriv_types
        * num_2nd_deriv_types
        * num_filter_types
        * num_deriv_closure_types
    )

    print(
        "There are",
        total_parameters,
        "different combinations of parameters",
    )

    original_deriv_type = raw_toml["dsolve::SOLVER_DERIV_TYPE"]
    original_sec_deriv_type = raw_toml["dsolve::SOLVER_2ND_DERIV_TYPE"]
    original_filter_type = raw_toml["dsolve::SOLVER_FILTER_TYPE"]
    original_closure_type = raw_toml["dsolve::SOLVER_DERIV_CLOSURE_TYPE"]

    bash_script_addition = SOLVER_BATCH_SCRIPT_HEADER.format(
        tasks=slurm_tasks,
        nodes=slurm_nodes,
        time=slurm_time,
        account=slurm_account,
        partition=slurm_partition,
    )

    total_files = 0

    for ii, deriv_type in enumerate(SOLVER_DERIV_TYPES):
        for jj, sec_deriv_type in enumerate(SOLVER_2ND_DERIV_TYPES):
            for kk, filter_type in enumerate(SOLVER_FILTER_TYPES):
                for ll, deriv_closure_type in enumerate(SOLVER_DERIV_CLOSURE_TYPES):
                    # closures do not affect explicit derivatives nor kim derivatives
                    if deriv_type in [-1, 3, 6, 7, 8] and ll > 0:
                        print(
                            "    - Skipping a repeat of derivative",
                            deriv_type,
                            "since there are multiple closures   ( closure is currently",
                            deriv_closure_type,
                            ")",
                        )
                        continue
                    if sec_deriv_type == [-1, 6, 7, 8] and ll > 0:
                        print(
                            "    - Skipping a repeat of 2nd derivative",
                            sec_deriv_type,
                            "since there are multiple closures   ( closure is currently",
                            deriv_closure_type,
                            ")",
                        )
                        continue

                    filename_addition = f"deriv{ii}_2deriv{jj}_filter{kk}_closure{ll}"

                    if deriv_type is not None:
                        raw_toml["dsolve::SOLVER_DERIV_TYPE"] = deriv_type
                    if sec_deriv_type is not None:
                        raw_toml["dsolve::SOLVER_2ND_DERIV_TYPE"] = sec_deriv_type
                    if filter_type is not None:
                        raw_toml["dsolve::SOLVER_FILTER_TYPE"] = filter_type
                    if deriv_closure_type is not None:
                        raw_toml[
                            "dsolve::SOLVER_DERIV_CLOSURE_TYPE"
                        ] = deriv_closure_type

                    new_vtu_output = vtu_folder / (
                        vtu_filename + "_" + filename_addition
                    )

                    new_profile_output = profile_folder / (
                        profile_filename + "_" + filename_addition
                    )

                    raw_toml["dsolve::SOLVER_VTU_FILE_PREFIX"] = str(new_vtu_output)
                    raw_toml["dsolve::SOLVER_PROFILE_FILE_PREFIX"] = str(
                        new_profile_output
                    )

                    new_param_filename = config_file_base + "_" + filename_addition
                    new_param_filename_extension = new_param_filename + ".toml"

                    # save the file to toml

                    bash_script_addition += (
                        SOLVER_INITIALIZATION_START.format(
                            outfile="out_" + new_param_filename,
                            paramfile=new_param_filename_extension,
                        )
                        + "\n"
                    )

                    # save the toml file
                    with open(output_folder / new_param_filename_extension, "w") as f:
                        toml.dump(raw_toml, f)

                    total_files += 1

    # then we can just save the string to our bash
    with open(output_folder / (config_file_base + "_submission.sh"), "w") as f:
        f.write(bash_script_addition)

    print("Script generated", total_files, "parameter combinations to run!")

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="Base configuration file for the solver's run (i.e. em3_simplified.param.toml)",
        default="config_aesz_model.toml",
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        help="Where to store the generated parameter files and bash script",
        default="temp_pars",
    )

    parser.add_argument(
        "--slurm_tasks",
        "-t",
        type=int,
        help="The number of tasks to fill into the generated batch script ",
        default=56,
    )
    parser.add_argument(
        "--slurm_nodes",
        "-n",
        type=int,
        help="The number of nodes to request via slurm",
        default=2,
    )
    parser.add_argument(
        "--slurm_time",
        "-s",
        type=str,
        help="The amount of wall time for the slurm job",
        default="08:00:00",
    )
    parser.add_argument(
        "--slurm_account",
        "-A",
        type=str,
        help="The slurm account '-A' flag",
        default="soc-kp",
    )

    parser.add_argument(
        "--slurm_partition",
        "-p",
        type=str,
        help="The slurm partition '-p' flag",
        default="soc-kp",
    )

    args = parser.parse_args()

    main(
        args.config_file,
        args.output_folder,
        args.slurm_tasks,
        args.slurm_nodes,
        args.slurm_time,
        args.slurm_account,
        args.slurm_partition,
    )
