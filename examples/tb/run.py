# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

import os
from glob import glob
import time
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run OpenEvolve initial programs with ROCm evaluator.")
parser.add_argument("--config", type=str, default="configs/claude.yaml", help="Path to the configuration file.")
parser.add_argument("--output", type=str, default="runs/tb", help="Output directory for the runs.")
parser.add_argument("--initial_programs", type=str, default="initial_programs/tb_kernels/*.py", help="Glob pattern for initial programs.")
parser.add_argument("--evaluator", type=str, required=True, help="Path to the ROCm evaluator script.")
parser.add_argument("--sort_by", type=str, default=None, help="JSON File containing filename and values to sort by in ascending order.")
parser.add_argument("--timeout", type=int, default=6*3600, help="Timeout for each run in seconds (default: 6 hours).")
args = parser.parse_args()

def get_time():
    ## get time formatted as YYYY-MM-DD HH:MM:SS
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

initial_programs = glob(args.initial_programs)

REMOVE_FILES = ["__init__.py", "cleanup.py"]

cleaned_initial_programs = []
for program in initial_programs:
    if os.path.basename(program) not in REMOVE_FILES:
        cleaned_initial_programs.append(program)

initial_programs = cleaned_initial_programs

print(f"{get_time()} Found {len(initial_programs)} initial programs.")

if args.sort_by:
    import json
    with open(args.sort_by, 'r') as f:
        sort_data = json.load(f)
    initial_programs = sorted(initial_programs, key=lambda x: sort_data.get(os.path.basename(x), float('inf')))
    print(f"{get_time()} Sorted initial programs by {args.sort_by}.")
    print(f"{get_time()} Sorted initial programs: {initial_programs}")
else:
    initial_programs = sorted(initial_programs)

for p in initial_programs:
    print(f"{get_time()} Initial program: {p}")
assert len(initial_programs) > 0, f"No initial programs found in the specified directory {initial_programs}."
LAST_INTERRUPT_TIME = time.time()
for initial_program in initial_programs:
    try:
        output = os.path.join(args.output, os.path.basename(initial_program).replace(".py", ""))
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
        command = f"openevolve-run {initial_program} {args.evaluator} --config {args.config} --output {output} > {output}/run.log 2> {output}/run.log"
        print(f"{get_time()} Running command: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True, timeout=args.timeout)
        if result.returncode != 0:
            print(f"{get_time()} Error running {initial_program}: {result.stderr}")
        else:
            print(f"{get_time()} Successfully ran {initial_program}. Output saved to {output}/run.log")
        print("-" * 80)
    except subprocess.TimeoutExpired:
        print(f"{get_time()} Timeout expired for {initial_program}. Skipping this run.")
    except KeyboardInterrupt:
        # print(f"{get_time()} Process interrupted by user. Exiting.")
        ## if time from last interrupt is less than 2 seconds, exit immediately
        if time.time() - LAST_INTERRUPT_TIME < 2:
            print(f"{get_time()} Exiting immediately due to rapid interrupt.")
            break
        LAST_INTERRUPT_TIME = time.time()
        print(f"{get_time()} Continuing to next initial program after interrupt.")
        continue
    except Exception as e:
        print(f"{get_time()} An error occurred while running {initial_program}: {e}")
        continue
    time.sleep(60)
