# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

import os
from glob import glob

root = "/shared-aig/vinay/openevolve/examples/tb/initial_programs/programs"

output = "/shared-aig/vinay/openevolve/examples/tb/initial_programs/rocm"

if not os.path.exists(output):
    os.makedirs(output, exist_ok=True)

files = glob(os.path.join(root, "*.py"))
assert len(files) > 0, f"No files found in the specified directory {root}."

checker = "#"*146

for file in files:
    with open(file, 'r') as f:
        content = f.readlines()
        print(f"number of lines in {file}: {len(content)}")
        for i, line in enumerate(content):
            if checker == line.strip():
                print(f"Found the checker in {file} at line {i+1}.")
                break

        new_file = os.path.join(output, os.path.basename(file))
        with open(new_file, 'w') as nf:
            new_lines = "".join(content[:i])
            nf.write(new_lines)
        print(f"Created new file {new_file} with content up to the checker line.")
