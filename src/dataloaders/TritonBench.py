import json
import os
import ast
import subprocess
from random import randint
from tqdm import tqdm
import signal
from multiprocessing import Pool, Lock, Value
from dataloaders.ProblemState import ProblemState
from dataloaders.TB_eval.utils import code_call_exec_success_allclose



class TritonBench:
    def __init__(self,
                 statis_path,
                 py_folder,
                 instruction_path,
                 golden_metrics,
                 py_interpreter,
                 perf_G_path,
                 perf_ref_folder=None,
                 result_path=None,
                 target_kernels=None
                 ):
        self.statis_path = statis_path
        self.py_folder = py_folder
        self.instruction_path = instruction_path
        self.golden_metrics_folder = golden_metrics
        self.py_interpreter = py_interpreter
        self.perf_ref_folder = perf_ref_folder
        self.perf_G_path = perf_G_path
        self.result_path = result_path

        self.problem_states = self.load_ps(result_path, target_kernels)
    
    def load_ps(self, path, target_kernels=None):
        problem_states = []
        if path is None:
            with open(self.instruction_path, "r", encoding='utf-8') as file:
                instructions = json.load(file)
            statis_data = json.loads(open(self.statis_path, 'r', encoding='utf-8').read())

            for line in instructions:
                instruction = line["instruction"]
                # label = line["output"]
                file = line["file"]
                label = None

                path = os.path.join(self.py_folder, file)
                assert os.path.exists(path), f"{file} not exist!"
                test_code = open(path, "r", encoding="utf-8").read().split("#"*146)[-1]
                assert "def test_" in  test_code, ""

                problemstate = ProblemState(instruction=instruction,
                                            label=label, 
                                            test_code=test_code, 
                                            filename=file, 
                                            )
                
                problem_states.append(
                    problemstate
                )
        else:
            with open(path, 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    content = json.loads(line)
                    problem_state = ProblemState(instruction=content["instruction"], 
                                                 label=content["label"], 
                                                 filename=content["filename"],
                                                )
                    if "test_code" in content:
                        problem_state.test_code = content["test_code"]
                    if "predict" in content:
                        problem_state.solution = content["predict"] 
                    problem_states.append(problem_state)
        return problem_states

    def __len__(self):
        return len(self.problem_states)
    
    def write_file(self, file_path, start_idx=0, datalen=None):
        """
        write the result into a jsonl or a json file, which can be evaluated by TB-eval
        """
        assert file_path.endswith(".jsonl") or file_path.endswith(".json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_len = datalen if datalen is not None else len(self)
        output_list = []
        with open(file_path, 'w') as f:
            for ps in self.problem_states[start_idx:(start_idx + data_len)]:
                output = {
                    "instruction": ps.instruction,
                    "label": ps.label,
                    "file": ps.filename,
                }
                if ps.test_code:
                    output["test_code"] = ps.test_code
                if ps.solution:
                    output["predict"] = ps.solution
                else:
                    output["predict"] = ""
                if file_path.endswith(".jsonl"):
                    f.write(json.dumps(output) + "\n")
                else:
                    output_list.append(output)
            if file_path.endswith(".json"):
                json.dump(output_list, f)
    
    @classmethod
    def run_single_call(cls, ps, tmp_dir="temp", gpu_id=0):
        """
        Runs the generated kernel to verify that it executes successfully without errors.
        """
        os.makedirs(tmp_dir, exist_ok=True)
        temp_path = os.path.join(tmp_dir, ps.filename)
        script_content = ps.solution
        try:
            with open(temp_path, "w") as temp_file:
                temp_file.write(script_content + "\n" + "#" * 146 + "\n" + ps.test_code)

            env = os.environ.copy()
            env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
            # Run the temporary Python file
            result = subprocess.run(
                ["python", temp_path], 
                capture_output=True, 
                text=True,
                env=env
            )

            success = result.returncode == 0  # Determine if execution was successful

            if success:
                ps.pass_call = True
                return True, None
            else:
                return False, result.stderr

        except Exception as e:
            return False, str(e)
    
    
    def write_perf_file(self, input_folder_path, results_path, tmp_dir):
        """
        Preparation step for latency measurement.
        This method collects generated kernels that passed both call and execution tests, 
        appends performance testing code, and stores the resulting scripts in tmp_dir.

        Args:
            input_folder_path: directory where codes that pass call and exe tests are stored
            results_path: directory where perf results (json files) are stored
            tmp_dir: Temporary directory for storing performance test scripts. This folder needs to be empty before use.
        """
        
        os.makedirs(tmp_dir, exist_ok=True)
        if os.path.exists(results_path):
            os.system(f'rm -rf {results_path}')
        os.mkdir(results_path)

        tab = ' ' * 4
        performance_utils_path = os.path.join(self.perf_G_path, "performance_utils.py")
        with open(performance_utils_path, 'r') as f:
            performance_utils = f.readlines()
        # performance_utils = performance_utils.replace('folder_path = "/home/lishangzhan/triton/bench_performance/results"', f'folder_path = "{results_path}"')
        performance_utils_lines = []
        for line in performance_utils:
            if 'folder_path = ' in line:
                line = tab * 2 + f'folder_path = "{results_path}"\n'
            performance_utils_lines.append(line)
        performance_utils = "".join(performance_utils_lines)
        with open(performance_utils_path, 'w') as f:
            f.write(performance_utils)
        input_file_list = os.listdir(input_folder_path)
        golden_metrics_list = os.listdir(self.golden_metrics_folder)
        for file in input_file_list:
            if file[-3:] == ".py":
                op = file[:-3]
                perf_file_name = op + "_perf.py"
                assert perf_file_name in golden_metrics_list, f"{perf_file_name} not in golden_metrics_list"
                with open(os.path.join(self.golden_metrics_folder, perf_file_name), "r") as f:
                    # golden_metrics = f.read()
                    lines = f.readlines()
                    # print(lines)
                    updated_lines = []
                    for line in lines:
                        if line == "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n":
                            updated_lines.append(f"sys.path.append('{input_folder_path}')\n")
                            updated_lines.append(f"sys.path.append('{self.perf_G_path}')\n")
                        line = line.replace("from TritonBench_v1.", "from ")
                        line = line.replace("op_perf.get_do_bench_config()", "op_perf.get_do_bench_config(warmup=100, rep=1000)")
                        line = line.replace('folder_path = "/home/lishangzhan/triton/bench_performance/results"', f'folder_path = "{results_path}"')
                        updated_lines.append(line)
                    golden_metrics = "".join(updated_lines)
                
                golden_metrics_lines = golden_metrics.split("\n")
                flag = False
                for i in range(len(golden_metrics_lines)):
                    if "input_tensor = self.to_cuda(input_tensor_)" in golden_metrics_lines[i]:
                        index_1 = i
                    if "results.append(result)" in golden_metrics_lines[i]:
                        index_2 = i + 1
                        flag = True
                if flag:
                    
                    for i in range(index_1, index_2):
                        golden_metrics_lines[i] = tab + golden_metrics_lines[i]                
                    golden_metrics_lines.insert(index_1, tab*3 + "try:")
                    golden_metrics_lines.insert(index_2 + 1, tab*3 + "except Exception as e:")
                    golden_metrics_lines.insert(index_2 + 2, tab*4 + 'print(f"Failed to run benchmark for input tensor. Error: {e}")')
                    golden_metrics = "\n".join(golden_metrics_lines)
                
                with open(os.path.join(tmp_dir, perf_file_name), "w") as f:
                    f.write(golden_metrics)
    
    def run_perf_scripts(self, script_dir = "./tmp", log_dir = "./logs", gpu_id=0):
        """
        Runs a given Python script on a specified GPU.
        """
        os.makedirs(log_dir, exist_ok=True)

        scripts = sorted([f for f in os.listdir(script_dir) if f.endswith(".py")])
        scripts = [os.path.join(script_dir, script) for script in scripts]
        total_scripts = len(scripts)
        timeout_sec = 600  # 10 mins
        with tqdm(total=total_scripts) as pbar:
            for idx, script in enumerate(scripts):
                env = os.environ.copy()
                env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
                try:
                    result = subprocess.run(
                            [self.py_interpreter, script], 
                            capture_output=True, 
                            text=True,
                            env=env,
                            timeout=timeout_sec
                        )
                except subprocess.TimeoutExpired:
                    print(f"The subprocess timed out.")
                except Exception as e:
                    print(f"The subprocess failed due to {e}")
                # finally:
                #     pass
                tqdm.write(f"✅ finished {idx+1}/{total_scripts}: {os.path.basename(script)}")
                pbar.update(1)



    def calculate(self, path_gen, path_ref=None):
        get_ms = lambda data: [item["ms"] for item in data]
        get_gbs = lambda data: [item["GB/s"] for item in data]
        get_tflops = lambda data: [item["TFLOPS"] for item in data]
        avg = lambda mss: round(sum(mss[0]) / sum(mss[1]), 4)

        data_gen = json.loads(open(path_gen, 'r', encoding='utf-8').read())
        if path_ref is not None:
            data_ref = json.loads(open(path_ref, 'r', encoding='utf-8').read())
            assert len(data_gen) == len(data_ref), ""
            ms_ref = get_ms(data_ref)

        ms_gen = get_ms(data_gen)

        spdup = avg((ms_ref, ms_gen)) if path_ref is not None else None
        
        # TODO: there is problem with efficiency calculation
        efficiency = max(round(max(get_gbs(data_gen)) * 100 / 2039, 4), round(max(get_tflops(data_gen)) * 100 / 312, 4))
        
        return spdup, efficiency, round(sum(ms_gen)/len(ms_gen), 4)
    
    def test_opt_correctness(self, code, filename, tmp_dir, save_scripts=True, exe_dir="pass_exe"):
        """
        Runs the given Python script on the specified GPU to verify both the executability and correctness of the generated kernel.
        Args:
            code: generated code
            filename: filename of the kernel, e.g `cosine_compute.py`
            tmp_dir: folder path to store the script.
            save_script: whether write the correct script in exe_dir. If you want to calculate latency, this need to be set to True
            exe_dir: folder path to store script for correct kernel
        """
        os.makedirs(exe_dir, exist_ok=True)
        call_status, exec_status, call_stdout, call_stderr, exe_stdout, exe_stderr = code_call_exec_success_allclose(code=code, fname=filename, temp_root=tmp_dir, py_folder=self.py_folder)
        pass_call = False
        pass_exe = False
        if "True" in str(call_status):
            pass_call=True
        if "True" in str(exec_status):
            pass_exe=True
            if save_scripts:
                file_exec = os.path.join(exe_dir, filename)
                with open(file_exec, 'w') as f:
                    f.write(code)

        return pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr
    
    
    