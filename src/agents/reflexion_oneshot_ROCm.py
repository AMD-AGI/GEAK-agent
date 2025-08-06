import os
from tqdm import tqdm
from loguru import logger
import json
from dataclasses import asdict
from agents.Reflexion import Reflexion
from utils.utils import extract_function_signatures, clear_code, extract_function_calls
from prompts import prompt_for_reflection
from memories.Memory import MemoryClassMeta
from models.Base import BaseModel
from retrievers.retriever import BM25Retriever
from prompts import prompt_for_generation
from concurrent.futures import ThreadPoolExecutor, as_completed



class Reflexion_Oneshot(Reflexion):
    """
    1. Given the instructions and contexts, the LLM generates candidate solutions.
    2. The generated solutions are executed to produce results or, in case of failure, error messages.
    3. If execution fails, the LLM analyzes the error trace and suggests a fix.
    4. The process repeats, with the LLM generating improved solutions based on previous feedback.
    """

    def __init__(self, model: BaseModel, dataset, corpus_path, mem_file=None):
        self.model = model
        self.dataset = dataset
        self.memories = []

        self.instruction_retriever = BM25Retriever()
        self.instruction_retriever.process(content_input_path=corpus_path)
        self.code_retriever = BM25Retriever(mode="code")
        self.code_retriever.process(content_input_path=corpus_path)

        self.memory_init(mem_file)

    def memory_init(self, mem_file=None):
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "err_msg", 
                                                             "reflection", 
                                                             "function_signatures", 
                                                             "oneshot",
                                                             "pass_exe", 
                                                            ]):
            pass
        
        if mem_file is not None:
            assert mem_file.endswith(".json"), f"expect a json file, but got {mem_file} instead"
            with open(mem_file, "r") as f:
                input_mems = json.load(f)
            assert len(input_mems) == len(self.dataset), f"expect {len(self.dataset)} samples, but got {len(input_mems)} instead"

        for ps in self.dataset.problem_states:
            if ps.label:
                fs_mem = extract_function_signatures(ps.label)
            else:
                fs_mem = None
            if mem_file is None:
                os_mem = self.instruction_retriever.query(ps.instruction)[0]
                tmp_mem = Memory(ps=ps, 
                                err_msg=None, 
                                reflection=None, 
                                function_signatures=fs_mem, 
                                oneshot=os_mem["code"],
                                pass_exe=False,
                                )
            else:
                input_mem = input_mems[ps.filename]
                tmp_mem = Memory(ps=ps, 
                                err_msg=input_mem["err_msg"], 
                                reflection=input_mem["reflection"], 
                                function_signatures=fs_mem, 
                                oneshot=input_mem["oneshot"],
                                pass_exe=input_mem["pass_exe"],
                                )
            self.memories.append(tmp_mem)

    def run(self, output_path=None, multi_thread=True, verbose=False, datalen=None, iteration_num=0, temperature=0):
        data_len = datalen if datalen else len(self.dataset)
        for iter in range(iteration_num):
            logger.info(f"\n=== Iteration {iter} ===")
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                iter_path = f"{root}_{iter}{extension}"

            if multi_thread:
                thread_num = 3
            
            # generate solution
            logger.info(f"\ngenerate solution")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_solution, mem, temperature): mem for mem in self.memories[:data_len]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[:data_len]:
                        self.generate_solution(mem, temperature=temperature)
                        pbar.update(1)
            
            """
            Run the scripts to verify whether the generated kernels are correct.
            """
            logger.info(f"\nrun scripts on gpu")
            if output_path is None or (hasattr(self.dataset, 'rocm_tests') and self.dataset.rocm_tests):
                tmp_dir = "tmp"
                exe_dir = "pass_exe"
                perf_result_dir = "perf_results"
                
            else:
                root, extension = os.path.splitext(output_path)
                tmp_dir = f"{root}_tmp"
                exe_dir = f"{root}_pass_exe"
                perf_result_dir = f"{root}_perf_results"
            
            for mem in tqdm(self.memories[:data_len]):
                if mem.pass_exe:
                    continue
                try:
                    pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr = self.dataset.test_opt_correctness(mem.ps.solution, mem.ps.filename, tmp_dir, exe_dir=exe_dir)
                
                except Exception as e:
                    logger.info(f"failed to test the code due to : {e}")
                    mem.err_msg = f"failed to test the code due to: {e}"
                    continue
                if not pass_call:
                    mem.err_msg = call_stderr
                elif not pass_exe:
                    mem.err_msg = exe_stderr
                else:
                    mem.pass_exe = True
            """
            To measure kernel speedup, follow these steps:
            
            root, extension = os.path.splitext(output_path)
            exe_dir = os.path.join(root,exe_dir) if output_path else exe_dir
            perf_result_dir = os.path.join(root, perf_result_dir) if output_path else perf_result_dir

            if not os.listdir(exe_dir):
                pass
                # logger.warning(f"No scripts passed correctness checks in iteration {iter}. Skipping performance evaluation.")
            else:
                # run performance evaluation
                # This block now only runs if there are files to evaluate.
                
                perf_results_dict = {}

                perf_results_dict = self.dataset.run_perf_evaluation(
                    exec_folder=exe_dir, 
                    gen_perf_folder=perf_result_dir
                )
                perf_results_list = list(perf_results_dict.values())
                # Create a list of memory objects that passed the correctness check
                passed_mems = [mem for mem in self.memories[:data_len] if mem.pass_exe]
                assert len(passed_mems) == len(perf_results_list)
                for mem, perf_data in zip(passed_mems, perf_results_list):
                    speedup = perf_data.get("ms")
                    efficiency = perf_data.get("efficiency")
            """     
            

            # generate reflections
            logger.info(f"\ngenerate reflections")
            with tqdm(total=data_len) as pbar:
                if multi_thread:
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_reflexion, mem, temperature): mem for mem in self.memories[:data_len]}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in self.memories[:data_len]:
                        self.generate_reflexion(mem, temperature=temperature)
                        pbar.update(1)
            
            if output_path is not None:
                self.dataset.write_file(iter_path)
                    

    
    def generate_solution(self, mem, temperature=0):
        if mem.pass_exe:
            return
        
        tab = "\n"
        fss_text = "".join(f"* {sig}{tab}" for sig in mem.function_signatures)
        text = prompt_for_generation.prompt.format(
            instruction=mem.ps.instruction,
            function_signatures=fss_text
        )

        if not mem.ps.solution:
            text += f"\nHere is an example snippet of code: {mem.oneshot}"
        else:
            one_shot = self.code_retriever.query(mem.ps.solution)[0]["code"]
            text += f"\nHere is an example snippet of code: {one_shot}"
            text += f"\nPrevious attempt implementation:{mem.ps.solution}"
            
                  
        if mem.err_msg:
            text += f"\nTest messages for previous attempt:{mem.err_msg}"
        
        if mem.reflection:
            text += f"\nReflection on previous attempt:{mem.reflection}"

        text += "Please output the codes only without explanation, which we can run directly."
        msg = [
            {"role": "user", "content": text},
        ]
        response = self.model.generate(msg, temperature=temperature)
        mem.ps.solution = clear_code(response)

        return



    def generate_reflexion(self, mem, temperature):
        if mem.pass_exe:
            return
        reflect_txt = prompt_for_reflection.prompt.format(
            problem=mem.ps.instruction,
            solution=mem.ps.solution,
            test_result=mem.err_msg
        )
        reflect_msg = [
            {
                "role": "user",
                "content": reflect_txt
            }
        ]
        mem.reflection = self.model.generate(reflect_msg, temperature=temperature)