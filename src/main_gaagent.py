from agents.GaAgent import GaAgent
from models.OpenAI import OpenAIModel
from models.Gemini import GeminiModel
from models.Claude import ClaudeModel
from dataloaders.TritonBench import TritonBench
from dataloaders.ROCm import ROCm
from args_config import load_config
import os

def main():
    args = load_config("configs/rocm_gaagent_config_new.yaml")
    args.log_root = os.path.abspath(args.output_path).replace(".jsonl", "")
    os.makedirs(args.log_root, exist_ok=True)

    # setup LLM model
    model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)

    # setup dataset
    dataset = ROCm(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path, 
                          py_interpreter=args.py_interpreter, 
                          log_root=args.log_root,
                          result_path=args.result_path)

    # setup agent
    agent = GaAgent(model=model, dataset=dataset, corpus_path=args.corpus_path, mem_file=args.mem_file, descendant_num=args.descendant_num)
    
    # run the agent
    agent.run(output_path=args.output_path, 
              multi_thread=args.multi_thread, 
              iteration_num=args.max_iteration, 
              temperature=args.temperature, 
              datalen=args.datalen,
              gpu_id=args.gpu_id,
              start_iter=args.start_iter,
              ancestor_num=args.ancestor_num,
              descendant_num=args.descendant_num,
              descendant_debug=args.descendant_debug,
              target_gpu=args.target_gpu,
              profiling=args.profiling,
              start_idx=args.start_idx)


if __name__ == "__main__":
    main()
