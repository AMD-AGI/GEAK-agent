
import os
from agents.reflexion_oneshot_ROCm import Reflexion_Oneshot
from models.OpenAI import OpenAIModel
from dataloaders.ROCm import ROCm
from args_config import load_config


def main():
    args = load_config("configs/rocm_oneshot_config.yaml")
    log_root, _ = os.path.splitext(args.output_path)
    args.log_root = log_root

    # setup LLM model
    model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)
    # setup dataset
    dataset = ROCm(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path, 
                          py_interpreter=args.py_interpreter,
                          log_root=args.log_root,
                          target_kernels=args.target_kernels)

    # setup agent
    agent = Reflexion_Oneshot(model=model, dataset=dataset, corpus_path=args.corpus_path)

    # run the agent
    agent.run(output_path=args.output_path, multi_thread=args.multi_thread, iteration_num=args.max_iteration, temperature=args.temperature, datalen=args.datalen)


if __name__ == "__main__":
    main()