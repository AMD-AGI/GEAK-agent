"""
GEAK Agent for HIP - Public API Version
Uses PublicLLMModel for direct Claude/OpenAI API access
"""

from agents.GaAgent_HIP import GaAgent_HIP
from models.PublicLLM import PublicLLMModel
from dataloaders.HIPBench import HIPBench
from args_config import load_config


def main():
    args = load_config("configs/hipbench_gaagent_config.yaml")

    # Auto-detect provider from model_id
    if 'claude' in args.model_id.lower():
        provider = 'claude'
    elif 'gpt' in args.model_id.lower():
        provider = 'openai'
    else:
        provider = None  # Auto-detect will handle it

    # Setup LLM model using public APIs
    model = PublicLLMModel(
        api_key=args.api_key,
        model_id=args.model_id,
        provider=provider
    )

    print(f"✓ Using {model.provider} with model {model.model_id}")

    dataset = HIPBench(
        instruction_path=args.instruction_path,
        result_path=args.result_path
    )

    # Setup agent
    agent = GaAgent_HIP(model=model, dataset=dataset, corpus_path=args.corpus_path)

    # Run the agent
    agent.run(
        output_path=args.output_path,
        multi_thread=args.multi_thread,
        iteration_num=args.max_iteration,
        temperature=args.temperature,
        datalen=None,
        ancestor_num=args.ancestor_num,
        descendant_num=args.descendant_num
    )


if __name__ == "__main__":
    main()
