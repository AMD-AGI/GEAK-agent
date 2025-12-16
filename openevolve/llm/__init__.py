# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
LLM module initialization
"""

from openevolve.llm.base import LLMInterface
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.llm.openai import OpenAILLM
from openevolve.llm.sampling import ThompsonSampling, GaussianThompsonSampling, RandomSampling, get_sampling_function
__all__ = ["LLMInterface", "OpenAILLM", "LLMEnsemble", 
           "ThompsonSampling", "GaussianThompsonSampling", "get_sampling_function", "RandomSampling"]
