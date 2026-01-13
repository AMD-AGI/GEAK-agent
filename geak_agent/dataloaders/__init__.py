# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

# Lazy imports to avoid loading tb_eval dependency at import time
__all__ = [
    "ProblemState",
    "ProblemStateROCm",
    "tempCode",
    "TritonBench",
    "ROCm",
]


def __getattr__(name):
    """Lazy import to avoid loading tb_eval at package import time."""
    if name in ("ProblemState", "ProblemStateROCm", "tempCode"):
        from geak_agent.dataloaders.ProblemState import ProblemState, ProblemStateROCm, tempCode
        if name == "ProblemState":
            return ProblemState
        elif name == "ProblemStateROCm":
            return ProblemStateROCm
        else:
            return tempCode
    elif name == "TritonBench":
        from geak_agent.dataloaders.TritonBench import TritonBench
        return TritonBench
    elif name == "ROCm":
        from geak_agent.dataloaders.ROCm import ROCm
        return ROCm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
