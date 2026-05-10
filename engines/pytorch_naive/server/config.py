"""Environment-driven configuration for E3.

All knobs are read from environment variables with sensible defaults so the
container can be reconfigured by the launcher without rebuilding the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model_name: str
    max_model_len: int
    dtype: str
    device: str
    temperature: float
    top_p: float

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
            dtype=os.environ.get("DTYPE", "bfloat16"),
            device=os.environ.get("GPU_DEVICE", "cuda:0"),
            temperature=float(os.environ.get("TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("TOP_P", "0.95")),
        )
