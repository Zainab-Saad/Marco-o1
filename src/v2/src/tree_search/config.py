from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field, model_validator

class ModelConfig(BaseModel):
    provider: str
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    # these fields needed for vllm
    dtype: str = "bfloat16"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.90
    
    @model_validator(mode="after")
    def resolve_api_key(self) -> "ModelConfig":
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == "openrouter":
                self.api_key = os.environ.get("OPENROUTER_API_KEY")
                if self.base_url is None:
                    self.base_url = "https://openrouter.ai/api/v1"
            elif self.provider == "huggingface":
                self.api_key = os.environ.get("HUGGING_FACE_TOKEN")
        return self

class DataConfig(BaseModel):
    dataset: str = "gsm8k"
    split: str = "train"
    num_samples: Optional[int] = None
    seed: int = 42
    levels: Optional[list[int | str]] = [5] #L5 math or all math

    
class Config(BaseModel):
    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)