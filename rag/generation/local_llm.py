from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from llama_cpp import Llama
from rag.settings import LLM_N_CTX, LLM_N_THREADS, LLM_N_GPU_LAYERS, LLM_TEMPERATURE, LLM_MAX_TOKENS

@dataclass
class LocalLLM:
    model_path: str
    n_ctx: int = LLM_N_CTX
    n_gpu_layers: int = LLM_N_GPU_LAYERS
    n_threads: int = LLM_N_THREADS
    verbose: bool = False

    def __post_init__(self) -> None:
        self.llm = Llama(
            model_path = self.model_path,
            n_ctx = self.n_ctx,
            n_gpu_layers = self.n_gpu_layers,
            n_threads = self.n_threads,
            verbose = self.verbose
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        
        out = self.llm.create_chat_completion(
            messages = prompt,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p = top_p,
            stop = stop,
        )

        return out["choices"][0]["message"]["content"]
    
