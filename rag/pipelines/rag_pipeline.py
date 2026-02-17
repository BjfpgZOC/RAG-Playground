from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.retriever.dense_retriever import DenseRetriever
from rag.generation.prompt_builder import build_rag_prompt
from rag.generation.local_llm import LocalLLM
from rag.settings import LLM_MODEL_PATH

@dataclass
class RAGPipeline:
    top_k: int = 5
    min_score: Optional[float] = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 256

    def __post_init__(self) -> None:
        self.retriever = DenseRetriever()
        self.llm = LocalLLM(
            model_path = LLM_MODEL_PATH,
            n_ctx = self.n_ctx,
            n_gpu_layers = self.n_gpu_layers,
            verbose = False,
        )
    
    def answer(self, question: str) -> Dict[str, Any]:
        contexts = self.retriever.retrieve(
            query_text = question,
            top_k = self.top_k,
            min_score = self.min_score,
        )

        sys_prompt = build_rag_prompt(contexts = contexts)

        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]

        response = self.llm.generate(
            prompt = prompt,
            max_tokens = self.max_tokens,
            temperature = self.temperature, 
            top_p = self.top_p,
            stop = None,
        )

        return{
            "question": question,
            "context": contexts,
            "response": response,
        }

        
