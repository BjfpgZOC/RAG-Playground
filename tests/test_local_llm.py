from __future__ import annotations

from rag.settings import LLM_MODEL_PATH
from rag.generation.local_llm import LocalLLM

def main():
    llm = LocalLLM(
        model_path = LLM_MODEL_PATH,
        n_ctx = 4096,
        n_gpu_layers = -1,
        verbose = False
    )

    
    text = llm.generate(
        prompt = prompt,
        max_tokens = 80,
        temperature = 0.2,
        top_p = 0.9,
        stop = None,
    )

    print("\nPROMPT:\n", prompt[1]["content"])
    print("\nRESPONSE:\n", text)

if __name__ == "__main__":
    main()