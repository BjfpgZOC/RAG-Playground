import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "models/embeddings/bge-small-en-v1.5")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/llm/Qwen3-4B-Instruct-2507.Q4_K_M.gguf")
CHUNK_TOKENIZER_ID = os.getenv("CHUNK_TOKENIZER_ID", EMBED_MODEL_ID)

LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "8"))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
