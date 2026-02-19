# RAG Playground

This repository contains practice projects for **RAG (Retrieval-Augmented Generation)**.

It provides a minimal RAG chatbot that:
- Loads local documents (PDF / TXT / MD)
- Token-chunks them
- Embeds chunks using Sentence-Transformers
- Stores vectors in Qdrant
- Retrieves top-k chunks for a question
- Answers with a local GGUF LLM via `llama-cpp-python`

## Requirements

- Docker Desktop
- (Optional GPU) NVIDIA GPU + Drivers + Docker GPU support

## Setup

### 1) Add your documents
Put files into:
datasets/raw/


Example:
datasets/raw/document.pdf


### 2) Add your GGUF model
Put a GGUF file into:
models/llm/


Example:
models/llm/Qwen3-4B-Instruct-2507.Q4_K_M.gguf

### 3) Add your embedding model (from Hugging Face)
Download the embedding model and place it under:
models/embeddings/

Example:
models/embeddings/bge-small-en-v1.5/

You can download with:
```bash
hf download BAAI/bge-small-en-v1.5 --local-dir models/embeddings/bge-small-en-v1.5
```

### 4) Configure environment in `deploy/compose.yaml`
Edit these:
- `LLM_MODEL_PATH` (your GGUF path)
- `EMBED_MODEL_ID` (local embedding model folder)

Example:
```yaml
environment:
  - QDRANT_URL=http://qdrant:6333
  - QDRANT_COLLECTION=docs
  - EMBED_MODEL_ID=models/embeddings/bge-small-en-v1.5
  - LLM_MODEL_PATH=models/llm/Qwen3-4B-Instruct-2507.Q4_K_M.gguf
```

## Run (Docker)

### 1) Start services
From repo root:

```bash
docker compose -f deploy/compose.yaml up -d --build
```

Health check:

```bash
curl http://localhost:8000/health
```

### 2) Ingest documents
Run ingestion using the wrapper script:

```bash
docker compose -f deploy/compose.yaml exec api python scripts/ingest.py --input_dir datasets/raw
```

Useful knobs:

```bash
docker compose -f deploy/compose.yaml exec api python scripts/ingest.py \\
  --input_dir datasets/raw \
  --chunk_tokens 300 \
  --overlap_tokens 50 \
  --batch_size 64
```

### 3) Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Summarize what my documents are about.\",\"top_k\":5}"
```

Debug (see retrieved contexts):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is this PDF about?\",\"top_k\":5,\"return_contexts\":true}"
```

## Chat CLI
Run the chat CLI inside the API container:

```bash
docker compose -f deploy/compose.yaml exec api python scripts/rag_chat_cli.py
```

## Refresh behavior (re-ingest safely)
Re-run ingestion after changing `datasets/raw/`.
The ingestion pipeline deletes existing points for the same `doc_id` before inserting new chunks, so re-ingest updates content cleanly.

```bash
docker compose -f deploy/compose.yaml exec api python scripts/ingest.py --input_dir datasets/raw
```

## API Endpoints
`GET /health`

`POST /retrieve`

Body:

```json
{ "query": "string", "top_k": 5, "min_score": null }
```

`POST /chat`

Body:

```json
{ "question": "string", "top_k": 5, "min_score": null, "return_contexts": false }
```

## Optional GPU checks

1) Verify Docker can see the GPU

```bash
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

2) Verify the API container can see the GPU

```bash
docker compose -f deploy/compose.yaml exec api nvidia-smi
```

3) Verify `llama-cpp-python` supports GPU offload

```bash
docker compose -f deploy/compose.yaml exec api python -c \\
"from llama_cpp import llama_cpp; print('supports_gpu_offload =', bool(llama_cpp.llama_supports_gpu_offload()))"
```
