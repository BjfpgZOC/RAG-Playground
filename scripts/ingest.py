from __future__ import annotations

import argparse
import uuid
from typing import List, Dict, Any

from qdrant_client import models as qm
from tqdm import tqdm

from rag.settings import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL_ID, CHUNK_TOKENIZER_ID
from rag.loaders.local_files import load_text_docs
from rag.chunking.simple_chunker import SimpleChunker
from rag.chunking.token_chunker import TokenChunker
from rag.embeddings.hf_embedder import HFEmbedder
from rag.vectorstore.qdrant_store import QdrantStore

def stable_point_id(doc_id: str, chunk_index: int) -> str:
    key = f"{doc_id}::chunk::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    
def main(args = None):
    docs = load_text_docs(args.input_dir)
    chunker = TokenChunker(chunk_tokens = args.chunk_tokens, overlap_tokens = args.overlap_tokens, tokenizer_id = CHUNK_TOKENIZER_ID)
    embedder = HFEmbedder(model_id = args.model_id, normalize = True)
    store = QdrantStore(url = args.qdrant_url, collection = args.collection)

    if args.reset_collection:
        store.recreate_collections(vector_size = embedder.dim())
    else:
        store.ensure_collections(vector_size = embedder.dim())

    print(f"Loaded {len(docs)} docs from: {args.input_dir}")
    print(f"Embedding dim: {embedder.dim()}")
    print(f"Qdrant: {args.qdrant_url} | Collection: {args.collection}")
    print()

    total_chunks = 0
    points_batch: List[qm.PointStruct] = []
    texts_batch: List[str] = []
    payloads_batch: List[Dict[str, Any]] = []

    for source_path, text in tqdm(docs, desc = "Ingesting documents"):
        source_path_norm = source_path.replace("\\", "/")
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_path_norm))
        store.delete_by_doc_id(doc_id)
        chunks = chunker.chunk_spans(text)
        
        for chunk_index, c in enumerate(chunks):
            total_chunks += 1
            chunk_text = c["text"]
            payload = {
                "doc_id": doc_id,
                "source": source_path_norm,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "token_start": c["token_start"],
                "token_end": c["token_end"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
            }

            texts_batch.append(c["text"])
            payloads_batch.append(payload)

            if len(texts_batch) >= args.batch_size:
                vectors = embedder.embed(texts_batch)

                for i in range(len(texts_batch)):
                    pid = stable_point_id(payloads_batch[i]["doc_id"], payloads_batch[i]["chunk_index"])

                    points_batch.append(
                        qm.PointStruct(
                            id = pid,
                            vector = vectors[i],
                            payload = payloads_batch[i],
                        )
                    )

                store.upsert_points(points_batch)

                points_batch.clear()
                texts_batch.clear()
                payloads_batch.clear()

    if texts_batch:
        vectors = embedder.embed(texts_batch)

        for i in range(len(texts_batch)):
            pid = stable_point_id(payloads_batch[i]["doc_id"], payloads_batch[i]["chunk_index"])

            points_batch.append(
                qm.PointStruct(
                    id = pid,
                    vector = vectors[i],
                    payload = payloads_batch[i],
                )
            )
        
        store.upsert_points(points_batch)

    print(f"\nDone. Total chunks processed: {total_chunks}")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type = str, default = "datasets/raw")
    parser.add_argument("--chunk_tokens", type = int, default = 300)
    parser.add_argument("--overlap_tokens", type = int, default = 50)
    parser.add_argument("--qdrant_url", type = str, default = QDRANT_URL)
    parser.add_argument("--collection", type = str, default = QDRANT_COLLECTION)
    parser.add_argument("--model_id", type = str, default = EMBED_MODEL_ID)
    parser.add_argument("--reset_collection", action = "store_true")
    parser.add_argument("--batch_size", type = int, default = 64)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    main(args)