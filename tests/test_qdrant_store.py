from __future__ import annotations

import uuid

from qdrant_client import models as qm

from rag.settings import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL_ID
from rag.loaders.local_files import load_text_docs
from rag.chunking.simple_chunker import SimpleChunker
from rag.embeddings.hf_embedder import HFEmbedder
from rag.vectorstore.qdrant_store import QdrantStore

def main():
    docs = load_text_docs("datasets/raw")
    chunker = SimpleChunker(chunk_size = 600, overlap = 100)
    embedder = HFEmbedder(EMBED_MODEL_ID, normalize = True)

    store = QdrantStore(QDRANT_URL, QDRANT_COLLECTION)
    store.ensure_collections(vector_size = embedder.dim())

    points = []
    max_chunks_to_insert = 20

    for path, text in docs:
        chunks = chunker.chunk(text)
        for ch in chunks:
            if len(points) >= max_chunks_to_insert:
                break
            
            vec = embedder.embed([ch])[0]

            points.append(
                qm.PointStruct(
                    id = str(uuid.uuid4()),
                    vector = vec,
                    payload = {
                        "source": path,
                        "text": ch,
                    },
                )
            )

        if len(points) >= max_chunks_to_insert:
            break
    
    store.upsert_points(points)
    print(f"Upserted {len(points)} points into collection '{QDRANT_COLLECTION}'")

    query_text = "What is Qdrant used for?"
    qvec = embedder.embed(query_text)

    results = store.query(qvec, limit = 3)

    print("\nQuery: ", query_text)
    print("Top results:\n")
    for i, r in enumerate(results, start = 1):
        print(f"{i}) score = {float(r.score):.4f} source = {r.payload.get('source')}")
        print("\nText: ", (r.payload.get('text') or "")[:120].replace("\n", " "))
        print()

if __name__ == "__main__":
    main()