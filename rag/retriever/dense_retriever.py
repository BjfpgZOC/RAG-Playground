from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.embeddings.hf_embedder import HFEmbedder
from rag.vectorstore.qdrant_store import QdrantStore
from rag.settings import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL_ID

@dataclass
class DenseRetriever:
    qdrant_url: str = QDRANT_URL
    collection: str = QDRANT_COLLECTION
    embed_model_id: str = EMBED_MODEL_ID

    def __post_init__(self) -> None:
        self.embedder = HFEmbedder(model_id = self.embed_model_id, normalize = True)
        self.store = QdrantStore(url = self.qdrant_url, collection = self.collection)

    def retrieve(self, query_text: str, top_k: int = 5, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed([query_text])[0]
        results = self.store.query(query_vector = query_vec, limit = top_k)

        out: List[Dict[str, Any]] = []

        for r in results:
            score = float(r.score)
            payload = r.payload or {}

            if min_score is not None and score < min_score:
                continue

            out.append({
                "score": score, 
                "source": payload.get("source"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
            })

        return out