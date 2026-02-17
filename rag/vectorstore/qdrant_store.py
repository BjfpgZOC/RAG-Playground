from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client import models as qm

@dataclass
class QdrantStore:
    url: str
    collection: str

    def _client(self) -> QdrantClient:
        return QdrantClient(url = self.url)
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        client = self._client()

        client.delete(
            collection_name = self.collection,
            points_selector = qm.FilterSelector(
                filter = qm.Filter(
                    must = [
                        qm.FieldCondition(
                            key = "doc_id",
                            match = qm.MatchValue(value = doc_id)
                        )
                    ]
                )
            )
        )

    def recreate_collections(self, vector_size: int) -> None:
        client = self._client()

        client.recreate_collection(
            collection_name = self.collection_name,
            vectors_config = qm.VectorParams(
                size = vector_size,
                distance = qm.Distance.COSINE,
            )
        )
    
    def ensure_collections(self, vector_size: int) -> None:
        client = self._client()

        existing = client.get_collections()
        existing_names = [c.name for c in existing.collections]

        if self.collection in existing_names:
            return
        
        client.create_collection(
            collection_name = self.collection,
            vectors_config = qm.VectorParams(
                size = vector_size,
                distance = qm.Distance.COSINE,
            ),
        )

    def upsert_points(self, points: List[qm.PointStruct]) -> None:
        client = self._client()

        client.upsert(
            collection_name = self.collection,
            points = points,
        )

    def query(self, query_vector: List[float], limit: int = 5) -> List[qm.ScoredPoint]:
        client = self._client()

        res = client.query_points(
            collection_name = self.collection, 
            query = query_vector,
            limit = limit, 
            with_payload = True,
        )

        return res.points