from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class HFEmbedder:
    model_id: str
    normalize: bool = True

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_id)

    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs =  self.model.encode(
            texts,
            normalize_embeddings = self.normalize,
            show_progress_bar = True,
        )
        
        vecs = np.array(vecs, dtype = np.float32)
        
        return vecs
