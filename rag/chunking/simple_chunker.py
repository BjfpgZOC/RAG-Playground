from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass
class SimpleChunker:
    chunk_size: int = 800
    overlap: int = 120

    def chunk(self, text: str) -> List[str]:
        text = " ".join(text.split())

        if len(text) <= self.chunk_size:
            return [text]
        
        chunks: List[str] = []

        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)

            chunk_text = text[start:end]
            chunks.append(chunk_text)

            if end == len(text):
                break
            
            start = max(0, end - self.overlap)

        return chunks