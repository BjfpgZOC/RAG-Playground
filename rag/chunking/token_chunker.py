from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict

from transformers import AutoTokenizer

@lru_cache(maxsize = 4)
def _get_tokenizer(tokenizer_id: str):

    tok =  AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast = True, 
        trust_remote_code = False,
    )

    tok.model_max_length = 10**9
    return tok

@dataclass
class TokenChunker:
    chunk_tokens: int
    overlap_tokens: int
    tokenizer_id: str

    def __post_init__(self) -> None:
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be > 0")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be < chunk_tokens")

        self._tok = _get_tokenizer(self.tokenizer_id)

    def chunk_spans(self, text: str) -> List[Dict]:
        if not text or not text.strip():
            return []

        enc = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"]
        offsets = enc.get("offset_mapping", None)

        if not input_ids:
            return []

        n_tokens = len(input_ids)
        step = self.chunk_tokens - self.overlap_tokens

        chunks: List[Dict] = []

        for t_start in range(0, n_tokens, step):
            t_end = min(t_start + self.chunk_tokens, n_tokens)

            window_ids = input_ids[t_start:t_end]
            chunk_text = self._tok.decode(window_ids, skip_special_tokens=True).strip()

            if not chunk_text:
                continue

            if (
                offsets
                and t_end - 1 < len(offsets)
                and offsets[t_start] != (0, 0)
                and offsets[t_end - 1] != (0, 0)
            ):
                c_start = offsets[t_start][0]
                c_end = offsets[t_end - 1][1]
            else:
                c_start = -1
                c_end = -1

            chunks.append(
                {
                    "text": chunk_text,
                    "token_start": t_start,
                    "token_end": t_end,
                    "char_start": c_start,
                    "char_end": c_end,
                }
            )

            if t_end >= n_tokens:
                break

        return chunks

    def chunk(self, text: str) -> List[str]:
        return [c["text"] for c in self.chunk_spans(text)]


