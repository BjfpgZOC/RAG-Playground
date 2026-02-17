from __future__ import annotations

from typing import Any, Dict, List

def build_rag_prompt(contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []

    for i, c in enumerate(contexts, start = 1):
        txt = ((c.get("text") or "").strip())
        ctx_lines.append(txt)

    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context retrieved)"

    sys_prompt = f"You are a helpful assistant. Use ONLY the context below to answer the question. If the context does not contain the answer, Say: I don't know based on the provided context.\nCONTEXT: {context_block}"

    return sys_prompt