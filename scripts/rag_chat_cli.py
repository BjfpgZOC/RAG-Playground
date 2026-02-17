from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import requests

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

def post_chat(
    question: str,
    top_k: int = 5, 
    min_score: Optional[float] = None,
    return_contexts: bool = False,
) -> Dict[str, Any]:
    
    r = requests.post(
        url = f"{API_URL}/chat",
        json = {
            "question": question,
            "top_k": top_k,
            "min_score": min_score,
            "return_contexts": return_contexts,
        },
        timeout = 120,
    )
    
    r.raise_for_status()
    
    return r.json()

def main() -> int:
    print(f"RAG CLI -> {API_URL}/chat")
    print("Type 'exit' or 'quit' to stop the chat.\n")

    while True:
        try:
            q = input("> ").strip()
        except(EOFError, KeyboardInterrupt):
            print()
            return 0
        
        if not q:
            continue
        if q.lower() in ['exit', 'quit']:
            return 0
        
        try:
            out = post_chat(q)
            print(out.get("response", "").strip())
            print()
        except requests.HTTPError as e:
            try:
                print(e.response.text)
            except Exception:
                pass
            print()
        except Exception as e:
            print("Error: {e}\n")

    return 0

if __name__ == "__main__":
    raise(SystemExit(main()))