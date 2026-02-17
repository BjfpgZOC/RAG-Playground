from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from rag.parsing.pdf_text import extract_text_from_pdf

def load_text_docs(input_dir: str) -> List[Tuple[str, str]]:
    root = Path(input_dir)

    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    docs: List[Tuple[str, str]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in [".txt", ".md"]:
            text = path.read_text(encoding = "utf-8", errors = "ignore")
            docs.append((str(path), text))
            continue

        if suffix in [".pdf"]:
            text = extract_text_from_pdf(str(path))
            docs.append((str(path), text))

    return docs