from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    reader = PdfReader(str(path))

    pages_text: List[str] = []

    for page_idx, page in enumerate(reader.pages, start = 1):
        page_text = page.extract_text() or ""
        pages_text.append(f"\n\n--- PAGE {page_idx + 1} ---\n\n{page_text}")

    return "".join(pages_text)