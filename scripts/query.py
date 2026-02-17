from __future__ import annotations

import argparse

from rag.retriever.dense_retriever import DenseRetriever

def main(args) -> None:
    retriever = DenseRetriever()
    hits = retriever.retrieve(args.query, args.top_k, args.min_score)

    print("\nQuery:", args.query)
    print("Returned:", len(hits), "chunks\n")

    for i, h in enumerate(hits, start = 1):
        print(f"{i}) score = {h['score']:.4f} source = {h['source']} chunk_index = {h['chunk_index']}")
        text_preview = (h["text"] or "").replace("\n", " ")[:200]
        print("Text:", text_preview)
        print()

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type = str, required = True)
    parser.add_argument("--top_k", type = int, default = 5)
    parser.add_argument("--min_score", type = float, default = None)    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    main(args)