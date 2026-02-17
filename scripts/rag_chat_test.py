from __future__ import annotations

import argparse

from rag.pipelines.rag_pipeline import RAGPipeline

def main(args):
    pipe = RAGPipeline(top_k = args.top_k)
    output = pipe.answer(question = args.question)

    print("\nQuestion:\n", output["question"])
    print("\nResponse:\n", output["response"])

    for i, c in enumerate(output["context"], start = 1):
        print(f"{i}] score: {c['score']:.4f} source: {c['source']} chunk_index: {c['chunk_index']}")
        print((c["text"] or "").replace("\n", " "))

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type = str, required = True)
    parser.add_argument("--top_k", type = int, default = 5)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    main(args)