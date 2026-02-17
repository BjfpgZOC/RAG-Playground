from rag.loaders.local_files import load_text_docs
from rag.chunking.simple_chunker import SimpleChunker

def main():
    docs = load_text_docs("datasets/raw")

    chunker = SimpleChunker(chunk_size = 60, overlap = 15)

    for doc_i, (path, text) in enumerate(docs):
        chunks = chunker.chunk(text)

        print("=" * 60)
        print(f"Doc {doc_i + 1}: {path}")
        print(f"Total chars: {len(text)}")
        print(f"Chunks: {len(chunks)}")

        for i, ch in enumerate(chunks):
            print(f"\n--- chunk {i + 1} (len = {len(ch)}) ---")
            print(ch)
        
if __name__ == "__main__":
    main()