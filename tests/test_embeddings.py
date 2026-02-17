from rag.loaders.local_files import load_text_docs
from rag.chunking.simple_chunker import SimpleChunker
from rag.embeddings.hf_embedder import HFEmbedder

from rag.settings import EMBED_MODEL_ID

def main():
    docs = load_text_docs("datasets/raw")
    chunker = SimpleChunker(chunk_size = 120, overlap = 30)
    embedder = HFEmbedder(EMBED_MODEL_ID, normalize = True)

    print("Embedding model: ", EMBED_MODEL_ID)
    print("Embedding dim: ", embedder.dim())
    print()

    path, text = docs[0]
    chunks = chunker.chunk(text)
    to_embed = chunks[:2]
    
    vectors = embedder.embed(to_embed)

    print("Doc: ", path)
    print("Chunks embedded: ", len(to_embed))
    print("Vectors returned: ", len(vectors))
    print("Vector[0] length: ", len(vectors[0]))
    print("Vector[0] preview (first 8 nums): ", vectors[0][:8])

if __name__ == "__main__":
    main()