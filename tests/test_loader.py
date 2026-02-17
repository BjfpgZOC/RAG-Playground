from rag.loaders.local_files import load_text_docs

def main():
    input_dir = "datasets/raw"

    docs = load_text_docs(input_dir)

    print(f"Loaded {len(docs)} documents\n")

    for i, (path, text) in enumerate(docs):
        print(f"--- Document {i + 1} ---")
        print(f"Path: ", path)
        print("Text preview", text[:120].replace("\n", " "))
        print()

if __name__ == "__main__":
    main()