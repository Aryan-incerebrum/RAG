"""
Take a user question, embed it (query-side), and retrieve the most
relevant chunks from the persisted ChromaDB collection.
"""
import torch.nn.functional as F
import chromadb
import torch
from transformers import AutoModel, AutoTokenizer

DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_chunks"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()


def embed_query(question: str) -> list[float]:
    # NOTE: "query: " prefix — different from "passage: " used at index time.
    # This asymmetry is required by E5; mixing them up silently hurts results.
    text = f"query: {question}"
    tokens = tokenizer(
        [text], padding=True, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**tokens)
        last_hidden = out.last_hidden_state.masked_fill(
            ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        embed = last_hidden.sum(dim=1) / tokens["attention_mask"].sum(dim=1)[..., None]
        embeddings = F.normalize(embed, p=2, dim=1)
    return embeddings.cpu().numpy().tolist()


def retrieve(question: str, top_k: int = TOP_K):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    # results is a dict of lists, one list per query — we only sent one query
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return list(zip(docs, metas, distances))


def main():
    question = input("Ask a question: ").strip()
    if not question:
        print("No question entered.")
        return

    matches = retrieve(question)

    print(f"\nTop {len(matches)} matches:\n")
    for i, (doc, meta, dist) in enumerate(matches, start=1):
        print(f"--- Match {i} (distance={dist:.4f}) ---")
        print(f"Source: {meta['source_file']} | chunk_id: {meta['chunk_id']}")
        print(doc[:300] + ("..." if len(doc) > 300 else ""))
        print()


if __name__ == "__main__":
    main()
