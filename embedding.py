
"""
Embed chunks (produced by your chunking script) and store them in a
persistent local ChromaDB collection.

Tried to use the GPU here but it's really not needed. 
At the end of the day we are performing matrix multiplication on a 
not so significant scale.

Chromadb is perfect for vector indexing. They use a HNSW
(Hierarchical Navigable Samll World) which essentialy splits
the clusters into layers, and we traverse through these layyers
as the clusters increase in numbers downwards. 
FAISS from meta was the original plan but they don't have much
Windows support for GPU:( which was surprising, and this was before
the gpu had problems.

Requires: pip install chromadb
"""
import torch.nn.functional as F
import json
from pathlib import Path

import chromadb
import torch
from transformers import AutoModel, AutoTokenizer

CHUNKS_DIR = Path("chunks")
DB_DIR = "chroma_db"          # persisted to disk here
COLLECTION_NAME = "rag_chunks"
BATCH_SIZE = 32                # embed in batches to avoid huge GPU memory spikes

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "intfloat/e5-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval() # Switching into inference mode


def embed(docs: list[str]) -> list[list[float]]:
    # NOTE: docs passed in here should already have "passage: " prefix
    # applied upstream (your chunk JSON already stores it that way).
    tokens = tokenizer(
        docs, padding=True, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad(): #we don't want gradient calculation since we aren't changing weights
        out = model(**tokens)
        #We'd want to mask the extra padding we added 
        last_hidden = out.last_hidden_state.masked_fill(
            ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        #We use mean pooling
        doc_embeds = last_hidden.sum(dim=1) / \
            tokens["attention_mask"].sum(dim=1)[..., None]
        embeddings = F.normalize(doc_embeds, p=2, dim=1)
    return embeddings.cpu().numpy().tolist()


def load_all_chunks():
    """Yield (id, text, metadata) for every chunk across every JSON file."""
    for json_file in CHUNKS_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for chunk in chunks:
            chunk_id = f"{json_file.stem}_{chunk['chunk_id']}"
            yield chunk_id, chunk["text"], {
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
            }


def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    for chunk_id, text, meta in load_all_chunks():
        ids.append(chunk_id)
        texts.append(text)
        metadatas.append(meta)

    print(f"Loaded {len(ids)} chunks total. Embedding in batches of {BATCH_SIZE}...")

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_ids = ids[start:end]
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]

        batch_embeds = embed(batch_texts)

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeds,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"  Embedded + stored {end if end < len(ids) else len(ids)}/{len(ids)}")

    print(f"\n✓ Done. Collection '{COLLECTION_NAME}' persisted to ./{DB_DIR}")
    print(f"  Total items in collection: {collection.count()}")


if __name__ == "__main__":
    main()