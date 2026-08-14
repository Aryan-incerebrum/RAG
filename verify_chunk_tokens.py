"""
Verify actual token counts of generated chunks against the configured limit.

Run this from the same directory where your chunking script ran (so that
the "chunks" folder is visible), using the same conda/pip env (rag2).
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

CHUNKS_DIR = Path("chunks")
CHUNK_SIZE_LIMIT = 512  # must match what you used in the splitter


tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")

total_chunks = 0
over_limit = []
token_counts = []

for json_file in CHUNKS_DIR.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        text = chunk["text"]
        # Same measurement the splitter used internally
        n_tokens = len(tokenizer.encode(text))
        token_counts.append(n_tokens)
        total_chunks += 1

        if n_tokens > CHUNK_SIZE_LIMIT:
            over_limit.append({
                "file": json_file.name,
                "chunk_id": chunk["chunk_id"],
                "tokens": n_tokens,
            })

# --- Report ---
print(f"Total chunks checked: {total_chunks}")

if token_counts:
    print(f"Min tokens:  {min(token_counts)}")
    print(f"Max tokens:  {max(token_counts)}")
    print(f"Avg tokens:  {sum(token_counts) / len(token_counts):.1f}")

if over_limit:
    print(f"\n⚠ {len(over_limit)} chunk(s) exceeded {CHUNK_SIZE_LIMIT} tokens:")
    for item in over_limit:
        print(f"  - {item['file']} chunk_id={item['chunk_id']} -> {item['tokens']} tokens")
else:
    print(f"\n✓ All chunks are within the {CHUNK_SIZE_LIMIT}-token limit.")
