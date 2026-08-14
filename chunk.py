"""
This file chunks the mark down files down to chunks
"""
from pathlib import Path
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import json

INPUT_DIR = Path("FormattedExtraction")
OUTPUT_DIR = Path("chunks")

OUTPUT_DIR.mkdir(exist_ok=True)

# Load the tokenizer for E5
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")

# Split using the same tokenizer as the embedding model
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=512,      # tokens
    chunk_overlap=64,    # tokens
)

prefix = "passage: "
prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=512 - prefix_tokens - 8,  # reserve room for the prefix, tested to find max chunk size without any chunks overextending
    chunk_overlap=64,
)

#glob is used for wildcard matching
for md_file in INPUT_DIR.rglob("*.md"):
    try:
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)

        json_data = []
        #Using json as its the standard. 

        for i, chunk in enumerate(chunks):
            json_data.append({
                "chunk_id": i,
                "source_file": md_file.name,
                "text": f"passage: {chunk}"
            })

        output_file = OUTPUT_DIR / f"{md_file.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        print(f"✓ {md_file.name} -> {len(chunks)} chunks")

    except Exception as e:
        print(f"✗ Failed: {md_file.name}")
        print(e)

print("\nChunking complete!")