"""
Full RAG pipeline: retrieve relevant chunks from Chroma, then generate a
grounded answer using a local Ollama model (qwen2.5:7b by default).

Requires:
    pip install ollama
    Ollama running locally with the model already pulled:
        ollama pull qwen2.5:7b

This file expects retrieve.py to be in the same directory — it reuses
its retrieve() function rather than duplicating the embedding logic.
"""

import ollama
from retrieve import retrieve  # reuses your existing retrieval pipeline

OLLAMA_MODEL = "qwen2.5:7b"


def build_prompt(question: str, matches: list) -> str:
    context_blocks = []
    for i, (doc, meta, dist) in enumerate(matches, start=1):
        # Strip the "passage: " prefix before showing it to the LLM —
        # that prefix was only needed for E5's embedding step, the LLM
        # doesn't need or want to see it.
        text = doc.removeprefix("passage: ")
        source = meta.get("source_file", "unknown")
        context_blocks.append(f"[{i}] (source: {source})\n{text}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions about company policy \
                using only the provided context. If the answer isn't in the context, say so clearly \
                instead of guessing.

                Context:
                {context}

                Question: {question}

                Answer:"""
    return prompt


def generate_answer(question: str, top_k: int = 5) -> str:
    matches = retrieve(question, top_k=top_k)

    if not matches:
        return "No relevant context found in the database."

    prompt = build_prompt(question, matches)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


def main():
    question = input("Ask a question: ").strip()
    if not question:
        print("No question entered.")
        return

    print("\nRetrieving context and generating answer...\n")
    answer = generate_answer(question)

    print("--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
