"""
Full RAG pipeline: retrieve relevant chunks from Chroma, then generate a
grounded answer using Google's free Gemini API.

Requires:
    pip install google-genai python-dotenv
    A .env file in this directory containing:
        GEMINI_API_KEY=your_key_here
    (Get a free key at https://aistudio.google.com — no card required.)

This file expects retrieve.py to be in the same directory — it reuses
its retrieve() function rather than duplicating the embedding logic.
"""

from dotenv import load_dotenv
from google import genai
from retrieve import retrieve  # reuses your existing retrieval pipeline

load_dotenv()  # reads GEMINI_API_KEY from .env into the environment

client = genai.Client()  # automatically picks up GEMINI_API_KEY

GEMINI_MODEL = "gemini-3.5-flash"  # fast + free-tier friendly


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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text


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