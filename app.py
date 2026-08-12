"""
FastAPI wrapper around the RAG pipeline (retrieve.py + generate.py).

This turns your terminal script into an HTTP service — the piece that
lets Google Chat (or anything else on the internet) actually reach it,
instead of only being callable from your own terminal.

Requires:
    pip install fastapi "uvicorn[standard]"

Run locally with:
    uvicorn app:app --reload --port 8000

Then test at http://127.0.0.1:8000/docs — FastAPI auto-generates an
interactive test page, no curl/Postman needed.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from gemini_llm import generate_answer

app = FastAPI(title="Policy RAG Bot")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    """Simple uptime check — hosting platforms often ping this."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = generate_answer(request.question)
    return ChatResponse(answer=answer)
