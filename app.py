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

ngrok http 8000 - for the live website!
make sure to run both terminals at the same time, we want the json api
to be running and a live site link too.
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
from generate import generate_answer

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


@app.post("/google-chat-webhook")
async def google_chat_webhook(request: Request):
    """
    Handles events Google Chat POSTs to this endpoint.

    This app was set up under Google's newer Workspace Add-ons framework
    for Chat apps, which uses a nested event schema — not the older flat
    {"type": "MESSAGE", ...} format. The real content lives under
    body["chat"]["messagePayload"] (for a message) or
    body["chat"]["addedToSpacePayload"] (for being added to a space).

    Responses also need a specific wrapper shape:
    {"hostAppDataAction": {"chatDataAction": {"createMessageAction": {"message": {...}}}}}
    — a bare {"text": "..."} is silently accepted (200 OK) but never
    displayed, which is exactly what was happening before this fix.
    """
    body = await request.json()
    chat_event = body.get("chat", {})

    def reply(text: str):
        return {
            "hostAppDataAction": {
                "chatDataAction": {
                    "createMessageAction": {
                        "message": {"text": text}
                    }
                }
            }
        }

    if "messagePayload" in chat_event:
        message = chat_event["messagePayload"].get("message", {})
        question = message.get("argumentText", message.get("text", "")).strip()

        if not question:
            return reply("Ask me something about company policy!")

        answer = generate_answer(question)
        return reply(answer)

    elif "addedToSpacePayload" in chat_event:
        return reply("Hi! I'm the policy bot — ask me anything about company policies.")

    # Other event types — nothing to reply with
    return {}