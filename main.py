import os

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import string
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Strict Local QA Engine")

print("Initializing local Question-Answering architecture...")
# 🚀 SWITCH: Using a dedicated QA pipeline which is immune to chat-tag loops
local_qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)
print("AI Model loaded successfully!")

SESSION_STORAGE = {}

LOCAL_KNOWLEDGE_BASE = [
    {
        "text": "The AI Summer School 2026 is a one-week residential programme at Holmewood House from 26 July to 2 August 2026."},
    {"text": "The AI Summer School application deadline is officially June 12th, 2026."},
    {
        "text": "The programme selects only 30 extraordinary students aged 14 to 17 and provides a full financial bursary."},
    {
        "text": "Course themes include AI for study, work productivity, research fact-checking, and human-machine collaboration."},
    {
        "text": "The course features hands-on group projects and industry excursions to places like Apple HQ, Bletchley Park, and Salesforce AI Centre."}
]


def search_local_context(user_query: str) -> str:
    clean_query = user_query.lower().translate(str.maketrans('', '', string.punctuation))
    query_words = clean_query.split()
    for item in LOCAL_KNOWLEDGE_BASE:
        if any(word in item["text"].lower() for word in query_words if len(word) > 3):
            return item["text"]
    return "No custom context found."


class ChatRequest(BaseModel):
    session_id: str
    prompt: str


@app.post("/chat/advanced")
async def advanced_chat_engine(request: ChatRequest):
    # Responsible AI: Privacy PII filter
    if any(char.isdigit() for char in request.prompt) and len(request.prompt) >= 11:
        raise HTTPException(status_code=400, detail="Safety Guard: Do not submit phone numbers.")

    retrieved_context = search_local_context(request.prompt)

    # State Tracking
    if request.session_id not in SESSION_STORAGE:
        SESSION_STORAGE[request.session_id] = {"name": "User"}

    # Track name if user introduces themselves
    lower_prompt = request.prompt.lower()
    if "my name is" in lower_prompt:
        name_part = request.prompt.split("is")[-1].strip().replace(".", "")
        SESSION_STORAGE[request.session_id]["name"] = name_part
        return {
            "session_id": request.session_id,
            "context_used": "Local State Memory",
            "response": f"Hello {name_part}! I have saved your name to this session memory."
        }

    # Handle identity questions locally using State Memory
    if "my name" in lower_prompt:
        saved_name = SESSION_STORAGE[request.session_id]["name"]
        return {
            "session_id": request.session_id,
            "context_used": "Local State Memory",
            "response": f"Your name is {saved_name}."
        }

    # If it is a regular question, run the local QA reasoning engine over the RAG context
    try:
        if retrieved_context == "No custom context found.":
            # Fallback for general greetings or out-of-bounds questions
            return {
                "session_id": request.session_id,
                "context_used": retrieved_context,
                "response": "Hello! I am your AI academic assistant. Ask me anything about the Holmewood House AI Summer School program."
            }

        # Run exact model reasoning over the extracted context
        qa_result = local_qa(
            question=request.prompt,
            context=retrieved_context
        )

        ai_message = qa_result["answer"].strip()

        return {
            "session_id": request.session_id,
            "context_used": retrieved_context,
            "response": f"Based on the official program details: {ai_message}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local QA Engine error: {str(e)}")
