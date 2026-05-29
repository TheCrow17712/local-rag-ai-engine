import os

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import string
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Verified Reasoning Engine")

print("Initializing local reasoning architecture... (Fast Download)")

local_generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    retrieved_context = search_local_context(request.prompt)

    if request.session_id not in SESSION_STORAGE:
        SESSION_STORAGE[request.session_id] = []

    if retrieved_context == "No custom context found.":
        full_prompt = f"<|system|>\nYou are a helpful assistant. User's name is Crow.<|end|>\n<|user|>\n{request.prompt}<|end|>\n<|assistant|>\n"
    else:
        full_prompt = f"<|system|>\nAnswer using ONLY this context sentence: {retrieved_context}<|end|>\n<|user|>\n{request.prompt}<|end|>\n<|assistant|>\n"

    try:
        outputs = local_generator(
            full_prompt,
            max_new_tokens=40,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1
        )

        raw_text = outputs[0]["generated_text"]
        ai_message = raw_text.replace(full_prompt, "").strip()

        return {
            "session_id": request.session_id,
            "context_used": retrieved_context,
            "response": ai_message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local Inference Engine error: {str(e)}")
