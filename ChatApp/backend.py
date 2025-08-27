from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"

class ChatRequest(BaseModel):
    messages: list

@app.post("/chat")
def chat_with_llm(chat_req: ChatRequest):
    payload = {
        "model": MODEL_NAME,
        "messages": chat_req.messages,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    return response.json()
