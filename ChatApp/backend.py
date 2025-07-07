from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
def chat_with_llm(request: ChatRequest):
    ollama_payload = {
        "model": MODEL,
        "prompt": request.prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=ollama_payload)
    result = response.json()
    return {"response": result.get("response", "")}
