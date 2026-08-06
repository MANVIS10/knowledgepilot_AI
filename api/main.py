from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.knowledge_base import KnowledgeBase

app = FastAPI()

kb = KnowledgeBase()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "KnowledgePilot API is running!"
    }


@app.post("/ask")
def ask(request: QuestionRequest):

    result = kb.ask(request.question)

    print(type(result))
    print(result)

    return result