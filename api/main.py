from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.knowledge_base import KnowledgeBase

app = FastAPI()

kb = KnowledgeBase()


class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None  # same id = same conversation; omit for a stateless call


@app.get("/")
def home():
    return {
        "message": "KnowledgePilot API is running!"
    }


@app.post("/ask")
def ask(request: QuestionRequest):

    result = kb.ask(request.question, session_id=request.session_id)

    print(type(result))
    print(result)

    return result