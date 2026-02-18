from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str
    confidence_score: float
    confidence_label: str