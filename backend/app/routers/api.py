# backend/app/routers/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from io import BytesIO
from ..models.schemas import ChatRequest, ChatResponse
from ..services.rag_service import process_upload, process_chat

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = BytesIO(await file.read())
    return process_upload(content, file.filename)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return process_chat(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))