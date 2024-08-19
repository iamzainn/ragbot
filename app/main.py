from fastapi import FastAPI, Depends, HTTPException
from app.models import EmbeddingRequest, QuestionRequest, DeleteEmbeddingRequest
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService
from app.utils.auth import get_api_key

app = FastAPI()

embedding_service = EmbeddingService()
rag_service = RAGService()

@app.post("/create_embeddings")
async def create_embeddings(request: EmbeddingRequest, api_key: str = Depends(get_api_key)):
    try:
        result = embedding_service.create_embeddings(
            request.userId,
            request.documentId,
            request.text
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_response")
async def get_response(request: QuestionRequest, api_key: str = Depends(get_api_key)):
    try:
        response = rag_service.get_response(
            request.userId,
            request.question
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_embeddings")
async def delete_embeddings(request: DeleteEmbeddingRequest, api_key: str = Depends(get_api_key)):
    try:
        result = embedding_service.delete_embeddings(
            request.userId,
            request.documentId
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))