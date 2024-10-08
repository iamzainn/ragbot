from fastapi import FastAPI, Depends, HTTPException
from requests import Session
from app.models import EmbeddingRequest, QuestionRequest, DeleteEmbeddingRequest
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService
# from app.utils.auth import get_api_key
from app.database import get_db, engine, Base

Base.metadata.create_all(bind=engine)


app = FastAPI()

embedding_service = EmbeddingService()
rag_service = RAGService()

@app.post("/create_embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        result = embedding_service.create_embeddings(
            request.userId,
            request.documentId,
            request.documentUrl,
            request.documentType
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_response")
async def get_response(request: QuestionRequest, db: Session = Depends(get_db)):
    try:
        response = rag_service.get_response(
            db,
            request.userId,
            request.documentId,
            request.question
        )
        return {"response": response}
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_embeddings")
async def delete_embeddings(request: DeleteEmbeddingRequest):
    try:
        result = embedding_service.delete_embeddings(
            request.userId,
            request.documentId
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deleteAll")
async def delete_all_embeddings():
    try:
        result = embedding_service.delete_all_embeddings()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))