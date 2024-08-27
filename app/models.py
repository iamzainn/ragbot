from pydantic import BaseModel, HttpUrl

class EmbeddingRequest(BaseModel):
    userId: str
    documentId: str
    documentUrl: HttpUrl
    documentType: str  # Add document type here

class QuestionRequest(BaseModel):
    userId: str
    question: str
    sessionId: str = None

class DeleteEmbeddingRequest(BaseModel):
    userId: str
    documentId: str