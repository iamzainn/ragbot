from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    userId: str
    documentId: str
    text: str

class QuestionRequest(BaseModel):
    userId: str
    question: str

class DeleteEmbeddingRequest(BaseModel):
    userId: str
    documentId: str