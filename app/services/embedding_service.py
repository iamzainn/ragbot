import requests
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from fastapi import HTTPException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import Config

class EmbeddingService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = Chroma(persist_directory=Config.CHROMA_PERSIST_DIRECTORY, embedding_function=self.embeddings)
        self.conversation_history = {}
        
    def add_to_history(self, user_id: str, question: str, answer: str):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({"question": question, "answer": answer})
        # print("history : ",self.conversation_history)
        # Keep only the last 5 exchanges
        self.conversation_history[user_id] = self.conversation_history[user_id][-5:]
        # print(" last 5 history : ",self.conversation_history)

    def get_history(self, user_id: str):
        return self.conversation_history.get(user_id, [])    

    def fetch_document_text(self, document_url: str, document_type: str):
        response = requests.get(document_url)
        response.raise_for_status()

        if document_type == 'application/pdf':
            return self._extract_pdf_text(response.content)
        elif document_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return self._extract_docx_text(response.content)
        elif document_type == 'text/plain':
            return response.text
        else:
            raise ValueError("Unsupported document type")

    def _extract_pdf_text(self, content: bytes):
        reader = PdfReader(BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def _extract_docx_text(self, content: bytes):
        document = Document(BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])
        return text

    def create_embeddings(self, user_id: str, document_id: str, document_url: str, document_type: str):
        try:
            text = self.fetch_document_text(document_url, document_type)
            
            chunks = self.text_splitter.split_text(text)
           
            metadatas = [{"user_id": user_id, "document_id": document_id} for _ in chunks]
            self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            return {"message": "Embeddings created successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

    def delete_embeddings(self, user_id: str, document_id: str):
        where_clause = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"document_id": {"$eq": document_id}}
            ]
        }

        results = self.vectorstore.get(where=where_clause)

        if not results['ids']:
            return {"message": "No matching embeddings found to delete"}

        self.vectorstore.delete(ids=results['ids'])

        return {"message": f"Deleted {len(results['ids'])} embeddings"}

    def delete_all_embeddings(self):
        # Retrieve all embeddings
        results = self.vectorstore.get(where={})

        if not results['ids']:
            return {"message": "No embeddings found to delete"}

        # Delete all embeddings by IDs
        self.vectorstore.delete(ids=results['ids'])

        return {"message": f"Deleted {len(results['ids'])} embeddings"}
    