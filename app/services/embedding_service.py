from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import Config

class EmbeddingService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = Chroma(persist_directory=Config.CHROMA_PERSIST_DIRECTORY, embedding_function=self.embeddings)

    def create_embeddings(self, user_id: str, document_id: str, text: str):
        chunks = self.text_splitter.split_text(text)
        metadatas = [{"user_id": user_id, "document_id": document_id} for _ in chunks]
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        return {"message": "Embeddings created successfully"}

    def delete_embeddings(self, user_id: str, document_id: str):
        # Construct the where clause using $and operator
        where_clause = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"document_id": {"$eq": document_id}}
            ]
        }

        # Get the IDs of the documents to delete
        results = self.vectorstore.get(where=where_clause)
        # print(results)
        
        if not results['ids']:
            return {"message": "No matching embeddings found to delete"}
        
        # Delete the documents using their IDs
        self.vectorstore.delete(ids=results['ids'])
        
        return {"message": f"Deleted {len(results['ids'])} embeddings"}