from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.services.embedding_service import EmbeddingService


from app.database import SessionLocal, ConversationHistory
from sqlalchemy.orm import Session

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        self.retriever = self.embedding_service.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    def add_to_history(self, db: Session, user_id: str, question: str, answer: str):
        db_item = ConversationHistory(user_id=user_id, question=question, answer=answer)
        db.add(db_item)
        db.commit()
        db.refresh(db_item)

    def get_history(self, db: Session, user_id: str, limit: int = 5):
        return db.query(ConversationHistory).filter(ConversationHistory.user_id == user_id).order_by(ConversationHistory.timestamp.desc()).limit(limit).all()

    def get_response(self, db: Session, user_id: str, question: str):
        history = self.get_history(db, user_id)
        print("history : ",history)
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. Consider the conversation history when answering follow-up questions."
            "\n\n"
            "Conversation history:\n{history}\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        history_text = "\n".join([f"Q: {h.question}\nA: {h.answer}" for h in history])
        
        response = rag_chain.invoke({
            "input": question,
            "history": history_text
        })
        
        answer = response["answer"]
        self.add_to_history(db, user_id, question, answer)
        
        return answer

