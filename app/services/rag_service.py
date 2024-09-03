from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.services.embedding_service import EmbeddingService
from app.database import ConversationHistory
from sqlalchemy.orm import Session

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        self.retriever = self.embedding_service.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def add_to_history(self, db: Session, user_id: str, document_id: str, question: str, answer: str):
        db_item = ConversationHistory(user_id=user_id, document_id=document_id, question=question, answer=answer)
        db.add(db_item)
        db.commit()
        db.refresh(db_item)

    def get_history(self, db: Session, user_id: str, document_id: str, limit: int = 5):
        return db.query(ConversationHistory).filter(
            ConversationHistory.user_id == user_id,
            ConversationHistory.document_id == document_id
        ).order_by(ConversationHistory.timestamp.desc()).limit(limit).all()

    def refactor_question(self, question: str, history: list):
        if not history:
            return question

        context = "\n".join([f"Q: {h.question}\nA: {h.answer}" for h in history])
        prompt = f"""Given the following conversation history and a new question, 
        rewrite the question to include relevant context if necessary. 
        If the question doesn't need context, return it as is.

        Conversation history:
        {context}

        New question: {question}

        Rewritten question:"""

        response = self.llm.invoke(prompt)
        return response.content

    def get_response(self, db: Session, user_id: str, document_id: str, question: str):
        history = self.get_history(db, user_id, document_id)
        
        # Always attempt to refactor the question using recent history
        refactored_question = self.refactor_question(question, history[-3:])
        print(f"refactored question: {refactored_question}") # Use last 3 for refactoring
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        try:
            response = rag_chain.invoke({"input": refactored_question})
            answer = response["answer"]
            self.add_to_history(db, user_id, document_id, question, answer)
            return answer
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your question. Please try again."