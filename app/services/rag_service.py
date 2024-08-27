from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.services.embedding_service import EmbeddingService


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        self.retriever = self.embedding_service.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.conversation_history = {}  # Add this line

    def get_history(self, user_id: str):
        return self.conversation_history.get(user_id, [])

    def add_to_history(self, user_id: str, question: str, answer: str):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({"question": question, "answer": answer})
        # Keep only the last 5 exchanges
        self.conversation_history[user_id] = self.conversation_history[user_id][-5:]

    def get_response(self, user_id: str, question: str):
        history = self.get_history(user_id)
        
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
        
        history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])
        
        response = rag_chain.invoke({
            "input": question,
            "history": history_text
        })
        
        answer = response["answer"]
        self.add_to_history(user_id, question, answer)
        
        return answer