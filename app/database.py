from sqlalchemy import Index, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime

from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./rag_chatbot.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    document_id = Column(String, index=True)
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_user_document', user_id, document_id),
    )