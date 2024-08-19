import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    CHROMA_PERSIST_DIRECTORY = 'chroma_db'