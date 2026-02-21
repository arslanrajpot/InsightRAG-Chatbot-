
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    # For scalability, add REDIS_URL = os.getenv("REDIS_URL") later

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables. Check your .env file.")