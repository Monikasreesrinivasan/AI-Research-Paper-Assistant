# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Settings:
#     PROJECT_NAME: str = "Research Paper Assistant"
#     PROJECT_VERSION: str = "1.0.0"
    
#     MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
#     DATABASE_NAME: str = os.getenv("DATABASE_NAME", "research_assistant")
    
#     SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
#     ALGORITHM: str = "HS256"
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
#     OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
#     OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    
#     SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
#     SEMANTIC_SCHOLAR_URL: str = "https://api.semanticscholar.org/graph/v1"
    
#     UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
#     VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "vector_db")
    



# settings = Settings()

# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Research Paper Assistant"
    PROJECT_VERSION: str = "1.0.0"
    
    # Database settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "research_assistant")
    
    # Authentication settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Ollama settings
    OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    
    # Semantic Scholar settings
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    SEMANTIC_SCHOLAR_URL: str = "https://api.semanticscholar.org/graph/v1"
    
    # File upload settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "vector_db")
    MAX_UPLOAD_SIZE: int = 20 * 1024 * 1024  # 20MB max upload size
    
    # Paper processor settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
    AUDIO_OUTPUT_DIR: str = os.getenv("AUDIO_OUTPUT_DIR", os.path.join(UPLOAD_DIR, "audio"))

settings = Settings()