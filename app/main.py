from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.auth.routes import router as auth_router
from app.papers.routes import router as papers_router
from app.chat.routes import router as chat_router
from app.similar.routes import router as similar_router
from app.config import settings
from app.similar.routes import router as similar_router
from app.analyzer.routes import router as analyzer_router
from app.comparison import comparison_bp  # Import the new blueprint
from app.pdf import pdf_router  # New PDF router
from app.search.routes import router as search_router  # Import the new search router
from app.processor.routes import router as processor_router  # Add this line
from app.plagiarism.routes import router as plagiarism_router  # Add the plagiarism router


from app.models.user import create_user_indexes
import os

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.AUDIO_OUTPUT_DIR, exist_ok=True)  # Add this line

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(papers_router, prefix="/papers", tags=["Papers"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(similar_router, prefix="/api/similar", tags=["Similar Papers"])
app.include_router(analyzer_router, prefix="/analyzer", tags=["analyzer"])
app.include_router(comparison_bp, prefix="/comparison", tags=["Paper Comparison"])  # Add the new router
app.include_router(pdf_router, prefix="/pdf", tags=["PDF Chat"])  # Add PDF router
app.include_router(search_router, prefix="/search", tags=["Research Article Search"])  # Add the new router
app.include_router(processor_router, prefix="/processor", tags=["Paper Processing"])  # Add this line
app.include_router(plagiarism_router, prefix="/plagiarism", tags=["Plagiarism Detection"])  # Add the plagiarism router

@app.on_event("startup")
async def startup_event():
    # Create user indexes
    await create_user_indexes()

@app.get("/")
async def root():
    return {"message": "Welcome to Research Paper Assistant API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
