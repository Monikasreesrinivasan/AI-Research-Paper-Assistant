from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict

from app.pdf import pdf_router
from app.pdf.utils import get_pdf_service, get_ollama_service

# Models
class Question(BaseModel):
    pdf_id: str
    question: str

class Answer(BaseModel):
    answer: str

class PDFResponse(BaseModel):
    pdf_id: str
    filename: str

class ActivePDFs(BaseModel):
    pdfs: List[PDFResponse]

# Routes
@pdf_router.post("/upload/", response_model=PDFResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    pdf_service=Depends(get_pdf_service)
):
    # Validate file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Create a new session
        pdf_id = pdf_service.create_session(file.filename, file_content)
        
        return {"pdf_id": pdf_id, "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@pdf_router.post("/ask/", response_model=Answer)
async def ask_question(
    question_data: Question,
    pdf_service=Depends(get_pdf_service),
    ollama_service=Depends(get_ollama_service)
):
    pdf_id = question_data.pdf_id
    
    try:
        # Get the session
        session = pdf_service.get_session(pdf_id)
        question = question_data.question
        
        # Retrieve relevant context using simple search
        context = " ".join(session.simple_search(question))
        
        # Generate answer using Ollama
        answer = ollama_service.answer_with_context(question, context)
        
        return {"answer": answer}
    
    except KeyError:
        raise HTTPException(status_code=404, detail="PDF session not found. Please upload the PDF again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@pdf_router.get("/active/", response_model=ActivePDFs)
async def get_active_pdfs(pdf_service=Depends(get_pdf_service)):
    pdfs = pdf_service.get_active_pdfs()
    return {"pdfs": pdfs}