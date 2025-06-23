# # app/paper_processor/routes.py

# import os
# import shutil
# from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, Form, BackgroundTasks
# from fastapi.responses import FileResponse, JSONResponse
# from typing import List, Dict, Any, Optional
# import uuid
# from pydantic import BaseModel
# from services.paper_processor_service import SinglePaperSystem
# from app.config import settings

# router = APIRouter()

# # Initiate the paper processor system with the Groq API key
# GROQ_API_KEY = "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"
# UPLOAD_DIR = os.path.join(settings.UPLOAD_DIR, "papers")
# AUDIO_DIR = os.path.join(settings.UPLOAD_DIR, "audio")

# # Create directories if they don't exist
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(AUDIO_DIR, exist_ok=True)

# # Initialize the paper processor system
# paper_system = SinglePaperSystem(
#     groq_api_key=GROQ_API_KEY,
#     upload_dir=UPLOAD_DIR,
#     audio_dir=AUDIO_DIR
# )

# class ProcessingResponse(BaseModel):
#     message: str
#     task_id: str

# class ProcessingResult(BaseModel):
#     title: str
#     authors: List[str]
#     abstract: str
#     summary: str
#     topics: List[str]
#     summary_audio: Optional[str]
#     podcast_audio: Optional[str]
#     podcast_script: Optional[str]

# # Store background tasks
# processing_tasks = {}

# @router.post("/upload", response_model=ProcessingResponse)
# async def upload_paper(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
# ):
#     """Upload and process a research paper"""
#     # Validate file
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     # Create a unique ID for this processing task
#     task_id = str(uuid.uuid4())
    
#     # Create a unique filename
#     unique_filename = f"{task_id}_{file.filename}"
#     file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
#     # Save the uploaded file
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # Process the paper in the background
#     async def process_paper_task():
#         result = await paper_system.process_paper_file(file_path)
#         processing_tasks[task_id] = result
    
#     background_tasks.add_task(process_paper_task)
    
#     return ProcessingResponse(
#         message="Paper uploaded and processing started",
#         task_id=task_id
#     )

# @router.get("/status/{task_id}", response_model=Optional[ProcessingResult])
# async def get_processing_status(task_id: str):
#     """Get the status of a paper processing task"""
#     if task_id not in processing_tasks:
#         raise HTTPException(status_code=404, detail="Task not found")
    
#     result = processing_tasks[task_id]
#     if isinstance(result, dict) and "error" in result:
#         raise HTTPException(status_code=500, detail=result["error"])
    
#     return result

# @router.get("/audio/{file_name}")
# async def get_audio_file(file_name: str):
#     """Get an audio file"""
#     file_path = os.path.join(AUDIO_DIR, file_name)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="Audio file not found")
    
#     return FileResponse(file_path, media_type="audio/mpeg")

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from services.paper_processor_service import SinglePaperSystem
from app.config import settings
import os
from typing import List, Dict, Any

router = APIRouter()

# Initialize the paper processor system
paper_system = SinglePaperSystem()

@router.post("/process-paper")
async def process_paper(file: UploadFile = File(...)):
    """
    Upload and process a single research paper.
    Returns summary, audio files, and podcast information.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Process the paper
        paper = await paper_system.process_paper(file)
        
        if not paper:
            raise HTTPException(status_code=500, detail="Failed to process paper")
        
        # Return the results
        response_data = {
            "title": paper.title,
            "authors": paper.authors,
            "summary": paper.summary,
            "topics": paper.topics,
            "summary_audio_url": f"/processor/audio/{os.path.basename(paper.summary_audio_path)}" if paper.summary_audio_path else None,
            "podcast_audio_url": f"/processor/audio/{os.path.basename(paper.podcast_audio_path)}" if paper.podcast_audio_path else None,
            "podcast_script": paper.podcast_script
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing paper: {str(e)}")

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Retrieve generated audio files
    """
    audio_dir = settings.AUDIO_OUTPUT_DIR
    file_path = f"{audio_dir}/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(file_path, media_type="audio/mpeg")