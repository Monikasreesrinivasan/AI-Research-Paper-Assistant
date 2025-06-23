from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import uuid
import tempfile
from app.config import settings
from app.auth.utils import get_current_user
from app.models.user import User
from .utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_paper_details,
    compare_papers
)

router = APIRouter()

@router.post("/upload-compare")
async def upload_compare_papers(
    background_tasks: BackgroundTasks,
    paper1: UploadFile = File(...),
    paper2: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and compare two research papers
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create temporary files for the uploaded papers
    temp_dir = os.path.join(settings.UPLOAD_DIR, "temp", task_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save uploaded files to temp directory
        paper1_path = os.path.join(temp_dir, paper1.filename)
        paper2_path = os.path.join(temp_dir, paper2.filename)
        
        # Save file 1
        with open(paper1_path, "wb") as f:
            f.write(await paper1.read())
        
        # Save file 2
        with open(paper2_path, "wb") as f:
            f.write(await paper2.read())
        
        # Add comparison task to background
        background_tasks.add_task(
            process_comparison,
            task_id,
            paper1_path,
            paper2_path, 
            current_user.id
        )
        
        return {"message": "Comparison started", "task_id": task_id}
    
    except Exception as e:
        # Clean up temp files in case of error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing papers: {str(e)}")

@router.get("/status/{task_id}")
async def get_comparison_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Check the status of a comparison task
    """
    # Check if results file exists
    results_path = os.path.join(settings.UPLOAD_DIR, "results", f"{task_id}.json")
    if os.path.exists(results_path):
        import json
        with open(results_path, "r") as f:
            results = json.load(f)
        return results
    
    # Check if error file exists
    error_path = os.path.join(settings.UPLOAD_DIR, "results", f"{task_id}_error.txt")
    if os.path.exists(error_path):
        with open(error_path, "r") as f:
            error_message = f.read()
        raise HTTPException(status_code=500, detail=error_message)
    
    # If neither exists, the job is still processing
    return {"status": "processing"}

# Background processing function
def process_comparison(task_id: str, paper1_path: str, paper2_path: str, user_id: str):
    """Process the paper comparison in the background"""
    import json
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(settings.UPLOAD_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Extract text from papers
        paper1_text = get_paper_text(paper1_path)
        paper2_text = get_paper_text(paper2_path)
        
        # Extract details from papers
        paper1_details = extract_paper_details(paper1_text)
        paper2_details = extract_paper_details(paper2_text)
        
        # Compare papers
        comparison_results = compare_papers(paper1_details, paper2_details)
        
        # Prepare results
        results = {
            "status": "completed",
            "paper1_title": paper1_details.get('title', 'Unknown'),
            "paper2_title": paper2_details.get('title', 'Unknown'),
            "comparison_results": comparison_results
        }
        
        # Save results
        results_path = os.path.join(results_dir, f"{task_id}.json")
        with open(results_path, "w") as f:
            json.dump(results, f)
            
    except Exception as e:
        # Save error
        error_path = os.path.join(results_dir, f"{task_id}_error.txt")
        with open(error_path, "w") as f:
            f.write(str(e))
    
    finally:
        # Clean up temp files
        import shutil
        temp_dir = os.path.join(settings.UPLOAD_DIR, "temp", task_id)
        shutil.rmtree(temp_dir, ignore_errors=True)

def get_paper_text(file_path):
    """Extract text based on file extension"""
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")