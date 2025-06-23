from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import os
import tempfile
from app.config import settings
from app.plagiarism.utils import PlagiarismDetector

router = APIRouter()

@router.post("/check")
async def check_plagiarism(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Check a PDF document for plagiarism.
    Returns plagiarism score and most similar resources.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temporary file to process
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_filename = temp_file.name
    temp_file.close()  # Close the file immediately after getting its name
    
    try:
        # Save uploaded content to temp file
        contents = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(contents)
        
        # Process the file
        detector = PlagiarismDetector()
        results = detector.detect_plagiarism(temp_filename)
        
        # Extract only required information (plagiarism score and similar resources)
        response = {
            "plagiarism_score": results.get("plagiarism_score", 0),
            "similar_resources": results.get("overall_similarity", [])[:5]  # Get top 5 similar resources
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temp file - ensure we're using the filename, not the file object
        # Add try-except to prevent errors if file deletion fails
        try:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        except Exception as e:
            # Log the error but don't fail the request
            print(f"Error removing temporary file: {e}")