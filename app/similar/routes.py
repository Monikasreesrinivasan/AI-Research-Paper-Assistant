# Add to app/similar/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
import logging
from services.paper_finder_service import find_similar_paper_urls

router = APIRouter()
logger = logging.getLogger(__name__)

# Your existing routes here...

@router.post("/find-similar-papers")
async def find_similar_papers(file: UploadFile = File(...)):
    """
    Upload a PDF research paper and find similar papers from various academic sources.
    
    Returns a list of similar papers with their URLs, titles, and other metadata.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file contents
        contents = await file.read()
        pdf_file = BytesIO(contents)
        
        # Process the PDF and find similar papers
        result = find_similar_paper_urls(pdf_file)
        
        if not result["success"]:
            return JSONResponse(
                status_code=404,
                content=result
            )
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error processing request: {str(e)}",
                "results": []
            }
        )