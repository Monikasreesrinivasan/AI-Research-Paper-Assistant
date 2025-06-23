from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.paper_analysis_service import PaperAnalysisService
import io

router = APIRouter()
paper_analyzer = PaperAnalysisService()

@router.post("/analyze")
async def analyze_paper(file: UploadFile = File(...)):
    """
    Analyze a research paper to extract and summarize Future Work and Limitations sections,
    and suggest research directions.
    """
    try:
        # Check file extension
        allowed_extensions = ['.pdf', '.docx', '.txt']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Please upload a PDF, DOCX, or TXT file."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process the file
        analysis_results = paper_analyzer.analyze_paper(file_content, file.filename)
        
        # Return results
        return JSONResponse(content=analysis_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")