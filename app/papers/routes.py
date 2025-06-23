from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from typing import List
from app.models.user import User
from app.models.paper import Paper, PaperInDB
from app.auth.utils import get_current_user
from app.papers.utils import save_uploaded_file, process_paper
from database.db import paper_collection
from bson import ObjectId

router = APIRouter()

@router.post("/upload", response_model=Paper)
async def upload_paper(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a research paper PDF."""
    # Check file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Save file
    file_path = await save_uploaded_file(file, current_user.id)
    
    # Process paper
    paper_data = await process_paper(file_path, current_user.id)
    
    return paper_data

@router.get("/list", response_model=List[Paper])
async def list_papers(current_user: User = Depends(get_current_user)):
    """List all papers uploaded by the user."""
    papers = await paper_collection.find({"user_id": current_user.id}).to_list(None)
    return [
        {
            "id": str(paper["_id"]),
            "title": paper["title"],
            "authors": paper["authors"],
            "abstract": paper.get("abstract", ""),
            "content": paper["content"],
            "file_path": paper["file_path"],
            "vector_id": paper.get("vector_id"),
            "user_id": paper["user_id"],
            "created_at": paper["created_at"],
            "updated_at": paper["updated_at"]
        }
        for paper in papers
    ]

@router.get("/{paper_id}", response_model=Paper)
async def get_paper(
    paper_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific paper by ID."""
    paper = await paper_collection.find_one({"_id": ObjectId(paper_id), "user_id": current_user.id})
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    return {
        "id": str(paper["_id"]),
        "title": paper["title"],
        "authors": paper["authors"],
        "abstract": paper.get("abstract", ""),
        "content": paper["content"],
        "file_path": paper["file_path"],
        "vector_id": paper.get("vector_id"),
        "user_id": paper["user_id"],
        "created_at": paper["created_at"],
        "updated_at": paper["updated_at"]
    }

@router.delete("/{paper_id}")
async def delete_paper(
    paper_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a paper."""
    paper = await paper_collection.find_one({"_id": ObjectId(paper_id), "user_id": current_user.id})
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    # Delete the file if it exists
    try:
        import os
        if os.path.exists(paper["file_path"]):
            os.remove(paper["file_path"])
    except Exception:
        pass  # Continue even if file deletion fails
    
    # Delete from database
    await paper_collection.delete_one({"_id": ObjectId(paper_id)})
    
    return {"message": "Paper deleted successfully"}


@router.get("/debug/{paper_id}")
async def debug_paper_chunks(
    paper_id: str,
    current_user: User = Depends(get_current_user)
):
    """Debug endpoint to check stored chunks for a paper."""
    # Verify paper exists and belongs to user
    paper = await paper_collection.find_one({
        "_id": ObjectId(paper_id),
        "user_id": current_user.id
    })
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    # Get paper title for querying
    title = paper.get("title", "")
    
    # Search for chunks related to this paper
    faiss_service = FaissService()
    chunks = await faiss_service.search_similar_chunks(title, k=10)
    
    # Filter for this paper's chunks
    paper_chunks = [chunk for chunk in chunks if chunk.get("paper_id") == paper_id]
    
    return {
        "paper_id": paper_id,
        "title": title,
        "chunks_found": len(paper_chunks),
        "chunks": paper_chunks
    }