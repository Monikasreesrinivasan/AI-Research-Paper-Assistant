import os
import shutil
from typing import Dict, List, Tuple
from fastapi import UploadFile, HTTPException, status
from app.config import settings
from utils.pdf_extractor import process_pdf
from services.faiss_service import FaissService
from database.db import paper_collection
from datetime import datetime
from bson import ObjectId

async def save_uploaded_file(file: UploadFile, user_id: str) -> str:
    """Save uploaded file to disk and return the path."""
    # Create user directory if it doesn't exist
    user_dir = os.path.join(settings.UPLOAD_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate file path
    file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = os.path.join(user_dir, file_name)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path

async def process_paper(file_path: str, user_id: str) -> Dict:
    """Process a paper file and store it in the database and vector store."""
    
    # Extract metadata and text chunks from PDF
    metadata, chunks = process_pdf(file_path)
    
    # Create paper entry in database
    paper_data = {
        "title": metadata.get("title", "Unknown Title"),
        "authors": metadata.get("authors", []),
        "abstract": metadata.get("abstract", ""),
        "content": "\n".join(chunks),  # Store full content for reference
        "file_path": file_path,
        "user_id": user_id,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Insert paper into the database
    result = await paper_collection.insert_one(paper_data)
    paper_id = str(result.inserted_id)
    
    # Add paper chunks to vector store
    faiss_service = FaissService()
    chunk_ids = await faiss_service.add_document(
        paper_id=paper_id,
        chunks=chunks,
        metadata={
            "title": metadata.get("title", "Unknown Title"),
            "authors": metadata.get("authors", []),
            "abstract": metadata.get("abstract", ""),
            "paper_id": paper_id  # ✅ Ensure paper_id is included in metadata
        }
    )
    
    # Update paper with vector_id (using the first chunk id as reference)
    await paper_collection.update_one(
        {"_id": ObjectId(paper_id)},
        {"$set": {"vector_id": chunk_ids[0] if chunk_ids else None}}
    )
    
    # Add the paper ID to the returned paper data
    paper_data["id"] = paper_id
    
    # Return the complete paper data
    return paper_data
