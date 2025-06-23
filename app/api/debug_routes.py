from fastapi import APIRouter
from services.faiss_service import FaissService

router = APIRouter()

@router.get("/debug/faiss/vectors")
async def get_all_vectors():
    """API to get all FAISS vectors and metadata."""
    faiss_service = FaissService()
    vectors = await faiss_service.get_all_vectors()
    return {"total_vectors": len(vectors), "vectors": vectors[:5]}  # Show first 5
