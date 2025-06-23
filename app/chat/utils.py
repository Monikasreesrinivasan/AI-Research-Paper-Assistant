from typing import List, Dict, Any
from services.faiss_service import FaissService
from services.ollama_service import OllamaService
from database.db import chat_collection
from datetime import datetime
from bson import ObjectId

async def retrieve_context(
    paper_id: str,
    query: str,
    k: int = 3
) -> List[Dict[str, Any]]:
    """Retrieve relevant context from the vector store."""
    faiss_service = FaissService()
    
    # First search for all relevant chunks
    chunks = await faiss_service.search_similar_chunks(query, k=k*3)  # Get more chunks initially
    
    # Filter chunks by paper_id if provided
    if paper_id:
        filtered_chunks = [chunk for chunk in chunks if chunk.get("paper_id") == paper_id]
        # If we have enough filtered chunks, use them
        if len(filtered_chunks) >= k:
            return filtered_chunks[:k]
        # If we don't have enough filtered chunks but have some, use what we have
        elif filtered_chunks:
            return filtered_chunks
    
    # If paper_id filtering returned nothing or no paper_id provided, fall back to all chunks
    return chunks[:k]

async def generate_answer(
    query: str,
    context: List[Dict[str, Any]]
) -> str:
    """Generate an answer using the Ollama model."""
    ollama_service = OllamaService()
    return await ollama_service.generate_response(query, context)

async def save_chat_message(
    user_id: str,
    paper_id: str,
    message: str,
    is_user: bool,
    context_ids: List[str] = None
) -> Dict[str, Any]:
    """Save a chat message to the database."""
    message_data = {
        "user_id": user_id,
        "paper_id": paper_id,
        "message": message,
        "is_user": is_user,
        "context_ids": context_ids or [],
        "created_at": datetime.now()
    }
    
    result = await chat_collection.insert_one(message_data)
    message_data["id"] = str(result.inserted_id)
    
    return message_data

async def get_chat_history(
    user_id: str,
    paper_id: str = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get chat history for a user, optionally filtered by paper_id."""
    query = {"user_id": user_id}
    if paper_id:
        query["paper_id"] = paper_id
    
    messages = await chat_collection.find(query).sort("created_at", -1).limit(limit).to_list(None)
    messages.reverse()  # Return in chronological order
    
    return [
        {
            "id": str(msg["_id"]),
            "user_id": msg["user_id"],
            "paper_id": msg["paper_id"],
            "message": msg["message"],
            "is_user": msg["is_user"],
            "context_ids": msg.get("context_ids", []),
            "created_at": msg["created_at"]
        }
        for msg in messages
    ]
