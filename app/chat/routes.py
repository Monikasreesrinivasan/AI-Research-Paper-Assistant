from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Dict, Any, Optional
from app.models.user import User
from app.auth.utils import get_current_user
from app.chat.utils import retrieve_context, generate_answer, save_chat_message, get_chat_history
from database.db import paper_collection
from bson import ObjectId
from pydantic import BaseModel

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    paper_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    context: List[Dict[str, Any]] = []

@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_message: ChatMessage,
    current_user: User = Depends(get_current_user)
):
    """Send a message to the chat system."""
    # Verify paper exists if paper_id is provided
    paper_title = None
    if chat_message.paper_id:
        paper = await paper_collection.find_one({
            "_id": ObjectId(chat_message.paper_id),
            "user_id": current_user.id
        })
        if not paper:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Paper not found"
            )
        paper_title = paper.get("title")
    
    # Retrieve relevant context
    context = await retrieve_context(chat_message.paper_id, chat_message.message)
    
    # Save user message
    await save_chat_message(
        user_id=current_user.id,
        paper_id=chat_message.paper_id,
        message=chat_message.message,
        is_user=True
    )
    
    # Generate answer
    answer = await generate_answer(chat_message.message, context)
    
    # If no context was found but paper_id was provided, include a note about this
    if not context and chat_message.paper_id and paper_title:
        additional_context = [{
            "text": f"This query is about the paper titled: {paper_title}. " +
                   "However, I couldn't find specific relevant sections in the paper for this query."
        }]
        answer = await generate_answer(chat_message.message, additional_context)
    
    # Save assistant message
    context_ids = [str(c.get("id")) for c in context]
    await save_chat_message(
        user_id=current_user.id,
        paper_id=chat_message.paper_id,
        message=answer,
        is_user=False,
        context_ids=context_ids
    )
    
    return {
        "message": answer,
        "context": context
    }

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_messages(
    paper_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get chat history for the user."""
    messages = await get_chat_history(current_user.id, paper_id)
    return messages
