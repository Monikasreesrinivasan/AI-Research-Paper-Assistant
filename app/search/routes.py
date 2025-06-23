from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional
from pydantic import BaseModel
import asyncio
import httpx

from .service import search_service

router = APIRouter()

class SearchFilters(BaseModel):
    """Request model for search filters"""
    query: str
    sources: List[str] = ["arxiv", "pubmed", "core"]
    max_results: int = 20
    sort_by: str = "relevance"  # Options: "relevance", "date"
    date_from: Optional[str] = None  # Format: YYYY-MM-DD
    date_to: Optional[str] = None  # Format: YYYY-MM-DD
    full_text: bool = False

class SearchResult(BaseModel):
    """Response model for search results"""
    title: str
    authors: str
    abstract: str
    published_date: str
    url: str
    source: str

@router.post("/", response_model=List[SearchResult])
async def search_articles(filters: SearchFilters):
    """
    Search for research articles across multiple sources
    
    - **query**: Search query string
    - **sources**: List of sources to search (arxiv, pubmed, core)
    - **max_results**: Maximum number of results to return per source
    - **sort_by**: Sort results by "relevance" or "date"
    - **date_from**: Only include results published after this date (YYYY-MM-DD)
    - **date_to**: Only include results published before this date (YYYY-MM-DD)
    - **full_text**: Whether to search in full text (when available)
    """
    try:
        results = await search_service.search(
            query=filters.query,
            sources=filters.sources,
            max_results=filters.max_results,
            sort_by=filters.sort_by,
            date_from=filters.date_from,
            date_to=filters.date_to,
            full_text=filters.full_text
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.get("/sources")
async def get_available_sources():
    """Get a list of available search sources"""
    return {
        "sources": list(search_service.sources.keys()),
        "descriptions": {
            "arxiv": "arXiv is a free distribution service and an open-access archive for scholarly articles in various fields.",
            "pubmed": "PubMed is a free search engine accessing primarily the MEDLINE database of references and abstracts on life sciences and biomedical topics.",
            "core": "CORE aggregates open access research outputs from repositories and journals worldwide."
        }
    }