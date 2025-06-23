from typing import List, Dict, Any
from services.semantic_scholar_service import SemanticScholarService
from services.arxiv_service import ArxivService
from database.db import paper_collection
from bson import ObjectId

async def find_similar_papers(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Find papers similar to the given paper using multiple sources."""
    # Get paper data
    paper = await paper_collection.find_one({"_id": ObjectId(paper_id)})
    if not paper:
        return []
    
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    
    # Initialize services
    semantic_scholar = SemanticScholarService()
    arxiv = ArxivService()
    
    # Search in both services
    semantic_results = await semantic_scholar.find_similar_papers(title, abstract, limit)
    arxiv_results = await arxiv.find_similar_papers(title, abstract, limit)
    
    # Combine results, ensuring no duplicates by title
    combined_results = []
    seen_titles = set()
    
    for result in semantic_results + arxiv_results:
        result_title = result.get("title", "").lower()
        if result_title and result_title not in seen_titles and result_title.lower() != title.lower():
            seen_titles.add(result_title)
            combined_results.append(result)
    
    # Sort by score
    combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return combined_results[:limit]
