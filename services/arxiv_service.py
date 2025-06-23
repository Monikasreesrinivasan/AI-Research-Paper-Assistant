import arxiv
import asyncio
from typing import List, Dict, Any

class ArxivService:
    def __init__(self):
        self.client = arxiv.Client()
    
    async def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on arXiv."""
        # Use run_in_executor to run the synchronous arxiv API in a thread pool
        loop = asyncio.get_event_loop()
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = await loop.run_in_executor(None, lambda: list(self.client.results(search)))
        
        # Format the response
        formatted_results = []
        for paper in results:
            formatted_results.append({
                "paper_id": paper.entry_id.split("/")[-1],
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "url": paper.pdf_url,
                "published": paper.published,
                "score": 0.0,  # Will be updated when using for recommendations
                "source": "arxiv"
            })
        
        return formatted_results
    
    async def find_similar_papers(self, title: str, abstract: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar papers based on title and abstract."""
        # Create a query from the title and abstract
        query = f"ti:\"{title}\" OR abs:\"{abstract[:300]}\""  # Limit abstract to avoid very long queries
        
        return await self.search_papers(query, limit)
    
    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a paper."""
        loop = asyncio.get_event_loop()
        search = arxiv.Search(
            id_list=[paper_id],
            max_results=1
        )
        
        results = await loop.run_in_executor(None, lambda: list(self.client.results(search)))
        
        if not results:
            return {}
        
        paper = results[0]
        return {
            "paper_id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "url": paper.pdf_url,
            "published": paper.published,
            "source": "arxiv"
        }
