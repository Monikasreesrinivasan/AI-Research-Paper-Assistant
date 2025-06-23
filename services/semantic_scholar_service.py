import aiohttp
import json
from typing import List, Dict, Any
from app.config import settings

class SemanticScholarService:
    def __init__(self, api_key: str = settings.SEMANTIC_SCHOLAR_API_KEY, api_url: str = settings.SEMANTIC_SCHOLAR_URL):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {"x-api-key": api_key} if api_key else {}
    
    async def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for papers using the Semantic Scholar API."""
        endpoint = f"{self.api_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,authors,abstract,url,year"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params, headers=self.headers) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                papers = data.get("data", [])
                
                # Format the response
                result = []
                for paper in papers:
                    authors = [author.get("name", "") for author in paper.get("authors", [])]
                    result.append({
                        "paper_id": paper.get("paperId", ""),
                        "title": paper.get("title", ""),
                        "authors": authors,
                        "abstract": paper.get("abstract", ""),
                        "url": paper.get("url", ""),
                        "year": paper.get("year"),
                        "score": 0.0,  # Will be updated when using for recommendations
                        "source": "semantic_scholar"
                    })
                
                return result
    
    async def find_similar_papers(self, title: str, abstract: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar papers based on title and abstract."""
        # Create a query from the title and abstract
        query = f"{title} {abstract[:300]}"  # Limit abstract to avoid very long queries
        
        return await self.search_papers(query, limit)
    
    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a paper."""
        endpoint = f"{self.api_url}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,authors,abstract,url,year,citationCount,references,citations"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params, headers=self.headers) as response:
                if response.status != 200:
                    return {}
                
                data = await response.json()
                authors = [author.get("name", "") for author in data.get("authors", [])]
                
                return {
                    "paper_id": data.get("paperId", ""),
                    "title": data.get("title", ""),
                    "authors": authors,
                    "abstract": data.get("abstract", ""),
                    "url": data.get("url", ""),
                    "year": data.get("year"),
                    "citation_count": data.get("citationCount", 0),
                    "source": "semantic_scholar"
                }