import requests
import pandas as pd
import datetime
import time
from typing import List, Dict, Any, Optional, Union
import xml.etree.ElementTree as ET
import httpx
import asyncio


class ResearchArticleSearch:
    """A tool to search for research articles across multiple sources."""
    
    def __init__(self):
        """Initialize the research article search tool."""
        self.sources = {
            "arxiv": self._search_arxiv,
            "pubmed": self._search_pubmed,
            "core": self._search_core_api
        }
        # Used to store the last search results
        self.last_results = []
        
    async def search(self, query: str, 
               sources: List[str] = ["arxiv", "pubmed", "core"],
               max_results: int = 20,
               sort_by: str = "relevance",  # Options: "relevance", "date"
               date_from: Optional[str] = None,  # Format: YYYY-MM-DD
               date_to: Optional[str] = None,    # Format: YYYY-MM-DD
               full_text: bool = False) -> List[Dict]:
        """
        Search for research articles based on query and filters.
        
        Args:
            query: Search query string
            sources: List of sources to search (arxiv, pubmed, core)
            max_results: Maximum number of results to return per source
            sort_by: Sort results by "relevance" or "date"
            date_from: Only include results published after this date
            date_to: Only include results published before this date
            full_text: Whether to search in full text (when available)
            
        Returns:
            List of dictionaries containing search results
        """
        if not query:
            raise ValueError("Search query cannot be empty")
            
        results = []
        
        # Validate sources
        valid_sources = [s for s in sources if s in self.sources]
        if not valid_sources:
            raise ValueError(f"No valid sources specified. Available sources: {list(self.sources.keys())}")
            
        # Search each source
        for source in valid_sources:
            try:
                source_results = await self.sources[source](
                    query=query,
                    max_results=max_results,
                    sort_by=sort_by,
                    date_from=date_from,
                    date_to=date_to,
                    full_text=full_text
                )
                results.extend(source_results)
                
                # Add a small delay to avoid API rate limits
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error searching {source}: {str(e)}")
        
        # Apply sorting
        if sort_by == "date":
            results.sort(key=lambda x: x.get("published_date", ""), reverse=True)
        
        # Store results for later reference
        self.last_results = results
        
        return results
    
    async def _search_arxiv(self, query: str, max_results: int = 20, 
                     sort_by: str = "relevance", date_from: Optional[str] = None, 
                     date_to: Optional[str] = None, full_text: bool = False) -> List[Dict]:
        """Search for articles on arXiv."""
        base_url = "http://export.arxiv.org/api/query"
        
        # Convert sort parameter to arXiv format
        sort_map = {
            "relevance": "relevance",
            "date": "submittedDate"
        }
        
        arxiv_sort = sort_map.get(sort_by, "relevance")
        
        # Build date filter if specified
        date_filter = ""
        if date_from or date_to:
            date_parts = []
            if date_from:
                date_parts.append(f"submittedDate:[{date_from} TO")
            else:
                date_parts.append("submittedDate:[* TO")
                
            if date_to:
                date_parts.append(f"{date_to}]")
            else:
                date_parts.append("*]")
                
            date_filter = " ".join(date_parts)
            query = f"{query} AND {date_filter}"
        
        params = {
            "search_query": f"all:{query}" if full_text else query,
            "start": 0,
            "max_results": max_results,
            "sortBy": arxiv_sort,
            "sortOrder": "descending"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"arXiv API returned status code {response.status_code}")
        
        # Parse XML response
        root = ET.fromstring(response.text)
        
        # Define namespace
        ns = {"atom": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        
        results = []
        for entry in root.findall(".//atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip()
            abstract_elem = entry.find("atom:summary", ns)
            abstract = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else "No abstract available"
            published_elem = entry.find("atom:published", ns)
            published = published_elem.text[:10] if published_elem is not None and published_elem.text else ""  # YYYY-MM-DD
            
            # Get authors
            authors = []
            for author in entry.findall(".//atom:author/atom:name", ns):
                if author.text:
                    authors.append(author.text)
            
            # Get link
            link = ""
            for link_tag in entry.findall("atom:link", ns):
                if link_tag.get("title") == "pdf":
                    link = link_tag.get("href")
                    break
            
            if not link and entry.find("atom:id", ns) is not None:
                link = entry.find("atom:id", ns).text
            
            results.append({
                "title": title,
                "authors": ", ".join(authors),
                "abstract": abstract,
                "published_date": published,
                "url": link,
                "source": "arXiv"
            })
        
        return results
    
    async def _search_pubmed(self, query: str, max_results: int = 20, 
                      sort_by: str = "relevance", date_from: Optional[str] = None, 
                      date_to: Optional[str] = None, full_text: bool = False) -> List[Dict]:
        """Search for articles on PubMed."""
        # PubMed E-utilities base URL
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        
        # Prepare date filters
        date_filter = ""
        if date_from:
            date_filter += f" AND {date_from}[PDAT]:"
            if date_to:
                date_filter += f"{date_to}[PDAT]"
            else:
                date_filter += "3000[PDAT]"
        elif date_to:
            date_filter += f" AND 1900[PDAT]:{date_to}[PDAT]"
        
        # Prepare search parameters
        search_params = {
            "db": "pubmed",
            "term": query + date_filter,
            "retmax": max_results,
            "sort": "pub+date" if sort_by == "date" else "relevance",
            "retmode": "json"
        }
        
        # Search for IDs
        async with httpx.AsyncClient() as client:
            response = await client.get(esearch_url, params=search_params)
        
        if response.status_code != 200:
            raise Exception(f"PubMed API returned status code {response.status_code}")
        
        search_data = response.json()
        
        if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
            return []
        
        id_list = search_data["esearchresult"]["idlist"]
        
        if not id_list:
            return []
        
        # Fetch details for the IDs
        summary_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        
        async with httpx.AsyncClient() as client:
            summary_response = await client.get(esummary_url, params=summary_params)
        
        if summary_response.status_code != 200:
            raise Exception(f"PubMed summary API returned status code {summary_response.status_code}")
        
        summary_data = summary_response.json()
        
        results = []
        if "result" in summary_data:
            for pmid in id_list:
                if pmid in summary_data["result"]:
                    article = summary_data["result"][pmid]
                    
                    # Extract authors
                    authors = []
                    if "authors" in article:
                        for author in article["authors"]:
                            if "name" in author:
                                authors.append(author["name"])
                    
                    # Extract date
                    pub_date = ""
                    if "pubdate" in article:
                        pub_date = article["pubdate"]
                        # Convert date format if needed
                        if len(pub_date) >= 4:  # At least year is present
                            try:
                                # Handle different date formats
                                if len(pub_date) >= 10:
                                    date_obj = datetime.datetime.strptime(pub_date[:10], "%Y/%m/%d")
                                    pub_date = date_obj.strftime("%Y-%m-%d")
                                elif len(pub_date) >= 7:
                                    date_obj = datetime.datetime.strptime(pub_date[:7], "%Y/%m")
                                    pub_date = date_obj.strftime("%Y-%m")
                                elif len(pub_date) >= 4:
                                    pub_date = pub_date[:4]
                            except ValueError:
                                # Keep original format if parsing fails
                                pass
                    
                    results.append({
                        "title": article.get("title", ""),
                        "authors": ", ".join(authors),
                        "abstract": article.get("abstract", "No abstract available"),
                        "published_date": pub_date,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "source": "PubMed"
                    })
        
        return results
    
    async def _search_core_api(self, query: str, max_results: int = 20, 
                        sort_by: str = "relevance", date_from: Optional[str] = None, 
                        date_to: Optional[str] = None, full_text: bool = False) -> List[Dict]:
        """
        Search for articles using CORE API.
        Note: This uses CORE's free API which has limited functionality.
        """
        base_url = "https://api.core.ac.uk/v3/search/works"
        
        # CORE requires an API key but offers limited functionality without one
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare query with filters
        filters = []
        
        if date_from or date_to:
            year_filter = {}
            if date_from:
                year_filter["gte"] = date_from[:4]  # Extract year
            if date_to:
                year_filter["lte"] = date_to[:4]    # Extract year
            
            if year_filter:
                filters.append({"year": year_filter})
        
        search_body = {
            "q": query,
            "limit": max_results,
            "scroll": True
        }
        
        if filters:
            search_body["filters"] = filters
        
        # Set sort method
        if sort_by == "date":
            search_body["sort"] = "datePublished:desc"
        else:
            search_body["sort"] = "relevance"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(base_url, headers=headers, json=search_body)
            
            if response.status_code != 200:
                # If API key is required or rate limited
                if response.status_code == 403 or response.status_code == 429:
                    print("Note: CORE API has limited access without an API key.")
                    return []
                raise Exception(f"CORE API returned status code {response.status_code}")
            
            data = response.json()
            
            results = []
            if "results" in data:
                for article in data["results"]:
                    authors = []
                    if "authors" in article:
                        for author in article["authors"]:
                            if isinstance(author, dict) and "name" in author:
                                authors.append(author["name"])
                            elif isinstance(author, str):
                                authors.append(author)
                    
                    # Handle publication date
                    pub_date = ""
                    if "datePublished" in article:
                        pub_date = article["datePublished"]
                    elif "year" in article:
                        pub_date = str(article["year"])
                    
                    abstract = article.get("abstract", "No abstract available")
                    
                    results.append({
                        "title": article.get("title", ""),
                        "authors": ", ".join(authors),
                        "abstract": abstract if abstract else "No abstract available",
                        "published_date": pub_date,
                        "url": article.get("downloadUrl", article.get("identifiers", [""])[0] if "identifiers" in article else ""),
                        "source": "CORE"
                    })
            
            return results
        except Exception as e:
            print(f"Error with CORE API: {str(e)}")
            return []


# Create singleton instance to be used by the API
search_service = ResearchArticleSearch()