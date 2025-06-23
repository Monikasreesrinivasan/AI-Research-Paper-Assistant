import PyPDF2
import requests
import re
import time
from bs4 import BeautifulSoup
import urllib.parse
import random
import logging
import os
import pandas as pd
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_message(msg):
    logger.info(msg)
    return msg

# Extract text and title with improved error handling
def extract_text_and_title(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        full_text = ""
        for page_num in range(min(3, len(pdf_reader.pages))):
            full_text += pdf_reader.pages[page_num].extract_text()

        lines = full_text.strip().split('\n')

        title_candidate1 = lines[0].strip() if lines else ""
        title_candidate2 = ""
        for line in lines[:10]:
            if line.isupper() and len(line) > 15:
                title_candidate2 = line
                break

        title_pattern = re.compile(r'^[A-Z].{10,150}[.!?]$', re.MULTILINE)
        title_matches = title_pattern.findall(full_text[:2000])
        title_candidate3 = title_matches[0] if title_matches else ""

        candidates = [t for t in [title_candidate1, title_candidate2, title_candidate3]
                      if 10 < len(t) < 200]
        title = max(candidates, key=len) if candidates else title_candidate1
        log_message(f"Extracted title: {title}")
        return title, full_text

    except Exception as e:
        log_message(f"Error extracting text from PDF: {e}")
        return None, ""

# Get URLs from Semantic Scholar
def get_semantic_scholar_urls(title, limit=10):
    urls = []
    try:
        query = re.sub(r'\b(and|the|of|on|in|for|a|an)\b', '', title, flags=re.IGNORECASE)
        query = re.sub(r'\s+', ' ', query).strip()

        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": limit, "fields": "title,url,authors,venue,year"}
        headers = {"Accept": "application/json"}

        log_message(f"Searching Semantic Scholar for: {query}")
        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if "data" in data and data["data"]:
                for paper in data["data"]:
                    if "url" in paper and paper["url"]:
                        urls.append({
                            "title": paper.get("title", "Unknown Title"),
                            "url": paper["url"],
                            "source": "Semantic Scholar",
                            "year": paper.get("year", "N/A"),
                            "authors": ", ".join([author.get("name", "") for author in paper.get("authors", [])]),
                            "venue": paper.get("venue", "N/A")
                        })
        else:
            log_message(f"Semantic Scholar API returned status code: {response.status_code}")
    except Exception as e:
        log_message(f"Error querying Semantic Scholar API: {e}")
    return urls

# Direct search using Google Scholar and fallback to DuckDuckGo
def search_papers_direct(title, limit=2):
    urls = []
    try:
        query = f"{title} filetype:pdf"
        search_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
        }

        log_message(f"Attempting direct search for: {title}")
        response = requests.get(search_url, headers=headers)
        time.sleep(random.uniform(2, 5))  # Random delay to avoid blocking

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            result_count = 0
            for result in soup.select('.gs_r'):
                if result_count >= limit:
                    break

                title_elem = result.select_one('.gs_rt')
                link_elem = result.select_one('.gs_rt a')

                if title_elem and link_elem and link_elem.get('href'):
                    paper_url = link_elem.get('href')
                    paper_title = title_elem.get_text(strip=True)
                    urls.append({
                        "title": paper_title,
                        "url": paper_url,
                        "source": "Google Scholar",
                        "year": "N/A",
                        "authors": "N/A",
                        "venue": "N/A"
                    })
                    result_count += 1

        if not urls:
            log_message("No results found on Google Scholar. Trying DuckDuckGo...")
            duckduckgo_url = f"https://duckduckgo.com/html?q={urllib.parse.quote(query)}"
            response = requests.get(duckduckgo_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for result in soup.select('.result__url')[:limit]:
                    urls.append({
                        "title": title,
                        "url": result.get_text(),
                        "source": "DuckDuckGo",
                        "year": "N/A",
                        "authors": "N/A",
                        "venue": "N/A"
                    })
    except Exception as e:
        log_message(f"Error in direct search: {e}")
    return urls

# Search arXiv with improved queries
def search_arxiv(title, limit=2):
    urls = []
    try:
        query = re.sub(r'\b(and|the|of|on|in|for|a|an)\b', '', title, flags=re.IGNORECASE)
        query = re.sub(r'\s+', ' ', query).strip()

        base_url = "http://export.arxiv.org/api/query"
        params = {"search_query": f"title:{query}", "start": 0, "max_results": limit}

        log_message(f"Searching arXiv for: {query}")
        response = requests.get(base_url, params=params)
        time.sleep(random.uniform(1, 3))  # Prevent rate limiting

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'xml')
            entries = soup.find_all('entry')
            for entry in entries:
                paper_title = entry.find('title').text.strip()
                paper_url = entry.find('id').text.strip()
                published = entry.find('published').text.strip()
                year = published.split('-')[0] if '-' in published else "N/A"

                author_tags = entry.find_all('author')
                authors = ", ".join([author.find('name').text.strip() for author in author_tags])

                urls.append({
                    "title": paper_title,
                    "url": paper_url,
                    "source": "arXiv",
                    "year": year,
                    "authors": authors,
                    "venue": "arXiv"
                })
    except Exception as e:
        log_message(f"Error querying arXiv API: {e}")
    return urls

# Main function to find similar papers
def find_similar_paper_urls(pdf_file):
    title, full_text = extract_text_and_title(pdf_file)
    
    if not title:
        return {
            "success": False,
            "message": "Could not extract title from PDF",
            "results": []
        }
    
    all_urls = []
    log_message("\nMethod 1: Searching Semantic Scholar...")
    semantic_scholar_urls = get_semantic_scholar_urls(title)
    all_urls.extend(semantic_scholar_urls)

    if len(all_urls) < 5:
        log_message("\nMethod 2: Trying direct search...")
        direct_search_urls = search_papers_direct(title)
        all_urls.extend(direct_search_urls)

    if len(all_urls) < 5:
        log_message("\nMethod 3: Searching arXiv...")
        arxiv_urls = search_arxiv(title)
        all_urls.extend(arxiv_urls)

    # Remove duplicates using URL hashing
    seen_urls = set()
    unique_urls = [paper for paper in all_urls if paper["url"] not in seen_urls and not seen_urls.add(paper["url"])]

    # Limit to first 5 results
    unique_urls = unique_urls[:5]

    if not unique_urls:
        return {
            "success": False,
            "message": "No similar papers found",
            "results": []
        }
    
    return {
        "success": True,
        "message": f"Found {len(unique_urls)} similar papers",
        "paper_title": title,
        "results": unique_urls
    }