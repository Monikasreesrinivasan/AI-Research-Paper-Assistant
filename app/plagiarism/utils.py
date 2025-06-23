import os
import re
import pdfplumber
import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer, util
import concurrent.futures
from fastapi.logger import logger

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class PlagiarismDetector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the plagiarism detector with the specified model."""
        logger.info("Initializing plagiarism detection system...")
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.document_text = ""
        self.sections = []
        self.paragraphs = []
        self.sources = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber with error handling."""
        logger.info("Extracting text from PDF...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n\n"
                
                if not full_text.strip():
                    logger.warning("Warning: Extracted text is empty. The PDF may be scanned or protected.")
                    return ""
                    
                # Store the full document text
                self.document_text = full_text
                return full_text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence punctuation
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text.strip()
            
    def segment_document(self, text: str) -> Tuple[List[str], List[str]]:
        """Segment document into both sections and paragraphs."""
        logger.info("Segmenting document into sections and paragraphs...")
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0]
        paragraphs = [p for p in paragraphs if len(p.split()) >= 5]  # Filter out very short paragraphs
        
        # Try to identify sections based on common patterns in academic papers
        section_patterns = [
            r'(?:^|\n+)(?:\d+[\.\s]+)?(?:Introduction|Abstract|Background|Literature Review|Methodology|' +
            r'Methods|Results|Discussion|Conclusion|References|Bibliography|Appendix)(?:\s|\n|:)'
        ]
        
        # Find potential section boundaries
        section_starts = [0]  # Start with the beginning of the document
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                section_starts.append(match.start())
        
        # Sort and deduplicate section starts
        section_starts = sorted(set(section_starts))
        
        # Create sections from the boundaries
        sections = []
        for i in range(len(section_starts)):
            start = section_starts[i]
            end = section_starts[i+1] if i+1 < len(section_starts) else len(text)
            section_text = text[start:end].strip()
            if len(section_text.split()) >= 50:  # Only include substantial sections
                sections.append(section_text)
        
        # If no clear sections were found, use paragraphs as sections
        if len(sections) <= 1:
            logger.info("No clear section structure found. Using paragraph-based segmentation.")
            # Group paragraphs into pseudo-sections of approximately 500 words
            sections = []
            current_section = ""
            for paragraph in paragraphs:
                current_section += paragraph + "\n\n"
                if len(current_section.split()) >= 500:
                    sections.append(current_section.strip())
                    current_section = ""
            if current_section.strip():
                sections.append(current_section.strip())
        
        self.sections = sections
        self.paragraphs = paragraphs
        
        return sections, paragraphs
    
    def extract_keywords(self, text: str, n: int = 8) -> List[str]:
        """Extract the most important keywords from text."""
        # Tokenize and clean
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and 
                 word not in self.stop_words and 
                 len(word) > 3]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in keywords[:n]]
    
    def _process_web_result(self, url: str) -> Dict:
        """Process a single web search result - used for parallel processing."""
        try:
            response = requests.get(url, timeout=4)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else url
            
            # Extract content from paragraphs
            paragraphs = soup.find_all('p')
            content = "\n".join([p.get_text().strip() for p in paragraphs 
                                if len(p.get_text().strip()) > 40])
            
            if content:
                return {
                    'title': title,
                    'abstract': content[:2000],  # Limit length
                    'url': url,
                    'year': 'Unknown',
                    'source_type': 'web'
                }
        except Exception as e:
            logger.error(f"Error processing web result {url}: {e}")
        
        return None
    
    def web_search(self, query: str, num_results: int = 5) -> List[str]:
        """
        Simplified web search function that returns list of URLs
        This is a mock implementation - in a real scenario, you'd integrate with a search API
        """
        # Mock search results - in production, replace with actual search API
        # These are dummy URLs, in production you'd use a real search engine API
        base_urls = [
            "https://en.wikipedia.org/wiki/",
            "https://scholar.google.com/scholar?q=",
            "https://www.researchgate.net/search?q=",
            "https://www.sciencedirect.com/search?qs=",
            "https://arxiv.org/search/?query=",
            "https://www.nature.com/search?q=",
            "https://www.science.org/search?q=",
            "https://academic.oup.com/journals/search-results?q=",
        ]
        
        # Create mock URLs based on query
        search_term = query.replace(' ', '+')
        results = [f"{url}{search_term}" for url in base_urls[:num_results]]
        
        return results
        
    def search_web(self, keywords: List[str], num_results: int = 5) -> List[Dict]:
        """Search web for related content - optimized with parallel requests."""
        logger.info(f"Searching web for related content...")
        results = []
        
        try:
            search_query = " ".join(keywords[:5])
            urls = self.web_search(search_query, num_results=num_results)
            
            # Process URLs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_results) as executor:
                future_to_url = {executor.submit(self._process_web_result, url): url for url in urls}
                
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future.result()
                    if result:
                        results.append(result)
                
        except Exception as e:
            logger.error(f"Web search error: {e}")
            
        return results
    
    def calculate_transformer_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using transformer embeddings."""
        try:
            # Create embeddings
            emb1 = self.model.encode(text1, convert_to_tensor=True)
            emb2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            cos_sim = util.pytorch_cos_sim(emb1, emb2)
            return cos_sim.item()
        except Exception as e:
            logger.error(f"Error calculating transformer similarity: {e}")
            return 0.0
    
    def _calculate_section_similarity(self, section: str, source: Dict) -> Dict:
        """Calculate similarity between a section and a source - for parallel processing."""
        similarity = self.calculate_transformer_similarity(section, source['abstract'])
        if similarity > 0.5:  # Only record significant similarities
            return {
                'title': source['title'],
                'similarity': similarity,
                'url': source['url'],
                'source_type': source['source_type']
            }
        return None
    
    def detect_plagiarism(self, pdf_path: str) -> Dict:
        """Main method to detect plagiarism in the document."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "Failed to extract text from PDF."}
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Segment document
        self.segment_document(clean_text)
        
        # Extract keywords for search
        document_keywords = self.extract_keywords(clean_text)
        
        # Search for web sources
        logger.info("Searching for external sources...")
        web_results = self.search_web(document_keywords, num_results=5)
        all_sources = web_results
        
        # Store all sources for later reference
        self.sources = all_sources
        
        # Initialize results structure
        results = {
            'document_analysis': {
                'filename': os.path.basename(pdf_path),
                'total_length': len(clean_text),
                'sections': len(self.sections),
                'paragraphs': len(self.paragraphs)
            },
            'overall_similarity': [],
            'plagiarism_score': 0.0
        }
        
        # Calculate document-level similarity with each source - in parallel
        logger.info("Calculating similarities with external sources...")
        doc_similarities = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_source = {}
            for source in all_sources:
                future = executor.submit(self.calculate_transformer_similarity, clean_text, source['abstract'])
                future_to_source[future] = source
            
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    similarity = future.result()
                    doc_similarities.append({
                        'title': source['title'],
                        'similarity': similarity,
                        'url': source['url'],
                        'source_type': source['source_type']
                    })
                except Exception as e:
                    logger.error(f"Error calculating document similarity: {e}")
        
        # Sort by similarity (highest first)
        doc_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        results['overall_similarity'] = doc_similarities
        
        # Calculate overall plagiarism score
        # This is a weighted score based on similarity factors
        max_overall_sim = max([s['similarity'] for s in doc_similarities]) if doc_similarities else 0
        
        # Calculate section-level similarities for a more comprehensive score
        section_similarities = []
        for section in self.sections:
            section_results = []
            for source in all_sources:
                result = self._calculate_section_similarity(section, source)
                if result:
                    section_results.append(result)
            
            section_results.sort(key=lambda x: x['similarity'], reverse=True)
            if section_results:
                section_similarities.append(max(s['similarity'] for s in section_results))
        
        max_section_sim = max(section_similarities) if section_similarities else 0
        
        # Weight factors for overall score
        overall_sim_weight = 0.4
        section_sim_weight = 0.6
        
        # Calculate weighted score
        plagiarism_score = (max_overall_sim * overall_sim_weight) + (max_section_sim * section_sim_weight)
        
        # Scale to percentage
        results['plagiarism_score'] = plagiarism_score * 100
        
        return results