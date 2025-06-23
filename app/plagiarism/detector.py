import os
import re
import tempfile
import numpy as np
from typing import List, Dict, Tuple, Any
import nltk
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import fitz  # PyMuPDF

# Ensure NLTK resources are available
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
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.document_text = ""
        self.sections = []
        self.paragraphs = []
        self.sources = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            full_text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    full_text += page.get_text() + "\n\n"
                
            if not full_text.strip():
                return ""
                    
            # Store the full document text
            self.document_text = full_text
            return full_text
                
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
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
            print(f"Error processing web result {url}: {e}")
        
        return None

    def search_web(self, keywords: List[str], num_results: int = 5) -> List[Dict]:
        """
        Search general web for related content - using an API service
        In a production environment, you would integrate with a proper search API like:
        - Google Custom Search API
        - Bing Search API
        - Serpapi
        """
        # For demonstration, we'll return mock results
        # In production, replace with actual API call
        
        search_query = " ".join(keywords[:5])
        mock_results = [
            {
                'title': f"Research about {keywords[0]} and {keywords[1]}",
                'abstract': f"This paper discusses {search_query} in the context of academic research.",
                'url': f"https://example.com/research/{keywords[0]}",
                'year': '2024',
                'source_type': 'web'
            },
            {
                'title': f"Latest developments in {keywords[0]}",
                'abstract': f"Recent advances in {search_query} have shown promising results.",
                'url': f"https://scholar.example.org/paper/{keywords[0]}",
                'year': '2023',
                'source_type': 'web'
            },
            {
                'title': f"Analysis of {keywords[2]} in scientific literature",
                'abstract': f"This comprehensive review covers {search_query} and related topics.",
                'url': f"https://repository.example.edu/{keywords[2]}",
                'year': '2024',
                'source_type': 'web'
            }
        ]
        
        return mock_results[:num_results]
    
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
            print(f"Error calculating transformer similarity: {e}")
            return 0.0
    
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
            'most_similar_sources': []
        }
        
        # Calculate document-level similarity with each source
        doc_similarities = []
        
        for source in all_sources:
            similarity = self.calculate_transformer_similarity(clean_text, source['abstract'])
            doc_similarities.append({
                'title': source['title'],
                'similarity': similarity,
                'url': source['url'],
                'source_type': source['source_type']
            })
        
        # Sort by similarity (highest first)
        doc_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        results['overall_similarity'] = doc_similarities
        
        # Select top 3 most similar sources
        results['most_similar_sources'] = doc_similarities[:3]
        
        # Calculate overall plagiarism score (weighted average)
        max_overall_sim = max([s['similarity'] for s in doc_similarities]) if doc_similarities else 0
        
        # Calculate weighted score (simplified for this implementation)
        plagiarism_score = max_overall_sim * 100
        
        # Scale to percentage and round to integer
        results['plagiarism_score'] = round(plagiarism_score)
        
        return results