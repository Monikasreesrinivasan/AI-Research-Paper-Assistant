import os
import PyPDF2
import re
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_metadata(text: str) -> Dict:
    """Extract metadata like title and authors from the paper text."""
    # This is a simplified version, in a real-world scenario you might need
    # more sophisticated techniques or use a library like GROBID
    
    lines = text.split("\n")
    title = lines[0].strip() if lines else "Unknown Title"
    
    # Try to find authors (this is a naive approach)
    authors = []
    for i in range(1, min(5, len(lines))):
        if not lines[i].strip():
            continue
        if re.match(r".*@.*|abstract|keywords", lines[i].lower()):
            break
        authors.append(lines[i].strip())
    
    # Try to find abstract
    abstract = ""
    abstract_started = False
    for line in lines:
        line = line.strip().lower()
        if "abstract" in line and not abstract_started:
            abstract_started = True
            continue
        if abstract_started:
            if "keywords" in line or "introduction" in line:
                break
            abstract += line + " "
    
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract.strip()
    }

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            # Keep some overlap for context
            words = current_chunk.split()
            if len(words) > overlap:
                current_chunk = " ".join(words[-overlap:]) + " " + sentence + " "
            else:
                current_chunk = sentence + " "
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def process_pdf(file_path: str) -> Tuple[Dict, List[str]]:
    """Process a PDF file, extract metadata and text chunks."""
    text = extract_text_from_pdf(file_path)
    metadata = extract_metadata(text)
    chunks = chunk_text(text)
    return metadata, chunks