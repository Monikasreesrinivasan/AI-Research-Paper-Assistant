import os
import fitz  # PyMuPDF
import docx
import requests
import json
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Functions for text extraction from files
def extract_text_from_pdf(file_path):
    """Extract text from PDF files using PyMuPDF"""
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX files using python-docx"""
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Improved text chunking using LangChain's RecursiveCharacterTextSplitter
def split_paper_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split paper text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Improved embedding function using BERT-based models
class BertEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, texts, batch_size=8):
        """Encode texts to embeddings"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and prepare input
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            # Get model output
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Mean pooling
            batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            all_embeddings.append(batch_embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)

# Function to interact with Hugging Face API with better error handling and rate limiting
def generate_response(prompt, max_tokens=1024, model="mistralai/Mistral-7B-Instruct-v0.2"):
    """Generate response from Hugging Face API with improved handling"""
    API_TOKEN = os.environ.get("HF_API_TOKEN", "YOUR_API_TOKEN")  # Use environment variable for security
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    
    # Improved prompt formatting for instruction models
    formatted_prompt = f"""<s>[INST] {prompt} [/INST]"""
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    # Improved rate limiting with exponential backoff
    max_retries = 5
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 429:  # Too Many Requests
                wait_time = min(retry_delay * (2 ** attempt), 60)
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error: Failed to generate response. Status code: {response.status_code}"
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"Error: {str(e)}"
    
    return "Failed to generate response after multiple attempts."

# Improved paper section extraction using structured prompts
def extract_paper_details(paper_text):
    """Extract key details from a research paper using improved prompts"""
    # Use a better model for extraction tasks
    extraction_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Truncate text to fit model context window while preserving key sections
    max_length = 12000
    truncated_text = paper_text[:max_length]
    
    # Create a more structured prompt for extraction
    sections_prompt = f"""
    Analyze the following research paper text and extract the following sections:
    1. TITLE: The complete title of the paper
    2. ABSTRACT: The full abstract section
    3. METHODOLOGY: A detailed summary of the methodology or approach
    4. RESULTS: A comprehensive summary of key results and findings
    5. CONCLUSIONS: The main conclusions and implications
    
    For each section, extract the exact text if present, or provide a concise summary if the section is spread throughout the paper.
    
    Here is the paper text:
    
    {truncated_text}
    
    Format your response as:
    TITLE: [extracted title]
    
    ABSTRACT: [extracted abstract]
    
    METHODOLOGY: [extracted methodology]
    
    RESULTS: [extracted results]
    
    CONCLUSIONS: [extracted conclusions]
    """
    
    # Extract all sections in one API call to maintain consistency
    extracted_sections = generate_response(sections_prompt, max_tokens=2048, model=extraction_model)
    
    # Parse the extracted sections
    sections = {}
    current_section = None
    section_text = ""
    
    for line in extracted_sections.split('\n'):
        if line.startswith('TITLE:'):
            current_section = 'title'
            section_text = line[6:].strip()
            sections[current_section] = section_text
            section_text = ""
        elif line.startswith('ABSTRACT:'):
            if current_section:
                sections[current_section] = section_text.strip()
            current_section = 'abstract'
            section_text = line[9:].strip()
        elif line.startswith('METHODOLOGY:'):
            if current_section:
                sections[current_section] = section_text.strip()
            current_section = 'methodology'
            section_text = line[12:].strip()
        elif line.startswith('RESULTS:'):
            if current_section:
                sections[current_section] = section_text.strip()
            current_section = 'results'
            section_text = line[8:].strip()
        elif line.startswith('CONCLUSIONS:'):
            if current_section:
                sections[current_section] = section_text.strip()
            current_section = 'conclusions'
            section_text = line[12:].strip()
        else:
            if current_section:
                section_text += " " + line.strip()
    
    # Add the last section
    if current_section and current_section not in sections:
        sections[current_section] = section_text.strip()
    
    # Ensure all required sections exist
    for section in ['title', 'abstract', 'methodology', 'results', 'conclusions']:
        if section not in sections or not sections[section]:
            sections[section] = f"No {section} found in the document."
    
    # Add full text
    sections['full_text'] = paper_text
    
    return sections

def compare_papers(paper1_details, paper2_details):
    """Compare two papers using improved embedding model and analysis"""
    print("Analyzing similarities and differences...")
    
    # Initialize the improved embedder
    embedder = BertEmbedder()  # Uses BAAI/bge-large-en-v1.5 by default
    
    # Create chunks from paper sections
    paper1_chunks = []
    paper2_chunks = []
    chunk_texts1 = []
    chunk_texts2 = []
    
    # Add key sections to chunks
    for section in ["abstract", "methodology", "results", "conclusions"]:
        section1_chunks = split_paper_into_chunks(paper1_details[section], chunk_size=512, chunk_overlap=100)
        section2_chunks = split_paper_into_chunks(paper2_details[section], chunk_size=512, chunk_overlap=100)
        
        # Add section prefix for better context
        prefixed_chunks1 = [f"{section.upper()}: {chunk}" for chunk in section1_chunks]
        prefixed_chunks2 = [f"{section.upper()}: {chunk}" for chunk in section2_chunks]
        
        chunk_texts1.extend(prefixed_chunks1)
        chunk_texts2.extend(prefixed_chunks2)
    
    # Create embeddings
    paper1_embeddings = embedder.encode(chunk_texts1)
    paper2_embeddings = embedder.encode(chunk_texts2)
    
    # Normalize embeddings for cosine similarity
    paper1_embeddings = paper1_embeddings / np.linalg.norm(paper1_embeddings, axis=1)[:, np.newaxis]
    paper2_embeddings = paper2_embeddings / np.linalg.norm(paper2_embeddings, axis=1)[:, np.newaxis]
    
    # Create FAISS index for efficient similarity search
    d = paper1_embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatIP(d)  # Inner product (equivalent to cosine similarity on normalized vectors)
    index.add(paper1_embeddings.astype('float32'))
    
    # Search for similar chunks between papers
    k = min(5, len(paper1_embeddings))  # Number of nearest neighbors to find
    similarities, indices = index.search(paper2_embeddings.astype('float32'), k)
    
    # Calculate average similarity
    avg_similarity = np.mean(similarities[:, 0])  # Average of top similarities
    
    # Find most similar and most different sections
    section_similarities = {}
    paper2_section_starts = {}
    paper2_section_ends = {}
    
    start_idx = 0
    for section in ["abstract", "methodology", "results", "conclusions"]:
        section2_chunks = split_paper_into_chunks(paper2_details[section], chunk_size=512, chunk_overlap=100)
        section_size = len(section2_chunks)
        paper2_section_starts[section] = start_idx
        paper2_section_ends[section] = start_idx + section_size
        start_idx += section_size
    
    # Calculate similarity by section
    for section, start_idx in paper2_section_starts.items():
        end_idx = paper2_section_ends[section]
        if end_idx > start_idx:  # Ensure section has chunks
            section_similarities[section] = np.mean(similarities[start_idx:end_idx, 0])
    
    # Extract most similar chunks for context
    most_similar_pairs = []
    similarity_threshold = 0.75
    
    for i in range(len(paper2_embeddings)):
        if similarities[i][0] > similarity_threshold:
            paper2_chunk = chunk_texts2[i]
            paper1_chunk = chunk_texts1[indices[i][0]]
            most_similar_pairs.append({
                "paper1_chunk": paper1_chunk,
                "paper2_chunk": paper2_chunk,
                "similarity": float(similarities[i][0])
            })
    
    # Sort by similarity (highest first)
    most_similar_pairs = sorted(most_similar_pairs, key=lambda x: x["similarity"], reverse=True)
    
    # Limit to top 5 most similar pairs
    most_similar_pairs = most_similar_pairs[:5]
    
    # Use a better model for the comparison analysis
    analysis_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Create enhanced comparison prompt with section-specific similarity scores
    comparison_prompt = f"""
    I have analyzed two research papers and need a detailed scholarly comparison.
    
    PAPER 1: {paper1_details['title']}
    
    PAPER 2: {paper2_details['title']}
    
    SIMILARITY METRICS:
    - Overall similarity score: {avg_similarity:.4f} (scale 0-1, higher means more similar)
    - Abstract similarity: {section_similarities.get('abstract', 'N/A'):.4f}
    - Methodology similarity: {section_similarities.get('methodology', 'N/A'):.4f}
    - Results similarity: {section_similarities.get('results', 'N/A'):.4f}
    - Conclusions similarity: {section_similarities.get('conclusions', 'N/A'):.4f}
    
    PAPER 1 ABSTRACT:
    {paper1_details['abstract']}
    
    PAPER 2 ABSTRACT:
    {paper2_details['abstract']}
    
    PAPER 1 METHODOLOGY SUMMARY:
    {paper1_details['methodology'][:500]}...
    
    PAPER 2 METHODOLOGY SUMMARY:
    {paper2_details['methodology'][:500]}...
    
    PAPER 1 RESULTS SUMMARY:
    {paper1_details['results'][:500]}...
    
    PAPER 2 RESULTS SUMMARY:
    {paper2_details['results'][:500]}...
    
    Based on this data, provide a comprehensive scholarly comparison with these sections:
    
    1. RESEARCH FOCUS COMPARISON: Compare the core research questions, objectives, and domains.
    
    2. METHODOLOGICAL APPROACH: Analyze similarities and differences in methodologies, frameworks, and techniques.
    
    3. FINDINGS AND RESULTS: Compare key findings, results, and their implications.
    
    4. SCHOLARLY CONTRIBUTION: Compare the significance and contributions to the field.
    
    5. SYNTHESIS: Conclude with an integrated analysis of how these papers relate to each other in the broader research landscape.
    
    Please provide specific details and maintain an objective, scholarly tone.
    """
    
    comparison_analysis = generate_response(comparison_prompt, max_tokens=2048, model=analysis_model)
    
    return {
        "vector_similarity": float(avg_similarity),
        "section_similarities": {k: float(v) for k, v in section_similarities.items()},
        "most_similar_pairs": most_similar_pairs,
        "comparison_analysis": comparison_analysis
    }