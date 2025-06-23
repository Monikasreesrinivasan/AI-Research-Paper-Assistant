import os
import uuid
from typing import List, Dict
from PyPDF2 import PdfReader

class PDFService:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self.sessions: Dict[str, PDFSession] = {}
        # Create uploads directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        
    def create_session(self, filename: str, file_content: bytes) -> str:
        """Create a new PDF session and process the uploaded file"""
        # Generate a unique ID for this session
        pdf_id = str(uuid.uuid4())
        
        # Save the uploaded file
        file_path = os.path.join(self.upload_dir, f"{pdf_id}.pdf")
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Create a new session
        session = PDFSession()
        session.filename = filename
        
        # Process the PDF
        try:
            text = session.extract_text_from_pdf(file_path)
            
            # Store text chunks
            session.store_text(text)
            
            # Store the session
            self.sessions[pdf_id] = session
            
            return pdf_id
        
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def get_session(self, pdf_id: str) -> 'PDFSession':
        """Get a PDF session by ID"""
        if pdf_id not in self.sessions:
            raise KeyError(f"PDF session not found: {pdf_id}")
        return self.sessions[pdf_id]
    
    def get_active_pdfs(self) -> List[Dict[str, str]]:
        """Get a list of all active PDF sessions"""
        return [{"pdf_id": pdf_id, "filename": session.filename} 
                for pdf_id, session in self.sessions.items()]


class PDFSession:
    def __init__(self):
        self.text_chunks = []
        self.filename = ""

    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text

    def store_text(self, text):
        # Simple text chunking by paragraphs
        self.text_chunks = [chunk for chunk in text.split("\n\n") if chunk.strip()]

    def simple_search(self, query, top_k=3):
        # Simple keyword matching (not as good as embeddings, but no dependencies)
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.text_chunks:
            chunk_words = set(chunk.lower().split())
            # Calculate word overlap
            score = len(query_words.intersection(chunk_words))
            scored_chunks.append((score, chunk))
        
        # Sort by score (higher is better)
        scored_chunks.sort(reverse=True)
        
        # Return top k chunks
        return [chunk for _, chunk in scored_chunks[:top_k]]