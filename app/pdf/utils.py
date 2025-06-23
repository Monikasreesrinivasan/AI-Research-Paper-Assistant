from services.pdf_service import PDFService
from services.ollama_service2 import OllamaService

# Singleton instances
_pdf_service = None
_ollama_service = None

def get_pdf_service():
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service

def get_ollama_service():
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
    return _ollama_service