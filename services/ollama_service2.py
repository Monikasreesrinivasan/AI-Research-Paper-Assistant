import requests
from typing import Dict, Any, Optional

class OllamaService:
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434/api/generate"):
        self.ollama_url = ollama_url
        
    def generate_response(self, prompt: str, model: str = "mistral", stream: bool = False) -> str:
        """Generate a response from Ollama"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return response.json().get('response', 'No valid response from Ollama.')
            else:
                return f"Error: {response.status_code}, {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def answer_with_context(self, question: str, context: str, model: str = "mistral") -> str:
        """Answer a question based on the provided context"""
        prompt = f"Based on the following research document context, answer the question. If the answer is not in the context, say you don't know:\n\nContext: {context}\n\nQuestion: {question}"
        return self.generate_response(prompt, model)