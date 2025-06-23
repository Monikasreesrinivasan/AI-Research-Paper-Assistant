import aiohttp
import json
from typing import List, Dict, Any
from app.config import settings

class OllamaService:
    def __init__(self, api_url: str = settings.OLLAMA_API_URL, model: str = settings.OLLAMA_MODEL):
        self.api_url = api_url
        self.model = model
        self.generate_url = f"{api_url}/api/generate"
    
    async def generate_response(self, prompt: str, context: List[Dict[str, str]] = None) -> str:
        """Generate a response from Ollama model."""
        system_prompt = """You are a helpful AI research assistant. Provide accurate, informative responses based on the context provided. Follow these guidelines:

- Keep responses concise (around 150-250 words) unless specifically asked for more detail
- Base your answers ONLY on the information in the provided context
- If the context doesn't contain relevant information, clearly state: "I don't have enough information in the provided context to answer this question accurately"
- Present information in a structured format with clear organization
- Use a professional yet conversational tone
- Highlight key points or important information when appropriate
- Provide specific examples from the context when relevant to illustrate your points
- Avoid speculation and never fabricate information beyond what's in the context
- If asked for an opinion on a topic, clarify that you can only provide information based on the given context
- For technical questions, ensure explanations are clear and accessible"""
        formatted_context = ""
        if context:
            formatted_context = "Context information:\n"
            for item in context:
                if 'text' in item:
                    formatted_context += f"{item['text']}\n\n"
        
        full_prompt = f"{system_prompt}\n\n{formatted_context}\nQuestion: {prompt}\n\nAnswer:"
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }
            
            async with session.post(self.generate_url, json=payload) as response:
                if response.status != 200:
                    return f"Error: Failed to generate response. Status: {response.status}"
                
                response_data = await response.json()
                return response_data.get("response", "No response generated")
    
    async def check_health(self) -> bool:
        """Check if Ollama service is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/tags") as response:
                    return response.status == 200
        except Exception:
            return False
