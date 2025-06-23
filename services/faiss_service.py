from typing import List, Dict, Any
from utils.vector_store import VectorStore
from app.config import settings

class FaissService:
    def __init__(self):
        self.vector_store = VectorStore()

    async def add_document(self, paper_id: str, chunks: List[str], metadata: Dict[str, Any]) -> List[str]:
        chunk_metadatas = []
        for i, chunk in enumerate(chunks):
        # Create a fresh copy of metadata for each chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["paper_id"] = paper_id
            chunk_metadata["text"] = chunk
            chunk_metadata["chunk_index"] = i  # Add chunk index for reference
            chunk_metadatas.append(chunk_metadata)
    
        return self.vector_store.add_texts(chunks, chunk_metadatas)

    async def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks that are similar to the query."""
        return self.vector_store.search(query, k)

    async def get_chunks_by_id(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get chunks by their IDs."""
        return self.vector_store.get_by_ids(chunk_ids)

    async def get_all_vectors(self) -> List[Dict[str, Any]]:
        """Retrieve all stored vectors and metadata."""
        return self.vector_store.get_all_vectors()
