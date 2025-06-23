import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.config import settings


class VectorStore:
    def __init__(self, vector_db_path: str = settings.VECTOR_DB_PATH):
        self.vector_db_path = vector_db_path
        self.index_path = os.path.join(vector_db_path, "faiss_index.bin")
        self.metadata_path = os.path.join(vector_db_path, "metadata.pkl")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create directory if it doesn't exist
        os.makedirs(vector_db_path, exist_ok=True)

        # Load or create index
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            # Create a new index
            dimension = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []

        # Encode texts to vectors
        embeddings = self.model.encode(texts)

        # Add vectors to index
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Store metadata
        ids = list(range(len(self.metadata), len(self.metadata) + len(texts)))

        if metadatas:
            for i, metadata in zip(ids, metadatas):
                metadata["id"] = i
                self.metadata.append(metadata)
        else:
            for i, text in zip(ids, texts):
                self.metadata.append({"id": i, "text": text})

        # Save index and metadata
        self._save()

        return [str(i) for i in ids]

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the vector store for similar texts."""
        # Encode query to vector
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar vectors
        distances, indices = self.index.search(query_embedding, k)

        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(1.0 - distances[0][i])
                results.append(result)

        return results

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get items by their IDs."""
        results = []
        for id_str in ids:
            try:
                idx = int(id_str)
                if 0 <= idx < len(self.metadata):
                    results.append(self.metadata[idx])
            except ValueError:
                continue
        return results

    def get_all_vectors(self) -> List[Dict[str, Any]]:
        """Retrieve all stored vectors and their metadata."""
        results = []
        for i in range(len(self.metadata)):
            metadata = self.metadata[i]
            results.append(metadata)
        print(f"✅ Total Vectors in FAISS: {len(results)}")
        return results

    def _save(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
