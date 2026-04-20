"""
MongoDB Atlas Vector Search fallback implementation.

Provides vector search using MongoDB Atlas' native vector search capability
as an alternative or fallback to FAISS when FAISS is unavailable.

Requires: MongoDB Atlas tier with vector search enabled.
"""

import numpy as np
from typing import List, Tuple, Optional

import core.config as config
import core.database as database
from core.utils import setup_logging

logger = setup_logging()


class MongoDBVectorSearch:
    """MongoDB Atlas vector search wrapper for face embeddings."""
    
    def __init__(self, dimension: int = 512):
        """
        Initialize MongoDB vector search.
        
        Args:
            dimension: Embedding vector dimension
        """
        self.dimension = dimension
        self.collection_name = "embeddings"  # Collection for vector data
        self.is_available = False
        
        # Check if vector search is available
        self._check_availability()
    
    def _check_availability(self):
        """Check if MongoDB vector search is available."""
        try:
            db = database.get_db()
            
            # Try to list search indexes
            if hasattr(db, 'command'):
                try:
                    result = db.command('listSearchIndexes', self.collection_name)
                    self.is_available = True
                    logger.info("MongoDB Atlas Vector Search available")
                except Exception as exc:
                    logger.debug(f"Vector search not available: {exc}")
                    self.is_available = False
        except Exception as exc:
            logger.debug(f"MongoDB vector search check failed: {exc}")
            self.is_available = False
    
    def add(
        self,
        embeddings: np.ndarray,
        student_ids: List[str]
    ) -> bool:
        """
        Add embeddings to MongoDB collection.
        
        Args:
            embeddings: (N, D) numpy array
            student_ids: List of N student IDs
            
        Returns:
            True if successful
        """
        if not self.is_available:
            logger.warning("MongoDB vector search not available")
            return False
        
        try:
            # Normalize embeddings
            embeddings = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            
            db = database.get_db()
            collection = db[self.collection_name]
            
            # Prepare documents
            docs = []
            for student_id, embedding in zip(student_ids, embeddings):
                doc = {
                    "student_id": student_id,
                    "embedding": embedding.tolist(),
                }
                docs.append(doc)
            
            # Insert
            if docs:
                collection.insert_many(docs)
                logger.debug(f"Added {len(docs)} embeddings to MongoDB")
            
            return True
        
        except Exception as exc:
            logger.error(f"Failed to add embeddings to MongoDB: {exc}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        Search for k nearest neighbors using MongoDB Atlas Vector Search.
        
        Args:
            query_embedding: (D,) embedding vector
            k: Number of neighbors
            
        Returns:
            List of (student_id, distance, confidence) tuples
        """
        if not self.is_available:
            logger.warning("MongoDB vector search not available")
            return []
        
        try:
            # Normalize query
            query = np.asarray(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            
            db = database.get_db()
            collection = db[self.collection_name]
            
            # Use $search with KNN for vector search
            pipeline = [
                {
                    "$search": {
                        "cosmosSearch": {
                            "vector": query.tolist(),
                            "k": k
                        },
                        "returnStoredSource": True
                    }
                },
                {
                    "$project": {
                        "student_id": 1,
                        "similarity": {"$meta": "searchScore"},
                        "_id": 0
                    }
                },
                {
                    "$limit": k
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            
            # Convert to output format
            output = []
            for result in results:
                student_id = result.get("student_id")
                similarity = result.get("similarity", 0.0)
                
                # Convert similarity score to confidence
                # MongoDB cosine similarity is in [0, 1], where 1 = identical
                confidence = float(similarity)
                
                # Approximate distance (1 - cosine similarity)
                distance = 1.0 - confidence
                
                output.append((student_id, distance, confidence))
            
            return output
        
        except Exception as exc:
            logger.error(f"MongoDB vector search failed: {exc}")
            return []
    
    def clear(self) -> bool:
        """Clear collection."""
        try:
            db = database.get_db()
            db[self.collection_name].delete_many({})
            logger.debug("Cleared MongoDB embeddings collection")
            return True
        except Exception as exc:
            logger.error(f"Failed to clear collection: {exc}")
            return False
    
    def get_size(self) -> int:
        """Get collection size."""
        try:
            if not self.is_available:
                return 0
            
            db = database.get_db()
            return db[self.collection_name].count_documents({})
        except Exception as exc:
            logger.debug(f"Failed to get collection size: {exc}")
            return 0
    
    def is_ready(self) -> bool:
        """Check if ready for searches."""
        return self.is_available


# Global instance
_global_search: Optional[MongoDBVectorSearch] = None


def get_mongodb_vector_search(
    dimension: int = 512,
) -> Optional[MongoDBVectorSearch]:
    """Get or create global MongoDB vector search instance."""
    global _global_search
    
    if _global_search is None:
        _global_search = MongoDBVectorSearch(dimension=dimension)
    
    return _global_search if _global_search.is_ready() else None


def reset_mongodb_vector_search():
    """Reset global instance."""
    global _global_search
    _global_search = None
