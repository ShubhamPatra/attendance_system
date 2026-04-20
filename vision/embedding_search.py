"""
FAISS-based vector search module for face embedding retrieval.

Implements efficient k-NN search using FAISS indices for scalable face matching.
Supports multiple index types (Flat, IVFFlat, HNSW) and graceful fallback to brute-force.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

import core.config as config
from core.utils import setup_logging

logger = setup_logging()

# Lazy import FAISS
_faiss = None
_FAISS_AVAILABLE = False


def _try_import_faiss():
    """Try to import FAISS, log warning if unavailable."""
    global _faiss, _FAISS_AVAILABLE
    
    if _FAISS_AVAILABLE or _faiss is not None:
        return _FAISS_AVAILABLE
    
    try:
        import faiss
        _faiss = faiss
        _FAISS_AVAILABLE = True
        logger.info("✓ FAISS library loaded successfully")
        return True
    except ImportError:
        logger.warning(
            "FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu). "
            "Falling back to brute-force face matching."
        )
        _FAISS_AVAILABLE = False
        return False


class FAISSIndex:
    """Manages FAISS vector search index for face embeddings."""
    
    def __init__(
        self,
        dimension: int = 512,
        index_type: str = "IVFFlat",
        nlist: int = 50,
        nprobe: int = 10,
    ):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding vector dimension (default 512 for ArcFace)
            index_type: "Flat", "IVFFlat", or "HNSW"
            nlist: Number of clusters for IVFFlat (ignored for other types)
            nprobe: Number of clusters to probe for IVFFlat
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.id_map = {}  # Maps index row -> student_id
        self.id_to_row = {}  # Maps student_id -> row(s)
        self.size = 0
        self.is_trained = False
        
        self._initialized = _try_import_faiss()
        
        if self._initialized:
            self._build_index()
    
    def _build_index(self):
        """Build FAISS index based on configured type."""
        try:
            if self.index_type == "Flat":
                # Brute-force index
                self.index = _faiss.IndexFlatL2(self.dimension)
                self.is_trained = True
                logger.debug("Built FAISS Flat index (L2 distance)")
            
            elif self.index_type == "IVFFlat":
                # Inverted file with flat quantizer
                quantizer = _faiss.IndexFlatL2(self.dimension)
                self.index = _faiss.IndexIVFFlat(
                    quantizer,
                    self.dimension,
                    self.nlist
                )
                self.index.nprobe = self.nprobe
                logger.debug(f"Built FAISS IVFFlat index (nlist={self.nlist}, nprobe={self.nprobe})")
            
            elif self.index_type == "HNSW":
                # Hierarchical navigable small world
                self.index = _faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 64
                self.is_trained = True
                logger.debug("Built FAISS HNSW index")
            
            else:
                logger.warning(f"Unknown index type: {self.index_type}, using Flat")
                self.index = _faiss.IndexFlatL2(self.dimension)
                self.is_trained = True
        
        except Exception as exc:
            logger.error(f"Failed to build FAISS index: {exc}")
            self._initialized = False
            self.index = None
    
    def add(
        self,
        embeddings: np.ndarray,
        student_ids: List[str]
    ) -> bool:
        """
        Add embeddings to index.
        
        Args:
            embeddings: (N, D) numpy array of L2-normalized embeddings
            student_ids: List of N student IDs (can have duplicates for multi-face students)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized or self.index is None:
            logger.warning("FAISS index not initialized")
            return False
        
        try:
            # Validate inputs
            if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
                logger.error(
                    f"Invalid embedding shape: {embeddings.shape}, expected (N, {self.dimension})"
                )
                return False
            
            if len(student_ids) != embeddings.shape[0]:
                logger.error(
                    f"Mismatch: {len(student_ids)} IDs for {embeddings.shape[0]} embeddings"
                )
                return False
            
            # Ensure embeddings are float32 and L2-normalized
            embeddings = np.asarray(embeddings, dtype=np.float32)
            
            # L2-normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
            
            # Train index if needed (for IVFFlat)
            if self.index_type == "IVFFlat" and not self.is_trained and embeddings.shape[0] >= self.nlist:
                self.index.train(embeddings)
                self.is_trained = True
                logger.debug(f"Trained FAISS IVFFlat index with {embeddings.shape[0]} samples")
            
            # Add embeddings
            start_row = self.size
            self.index.add(embeddings)
            
            # Update ID mappings
            for i, student_id in enumerate(student_ids):
                row = start_row + i
                self.id_map[row] = student_id
                
                if student_id not in self.id_to_row:
                    self.id_to_row[student_id] = []
                self.id_to_row[student_id].append(row)
            
            self.size += embeddings.shape[0]
            logger.debug(f"Added {embeddings.shape[0]} embeddings (total index size: {self.size})")
            return True
        
        except Exception as exc:
            logger.error(f"Failed to add embeddings to FAISS index: {exc}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: (D,) normalized embedding vector
            k: Number of neighbors to retrieve
            
        Returns:
            List of (student_id, distance, confidence) tuples
            where confidence = 1.0 - (distance / 2) [maps L2 distance to [0, 1]]
        """
        if not self._initialized or self.index is None or self.size == 0:
            logger.warning("FAISS index not available or empty")
            return []
        
        try:
            # Ensure query is float32 and L2-normalized
            query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # L2-normalize query
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            
            # Limit k to index size
            actual_k = min(k, self.size)
            
            # Search
            distances, indices = self.index.search(query, actual_k)
            
            # Convert to output format
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= self.size:
                    continue  # Invalid result
                
                if idx not in self.id_map:
                    logger.warning(f"Index {idx} not in ID map")
                    continue
                
                student_id = self.id_map[idx]
                # Convert L2 distance to confidence [0, 1]
                # L2 distance range for normalized vectors: [0, 2]
                confidence = 1.0 - (float(distance) / 2.0)
                confidence = max(0.0, min(1.0, confidence))
                
                results.append((student_id, float(distance), confidence))
            
            return results
        
        except Exception as exc:
            logger.error(f"FAISS search failed: {exc}")
            return []
    
    def clear(self) -> bool:
        """Clear index."""
        try:
            self._build_index()
            self.id_map.clear()
            self.id_to_row.clear()
            self.size = 0
            logger.debug("Cleared FAISS index")
            return True
        except Exception as exc:
            logger.error(f"Failed to clear index: {exc}")
            return False
    
    def save(self, filepath: str) -> bool:
        """
        Save index to disk.
        
        Args:
            filepath: Path to save index
            
        Returns:
            True if successful
        """
        if not self._initialized or self.index is None:
            logger.warning("FAISS index not initialized")
            return False
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save index
            _faiss.write_index(self.index, filepath)
            
            # Save metadata
            metadata = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "id_map": self.id_map,
                "id_to_row": self.id_to_row,
                "size": self.size,
                "is_trained": self.is_trained,
            }
            
            metadata_path = filepath + ".meta"
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved FAISS index to {filepath} ({self.size} vectors)")
            return True
        
        except Exception as exc:
            logger.error(f"Failed to save FAISS index: {exc}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load index from disk.
        
        Args:
            filepath: Path to load index from
            
        Returns:
            True if successful
        """
        if not self._initialized:
            logger.warning("FAISS not available")
            return False
        
        try:
            # Load index
            self.index = _faiss.read_index(filepath)
            
            # Load metadata
            metadata_path = filepath + ".meta"
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                
                self.dimension = metadata.get("dimension", 512)
                self.index_type = metadata.get("index_type", "IVFFlat")
                self.nlist = metadata.get("nlist", 50)
                self.nprobe = metadata.get("nprobe", 10)
                self.id_map = metadata.get("id_map", {})
                self.id_to_row = metadata.get("id_to_row", {})
                self.size = metadata.get("size", 0)
                self.is_trained = metadata.get("is_trained", False)
            
            logger.info(f"Loaded FAISS index from {filepath} ({self.size} vectors)")
            return True
        
        except Exception as exc:
            logger.error(f"Failed to load FAISS index: {exc}")
            return False
    
    def get_size(self) -> int:
        """Get current index size."""
        return self.size if self._initialized else 0
    
    def is_available(self) -> bool:
        """Check if FAISS is available and index is ready."""
        return self._initialized and self.index is not None


# Global singleton instance
_global_index: Optional[FAISSIndex] = None


def get_global_index(
    dimension: int = 512,
    index_type: Optional[str] = None,
) -> Optional[FAISSIndex]:
    """
    Get or create global FAISS index instance.
    Loads persisted index from disk if available.
    
    Args:
        dimension: Embedding dimension
        index_type: Override configured index type
        
    Returns:
        FAISSIndex instance or None if FAISS unavailable
    """
    global _global_index
    
    if _global_index is None:
        idx_type = index_type or config.FAISS_INDEX_TYPE
        _global_index = FAISSIndex(
            dimension=dimension,
            index_type=idx_type,
            nlist=config.FAISS_INDEX_NLIST,
            nprobe=config.FAISS_INDEX_NPROBE,
        )
        
        # Try to load persisted index from disk
        if _global_index.is_available() and config.VECTOR_SEARCH_INDEX_PATH:
            from pathlib import Path
            if Path(config.VECTOR_SEARCH_INDEX_PATH).exists():
                try:
                    _global_index.load(config.VECTOR_SEARCH_INDEX_PATH)
                    logger.debug(f"Loaded FAISS index from {config.VECTOR_SEARCH_INDEX_PATH}")
                except Exception as exc:
                    logger.warning(f"Failed to load persisted FAISS index: {exc}")
    
    return _global_index if _global_index.is_available() else None


def reset_global_index():
    """Reset global index instance."""
    global _global_index
    _global_index = None
