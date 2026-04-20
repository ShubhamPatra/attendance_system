"""
Migration script: Build and persist FAISS index for face embeddings.

This script:
1. Loads all student embeddings from MongoDB
2. Builds a FAISS index
3. Persists the index to disk
4. Updates student records with FAISS row IDs (optional)

Usage:
    python scripts/migrate_to_vector_search.py
    
    Or with custom index type:
    python scripts/migrate_to_vector_search.py --index-type IVFFlat --nlist 50
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import core.config as config
import core.database as database
from vision.embedding_search import FAISSIndex
from core.utils import setup_logging

logger = setup_logging()


def migrate_to_faiss_index(
    index_type: str = "IVFFlat",
    nlist: int = 50,
    force_rebuild: bool = False,
) -> bool:
    """
    Migrate embeddings to FAISS index.
    
    Args:
        index_type: "Flat", "IVFFlat", or "HNSW"
        nlist: Number of clusters for IVFFlat
        force_rebuild: Force rebuild even if index exists
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("FAISS MIGRATION: Building vector search index")
        logger.info("=" * 70)
        
        # Check if index already exists
        index_path = config.VECTOR_SEARCH_INDEX_PATH
        if index_path and Path(index_path).exists() and not force_rebuild:
            logger.info(f"Index already exists at {index_path}")
            response = input("Rebuild? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Migration skipped")
                return True
        
        # Get all student embeddings
        logger.info("\n1. Loading student embeddings from MongoDB...")
        students_with_encodings = database.get_student_encodings()
        
        if not students_with_encodings:
            logger.error("No student encodings found!")
            return False
        
        total_embeddings = 0
        embeddings_list = []
        student_ids_list = []
        
        for student_id, name, encodings in students_with_encodings:
            if not encodings:
                continue
            
            for encoding in encodings:
                if encoding is not None and len(encoding) > 0:
                    embeddings_list.append(encoding)
                    student_ids_list.append(str(student_id))
                    total_embeddings += 1
        
        logger.info(f"   ✓ Loaded {total_embeddings} embeddings from {len(students_with_encodings)} students")
        
        if total_embeddings == 0:
            logger.error("No valid embeddings found!")
            return False
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        logger.info(f"   ✓ Embeddings shape: {embeddings_array.shape}")
        
        # Build FAISS index
        # For small datasets, fall back to Flat index if using IVFFlat with too few embeddings
        effective_index_type = index_type
        effective_nlist = nlist
        
        if index_type == "IVFFlat" and total_embeddings < nlist:
            logger.info(f"   ⚠ Warning: {total_embeddings} embeddings < nlist={nlist} clusters")
            logger.info(f"   ⚠ Falling back to Flat index for small dataset")
            effective_index_type = "Flat"
        
        logger.info(f"\n2. Building FAISS index (type={effective_index_type}, nlist={effective_nlist})...")
        index = FAISSIndex(
            dimension=embeddings_array.shape[1],
            index_type=effective_index_type,
            nlist=effective_nlist,
        )
        
        if not index.is_available():
            logger.error("FAISS not available!")
            return False
        
        # Add embeddings to index
        success = index.add(embeddings_array, student_ids_list)
        if not success:
            logger.error("Failed to add embeddings to index")
            return False
        
        logger.info(f"   ✓ Index size: {index.get_size()} vectors")
        
        # Save index
        logger.info(f"\n3. Saving index to {index_path}...")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        success = index.save(index_path)
        if not success:
            logger.error("Failed to save index")
            return False
        
        logger.info(f"   ✓ Index saved successfully")
        
        # Verify index load
        logger.info("\n4. Verifying index...")
        test_index = FAISSIndex(
            dimension=embeddings_array.shape[1],
            index_type=index_type,
            nlist=nlist,
        )
        success = test_index.load(index_path)
        if not success:
            logger.error("Failed to load index for verification")
            return False
        
        logger.info(f"   ✓ Index verified (size: {test_index.get_size()})")
        
        # Test search
        logger.info("\n5. Testing search...")
        test_embedding = embeddings_array[0:1]  # First embedding
        results = test_index.search(test_embedding[0], k=5)
        
        if results:
            logger.info(f"   ✓ Search successful ({len(results)} results)")
            for i, (student_id, distance, confidence) in enumerate(results[:3], 1):
                logger.info(f"      {i}. Student {student_id}: distance={distance:.4f}, confidence={confidence:.4f}")
        else:
            logger.warning("   ⚠ Search returned no results")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✓ FAISS MIGRATION COMPLETE")
        logger.info(f"  Index: {index_path}")
        logger.info(f"  Size: {total_embeddings} vectors")
        logger.info(f"  Type: {index_type}")
        logger.info("=" * 70)
        
        return True
    
    except Exception as exc:
        logger.error(f"Migration failed: {exc}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate face embeddings to FAISS index"
    )
    parser.add_argument(
        '--index-type',
        choices=['Flat', 'IVFFlat', 'HNSW'],
        default='IVFFlat',
        help='FAISS index type (default: IVFFlat)'
    )
    parser.add_argument(
        '--nlist',
        type=int,
        default=50,
        help='Number of clusters for IVFFlat (default: 50)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    
    args = parser.parse_args()
    
    success = migrate_to_faiss_index(
        index_type=args.index_type,
        nlist=args.nlist,
        force_rebuild=args.force,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
