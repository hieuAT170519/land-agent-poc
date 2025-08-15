"""Document retrieval using TF-IDF similarity."""
import os
import logging
from typing import List, Dict, Optional

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Default paths
CACHE_DIR = ".cache/rag"
VECTORIZER_PATH = f"{CACHE_DIR}/vectorizer.pkl"
MATRIX_PATH = f"{CACHE_DIR}/matrix.pkl"
DOCS_PATH = f"{CACHE_DIR}/docs.pkl"


class DocumentRetriever:
    """TF-IDF based document retriever."""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        """
        Initialize retriever with cached artifacts.
        
        Args:
            cache_dir: Directory containing index artifacts
        """
        self.cache_dir = cache_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_metadata = None
        self.loaded = False
        
    def load_index(self) -> bool:
        """
        Load TF-IDF index artifacts from disk.
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            vectorizer_path = os.path.join(self.cache_dir, "vectorizer.pkl")
            matrix_path = os.path.join(self.cache_dir, "matrix.pkl")
            docs_path = os.path.join(self.cache_dir, "docs.pkl")
            
            # Check if all files exist
            if not all(os.path.exists(f) for f in [vectorizer_path, matrix_path, docs_path]):
                logger.error("Index artifacts not found")
                return False
            
            # Load artifacts
            self.vectorizer = joblib.load(vectorizer_path)
            self.tfidf_matrix = joblib.load(matrix_path)
            self.doc_metadata = joblib.load(docs_path)
            
            self.loaded = True
            logger.info(f"Loaded index with {len(self.doc_metadata)} passages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def query(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Query the index for similar passages.
        
        Args:
            question: Query text
            top_k: Number of top results to return
            
        Returns:
            List of result dictionaries with passage, source, and score
        """
        if not self.loaded:
            if not self.load_index():
                raise RuntimeError("Index not available. Please build index first using POST /rag/index")
        
        if not question.strip():
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([question])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Format results
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > 0:  # Only include non-zero similarity scores
                    metadata = self.doc_metadata[idx]
                    results.append({
                        "passage": metadata["passage_text"],
                        "source": metadata["source"],
                        "filename": metadata["filename"],
                        "passage_index": metadata["passage_index"],
                        "score": float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise RuntimeError(f"Query processing failed: {e}")
    
    def is_available(self) -> bool:
        """
        Check if retriever is ready to process queries.
        
        Returns:
            True if index is loaded and ready
        """
        return self.loaded and all([
            self.vectorizer is not None,
            self.tfidf_matrix is not None,
            self.doc_metadata is not None
        ])


# Global retriever instance
_retriever = None


def get_retriever() -> DocumentRetriever:
    """
    Get global retriever instance (singleton pattern).
    
    Returns:
        DocumentRetriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever


def query_documents(question: str, top_k: int = 5) -> List[Dict]:
    """
    Convenience function to query documents.
    
    Args:
        question: Query text
        top_k: Number of top results to return
        
    Returns:
        List of result dictionaries
    """
    retriever = get_retriever()
    return retriever.query(question, top_k)


def is_retriever_ready() -> bool:
    """
    Check if retriever is ready to process queries.
    
    Returns:
        True if retriever is ready
    """
    retriever = get_retriever()
    return retriever.is_available() or retriever.load_index()


def get_retriever_stats() -> Dict:
    """
    Get statistics about the loaded retriever.
    
    Returns:
        Dictionary with retriever statistics
    """
    retriever = get_retriever()
    
    if not retriever.loaded:
        return {"loaded": False, "available": False}
    
    return {
        "loaded": True,
        "available": retriever.is_available(),
        "num_passages": len(retriever.doc_metadata) if retriever.doc_metadata else 0,
        "vocabulary_size": retriever.tfidf_matrix.shape[1] if retriever.tfidf_matrix is not None else 0
    }