"""TF-IDF indexing for legal documents."""
import os
import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Default paths
LEGAL_TEXTS_DIR = "data/legal_texts"
CACHE_DIR = ".cache/rag"
VECTORIZER_PATH = f"{CACHE_DIR}/vectorizer.pkl"
MATRIX_PATH = f"{CACHE_DIR}/matrix.pkl"
DOCS_PATH = f"{CACHE_DIR}/docs.pkl"


def split_text_into_passages(text: str, max_length: int = 350, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping passages.
    
    Args:
        text: Input text to split
        max_length: Maximum length of each passage
        overlap: Number of characters to overlap between passages
        
    Returns:
        List of text passages
    """
    if len(text) <= max_length:
        return [text]
    
    passages = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at word boundary if possible
        if end < len(text):
            # Look for space within last 50 characters
            space_pos = text.rfind(' ', end - 50, end)
            if space_pos > start:
                end = space_pos
        
        passage = text[start:end].strip()
        if passage:
            passages.append(passage)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return passages


def load_legal_texts(texts_dir: str = LEGAL_TEXTS_DIR) -> List[Tuple[str, str]]:
    """
    Load all legal text files from directory.
    
    Args:
        texts_dir: Directory containing .txt files
        
    Returns:
        List of (filename, content) tuples
    """
    texts = []
    texts_path = Path(texts_dir)
    
    if not texts_path.exists():
        logger.warning(f"Legal texts directory does not exist: {texts_dir}")
        return texts
    
    for txt_file in texts_path.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    texts.append((txt_file.name, content))
                    logger.info(f"Loaded legal text: {txt_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {txt_file}: {e}")
    
    return texts


def build_index(texts_dir: str = LEGAL_TEXTS_DIR, cache_dir: str = CACHE_DIR) -> bool:
    """
    Build TF-IDF index from legal texts and persist to disk.
    
    Args:
        texts_dir: Directory containing legal text files
        cache_dir: Directory to save index artifacts
        
    Returns:
        True if indexing successful, False otherwise
    """
    try:
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load legal texts
        legal_texts = load_legal_texts(texts_dir)
        
        if not legal_texts:
            logger.warning("No legal texts found to index")
            return False
        
        # Split texts into passages and prepare documents
        all_passages = []
        doc_metadata = []  # (filename, passage_index, original_text)
        
        for filename, content in legal_texts:
            passages = split_text_into_passages(content)
            for i, passage in enumerate(passages):
                all_passages.append(passage)
                doc_metadata.append({
                    "filename": filename,
                    "passage_index": i,
                    "passage_text": passage,
                    "source": f"{filename}#{i}"
                })
        
        if not all_passages:
            logger.warning("No passages extracted from legal texts")
            return False
        
        # Build TF-IDF index
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # Keep Vietnamese stop words for now
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_passages)
        
        # Save artifacts
        vectorizer_path = os.path.join(cache_dir, "vectorizer.pkl")
        matrix_path = os.path.join(cache_dir, "matrix.pkl")
        docs_path = os.path.join(cache_dir, "docs.pkl")
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(tfidf_matrix, matrix_path)
        joblib.dump(doc_metadata, docs_path)
        
        logger.info(f"Successfully built index with {len(all_passages)} passages from {len(legal_texts)} documents")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return False


def is_index_available(cache_dir: str = CACHE_DIR) -> bool:
    """
    Check if TF-IDF index artifacts are available.
    
    Args:
        cache_dir: Directory containing index artifacts
        
    Returns:
        True if all required artifacts exist
    """
    required_files = [
        os.path.join(cache_dir, "vectorizer.pkl"),
        os.path.join(cache_dir, "matrix.pkl"),
        os.path.join(cache_dir, "docs.pkl")
    ]
    
    return all(os.path.exists(f) for f in required_files)


def get_index_stats(cache_dir: str = CACHE_DIR) -> dict:
    """
    Get statistics about the current index.
    
    Args:
        cache_dir: Directory containing index artifacts
        
    Returns:
        Dictionary with index statistics
    """
    if not is_index_available(cache_dir):
        return {"available": False}
    
    try:
        docs_path = os.path.join(cache_dir, "docs.pkl")
        matrix_path = os.path.join(cache_dir, "matrix.pkl")
        
        doc_metadata = joblib.load(docs_path)
        tfidf_matrix = joblib.load(matrix_path)
        
        # Count unique source files
        unique_files = set(doc["filename"] for doc in doc_metadata)
        
        return {
            "available": True,
            "num_passages": len(doc_metadata),
            "num_source_files": len(unique_files),
            "vocabulary_size": tfidf_matrix.shape[1],
            "source_files": list(unique_files)
        }
        
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        return {"available": False, "error": str(e)}