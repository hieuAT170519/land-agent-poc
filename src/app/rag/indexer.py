"""RAG indexer for building TF-IDF index from legal texts"""

import os
import glob
import logging
from typing import List, Tuple, Dict, Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logger = logging.getLogger(__name__)

# Configuration
LEGAL_TEXTS_DIR = "data/legal_texts"
CACHE_DIR = ".cache/rag"
VECTORIZER_FILE = "vectorizer.pkl"
MATRIX_FILE = "matrix.pkl"
DOCS_FILE = "docs.pkl"

# Text processing parameters
CHUNK_SIZE = 400  # Characters per passage
OVERLAP_SIZE = 50  # Character overlap between passages


def build_index() -> Tuple[int, int]:
    """
    Build TF-IDF index from legal text files.
    
    Returns:
        Tuple of (files_processed, total_tokens)
    """
    logger.info("Starting to build RAG index...")
    
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Find all text files
    text_files = glob.glob(os.path.join(LEGAL_TEXTS_DIR, "*.txt"))
    
    if not text_files:
        logger.warning(f"No .txt files found in {LEGAL_TEXTS_DIR}")
        return 0, 0
    
    logger.info(f"Found {len(text_files)} text files to process")
    
    # Read and process all documents
    documents = []
    document_metadata = []
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into passages
            passages = _split_into_passages(content)
            
            for i, passage in enumerate(passages):
                documents.append(passage)
                document_metadata.append({
                    "source_file": os.path.basename(file_path),
                    "passage_index": i,
                    "file_path": file_path
                })
            
            logger.info(f"Processed {file_path}: {len(passages)} passages")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not documents:
        logger.warning("No valid documents found to index")
        return 0, 0
    
    logger.info(f"Total passages to index: {len(documents)}")
    
    # Build TF-IDF vectorizer and matrix
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit vocabulary size
        stop_words=None,  # Keep Vietnamese stop words for now
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=1,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency
        lowercase=True,
        token_pattern=r'[a-zA-ZÀ-ỹ]+'  # Vietnamese character support
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Save artifacts
        vectorizer_path = os.path.join(CACHE_DIR, VECTORIZER_FILE)
        matrix_path = os.path.join(CACHE_DIR, MATRIX_FILE)
        docs_path = os.path.join(CACHE_DIR, DOCS_FILE)
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(tfidf_matrix, matrix_path)
        joblib.dump({
            "documents": documents,
            "metadata": document_metadata
        }, docs_path)
        
        # Get vocabulary size
        vocab_size = len(vectorizer.vocabulary_)
        
        logger.info(f"Index built successfully:")
        logger.info(f"  - Files processed: {len(text_files)}")
        logger.info(f"  - Total passages: {len(documents)}")
        logger.info(f"  - Vocabulary size: {vocab_size}")
        logger.info(f"  - Matrix shape: {tfidf_matrix.shape}")
        
        return len(text_files), vocab_size
        
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        raise


def _split_into_passages(text: str) -> List[str]:
    """
    Split text into overlapping passages.
    
    Args:
        text: Input text to split
        
    Returns:
        List of text passages
    """
    # Clean up text
    text = _clean_text(text)
    
    if len(text) <= CHUNK_SIZE:
        return [text] if text.strip() else []
    
    passages = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        
        if end >= len(text):
            # Last passage
            passage = text[start:].strip()
            if passage:
                passages.append(passage)
            break
        
        # Try to break at sentence boundary
        passage_text = text[start:end]
        
        # Look for sentence endings near the end
        sentence_endings = ['.', '!', '?', '\n']
        best_break = -1
        
        for i in range(len(passage_text) - 1, max(len(passage_text) - 100, 0), -1):
            if passage_text[i] in sentence_endings:
                best_break = i + 1
                break
        
        if best_break > 0:
            passage = passage_text[:best_break].strip()
            next_start = start + best_break
        else:
            # No good break point, use word boundary
            words = passage_text.split()
            if len(words) > 1:
                passage = ' '.join(words[:-1])
                next_start = start + len(passage)
            else:
                passage = passage_text.strip()
                next_start = end
        
        if passage:
            passages.append(passage)
        
        # Move start position with overlap
        start = max(next_start - OVERLAP_SIZE, start + 1)
    
    return passages


def _clean_text(text: str) -> str:
    """Clean and normalize text for indexing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Vietnamese diacritics
    text = re.sub(r'[^\w\s\.,!?\-À-ỹ]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clear_index() -> bool:
    """
    Clear the existing index files.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        files_to_remove = [
            os.path.join(CACHE_DIR, VECTORIZER_FILE),
            os.path.join(CACHE_DIR, MATRIX_FILE),
            os.path.join(CACHE_DIR, DOCS_FILE)
        ]
        
        removed_count = 0
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        logger.info(f"Cleared {removed_count} index files")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        return False


def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the current index.
    
    Returns:
        Dictionary with index statistics
    """
    try:
        vectorizer_path = os.path.join(CACHE_DIR, VECTORIZER_FILE)
        matrix_path = os.path.join(CACHE_DIR, MATRIX_FILE)
        docs_path = os.path.join(CACHE_DIR, DOCS_FILE)
        
        if not all(os.path.exists(path) for path in [vectorizer_path, matrix_path, docs_path]):
            return {"status": "not_indexed", "error": "Index files not found"}
        
        # Load index components
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = joblib.load(matrix_path)
        docs_data = joblib.load(docs_path)
        
        # Get unique source files
        source_files = set(meta["source_file"] for meta in docs_data["metadata"])
        
        return {
            "status": "indexed",
            "vocabulary_size": len(vectorizer.vocabulary_),
            "total_passages": len(docs_data["documents"]),
            "source_files": list(source_files),
            "source_file_count": len(source_files),
            "matrix_shape": tfidf_matrix.shape,
            "index_files": {
                "vectorizer": os.path.exists(vectorizer_path),
                "matrix": os.path.exists(matrix_path),
                "documents": os.path.exists(docs_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting index stats: {str(e)}")
        return {"status": "error", "error": str(e)}