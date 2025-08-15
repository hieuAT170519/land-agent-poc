"""RAG retriever for querying the TF-IDF index"""

import os
import logging
from typing import List, Dict, Any
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = ".cache/rag"
VECTORIZER_FILE = "vectorizer.pkl"
MATRIX_FILE = "matrix.pkl"
DOCS_FILE = "docs.pkl"


def query_documents(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the document index and return top-K relevant passages.
    
    Args:
        question: Query string
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing passage text, source, and similarity score
    """
    logger.info(f"Querying documents with: '{question}' (top_k={top_k})")
    
    # Load index components
    vectorizer, tfidf_matrix, docs_data = _load_index()
    
    # Vectorize the query
    query_vector = vectorizer.transform([question])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top-K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Prepare results
    results = []
    for i, idx in enumerate(top_indices):
        similarity_score = similarities[idx]
        
        # Skip very low similarity scores
        if similarity_score < 0.01:
            continue
        
        document = docs_data["documents"][idx]
        metadata = docs_data["metadata"][idx]
        
        result = {
            "passage": document,
            "source_file": metadata["source_file"],
            "passage_index": metadata["passage_index"],
            "similarity_score": float(similarity_score),
            "rank": i + 1
        }
        
        results.append(result)
    
    logger.info(f"Retrieved {len(results)} relevant passages")
    
    return results


def _load_index():
    """Load the TF-IDF index components"""
    vectorizer_path = os.path.join(CACHE_DIR, VECTORIZER_FILE)
    matrix_path = os.path.join(CACHE_DIR, MATRIX_FILE)
    docs_path = os.path.join(CACHE_DIR, DOCS_FILE)
    
    # Check if all required files exist
    missing_files = []
    if not os.path.exists(vectorizer_path):
        missing_files.append(VECTORIZER_FILE)
    if not os.path.exists(matrix_path):
        missing_files.append(MATRIX_FILE)
    if not os.path.exists(docs_path):
        missing_files.append(DOCS_FILE)
    
    if missing_files:
        raise FileNotFoundError(f"Index files not found: {', '.join(missing_files)}. Please build the index first.")
    
    try:
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = joblib.load(matrix_path)
        docs_data = joblib.load(docs_path)
        
        logger.info(f"Loaded index: {len(docs_data['documents'])} documents, vocab size: {len(vectorizer.vocabulary_)}")
        
        return vectorizer, tfidf_matrix, docs_data
        
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        raise


def semantic_search(query: str, top_k: int = 5, min_score: float = 0.01) -> List[Dict[str, Any]]:
    """
    Perform semantic search with additional filtering and ranking.
    
    Args:
        query: Search query
        top_k: Maximum number of results to return
        min_score: Minimum similarity score threshold
        
    Returns:
        List of ranked search results
    """
    # Get basic results
    results = query_documents(query, top_k * 2)  # Get more results for filtering
    
    # Filter by minimum score
    filtered_results = [r for r in results if r["similarity_score"] >= min_score]
    
    # Additional ranking based on passage quality
    for result in filtered_results:
        passage = result["passage"]
        
        # Boost score for longer, more informative passages
        length_boost = min(len(passage) / 500, 1.0) * 0.1
        
        # Boost score for passages with numbers (likely to contain specific info)
        number_boost = 0.05 if any(c.isdigit() for c in passage) else 0
        
        # Apply boosts
        result["similarity_score"] += length_boost + number_boost
        
        # Add explanation
        result["relevance_factors"] = []
        if length_boost > 0:
            result["relevance_factors"].append("detailed_content")
        if number_boost > 0:
            result["relevance_factors"].append("contains_numbers")
    
    # Re-sort by updated scores
    filtered_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Return top-K results
    return filtered_results[:top_k]


def get_document_summary(source_file: str) -> Dict[str, Any]:
    """
    Get summary information about a specific source document.
    
    Args:
        source_file: Name of the source file
        
    Returns:
        Dictionary with document summary
    """
    try:
        _, _, docs_data = _load_index()
        
        # Find all passages from this source file
        file_passages = [
            (i, doc, meta) for i, (doc, meta) in enumerate(
                zip(docs_data["documents"], docs_data["metadata"])
            ) if meta["source_file"] == source_file
        ]
        
        if not file_passages:
            return {"error": f"No passages found for file: {source_file}"}
        
        # Calculate statistics
        total_passages = len(file_passages)
        total_chars = sum(len(doc) for _, doc, _ in file_passages)
        avg_passage_length = total_chars / total_passages
        
        # Get first few passages as preview
        preview_passages = [doc for _, doc, _ in file_passages[:3]]
        
        return {
            "source_file": source_file,
            "total_passages": total_passages,
            "total_characters": total_chars,
            "average_passage_length": avg_passage_length,
            "preview_passages": preview_passages
        }
        
    except FileNotFoundError:
        return {"error": "Index not found. Please build the index first."}
    except Exception as e:
        logger.error(f"Error getting document summary: {str(e)}")
        return {"error": str(e)}


def find_similar_passages(passage_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find passages similar to a given passage.
    
    Args:
        passage_text: Text to find similar passages for
        top_k: Number of similar passages to return
        
    Returns:
        List of similar passages with similarity scores
    """
    return query_documents(passage_text, top_k)


def get_vocabulary_stats() -> Dict[str, Any]:
    """
    Get statistics about the vocabulary in the index.
    
    Returns:
        Dictionary with vocabulary statistics
    """
    try:
        vectorizer, _, _ = _load_index()
        
        vocabulary = vectorizer.vocabulary_
        feature_names = vectorizer.get_feature_names_out()
        
        # Get most common terms (those with highest IDF scores tend to be less common)
        idf_scores = vectorizer.idf_
        term_scores = list(zip(feature_names, idf_scores))
        term_scores.sort(key=lambda x: x[1])  # Sort by IDF (lower = more common)
        
        return {
            "vocabulary_size": len(vocabulary),
            "most_common_terms": [term for term, _ in term_scores[:20]],
            "least_common_terms": [term for term, _ in term_scores[-20:]],
            "average_idf": float(np.mean(idf_scores)),
            "idf_range": {
                "min": float(np.min(idf_scores)),
                "max": float(np.max(idf_scores))
            }
        }
        
    except FileNotFoundError:
        return {"error": "Index not found. Please build the index first."}
    except Exception as e:
        logger.error(f"Error getting vocabulary stats: {str(e)}")
        return {"error": str(e)}