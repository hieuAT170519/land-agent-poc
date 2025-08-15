"""Document classification using Vietnamese keyword heuristics."""
import re
from typing import Dict, List


def classify_document(text: str) -> str:
    """
    Classify Vietnamese land transfer document type based on keywords.
    
    Args:
        text: OCR extracted text
        
    Returns:
        Document type: 'contract', 'certificate', 'receipt', or 'unknown'
    """
    if not text:
        return "unknown"
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Contract keywords
    contract_keywords = [
        "hợp đồng chuyển nhượng",
        "hợp đồng mua bán",
        "hợp đồng",
        "chuyển nhượng",
        "mua bán"
    ]
    
    # Certificate keywords
    certificate_keywords = [
        "giấy chứng nhận",
        "sổ đỏ",
        "chứng nhận quyền",
        "quyền sử dụng đất",
        "quyền sở hữu"
    ]
    
    # Receipt keywords
    receipt_keywords = [
        "biên lai",
        "hóa đơn",
        "phiếu thu",
        "thu tiền",
        "thanh toán"
    ]
    
    # Score each category
    contract_score = sum(1 for keyword in contract_keywords if keyword in text_lower)
    certificate_score = sum(1 for keyword in certificate_keywords if keyword in text_lower)
    receipt_score = sum(1 for keyword in receipt_keywords if keyword in text_lower)
    
    # Determine document type based on highest score
    scores = {
        "contract": contract_score,
        "certificate": certificate_score,
        "receipt": receipt_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return "unknown"
    
    # Return the type with highest score
    for doc_type, score in scores.items():
        if score == max_score:
            return doc_type
    
    return "unknown"


def get_classification_confidence(text: str) -> Dict[str, float]:
    """
    Get confidence scores for each document type.
    
    Args:
        text: OCR extracted text
        
    Returns:
        Dictionary with confidence scores for each type
    """
    if not text:
        return {"contract": 0.0, "certificate": 0.0, "receipt": 0.0, "unknown": 1.0}
    
    text_lower = text.lower()
    
    # Count keyword matches
    contract_matches = len([kw for kw in ["hợp đồng", "chuyển nhượng", "mua bán"] if kw in text_lower])
    certificate_matches = len([kw for kw in ["giấy chứng nhận", "sổ đỏ", "quyền sử dụng"] if kw in text_lower])
    receipt_matches = len([kw for kw in ["biên lai", "hóa đơn", "phiếu thu"] if kw in text_lower])
    
    total_matches = contract_matches + certificate_matches + receipt_matches
    
    if total_matches == 0:
        return {"contract": 0.0, "certificate": 0.0, "receipt": 0.0, "unknown": 1.0}
    
    return {
        "contract": contract_matches / total_matches,
        "certificate": certificate_matches / total_matches,
        "receipt": receipt_matches / total_matches,
        "unknown": 0.0
    }