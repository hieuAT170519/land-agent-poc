"""Document classification using heuristic keyword matching"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Define document types and their keywords
DOCUMENT_PATTERNS = {
    "contract": [
        "hợp đồng chuyển nhượng",
        "hợp đồng mua bán",
        "contract",
        "agreement",
        "chuyển nhượng quyền sử dụng đất",
        "mua bán nhà đất",
        "thỏa thuận"
    ],
    "certificate": [
        "giấy chứng nhận",
        "certificate",
        "sổ đỏ",
        "quyền sử dụng đất",
        "ownership certificate",
        "land use rights certificate",
        "gcn",
        "chứng nhận quyền sử dụng đất"
    ],
    "receipt": [
        "biên lai",
        "receipt",
        "hóa đơn",
        "invoice",
        "phiếu thu",
        "thanh toán",
        "payment",
        "thu phí",
        "lệ phí"
    ],
    "application": [
        "đơn xin",
        "application",
        "đăng ký",
        "registration",
        "khai báo",
        "declaration",
        "thủ tục",
        "procedure"
    ],
    "id_document": [
        "chứng minh thư",
        "căn cước công dân",
        "identity card",
        "passport",
        "hộ chiếu",
        "cccd",
        "cmnd"
    ]
}


def classify_document(text: str) -> str:
    """
    Classify document type based on text content using heuristic keyword matching.
    
    Args:
        text: OCR text from the document
        
    Returns:
        Document type string (contract, certificate, receipt, application, id_document, unknown)
    """
    if not text:
        return "unknown"
    
    # Normalize text for matching
    text_lower = text.lower()
    text_normalized = re.sub(r'\s+', ' ', text_lower.strip())
    
    # Score each document type
    scores = {}
    
    for doc_type, keywords in DOCUMENT_PATTERNS.items():
        score = 0
        matches = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Count occurrences of this keyword
            count = text_normalized.count(keyword_lower)
            if count > 0:
                # Weight longer keywords more heavily
                weight = len(keyword.split()) * 2
                score += count * weight
                matches.append(f"{keyword}({count})")
        
        scores[doc_type] = score
        
        if matches:
            logger.info(f"Document type '{doc_type}' matches: {', '.join(matches)} (score: {score})")
    
    # Find the best match
    if not any(scores.values()):
        logger.info("No keyword matches found, classifying as 'unknown'")
        return "unknown"
    
    best_type = max(scores.keys(), key=lambda k: scores[k])
    best_score = scores[best_type]
    
    logger.info(f"Classified document as '{best_type}' with score {best_score}")
    
    return best_type


def get_confidence_score(text: str, doc_type: str) -> float:
    """
    Get confidence score for a document classification.
    
    Args:
        text: OCR text from the document
        doc_type: Classified document type
        
    Returns:
        Confidence score between 0 and 1
    """
    if doc_type == "unknown":
        return 0.0
    
    if doc_type not in DOCUMENT_PATTERNS:
        return 0.0
    
    text_lower = text.lower()
    text_normalized = re.sub(r'\s+', ' ', text_lower.strip())
    
    total_keywords = len(DOCUMENT_PATTERNS[doc_type])
    matched_keywords = 0
    
    for keyword in DOCUMENT_PATTERNS[doc_type]:
        if keyword.lower() in text_normalized:
            matched_keywords += 1
    
    # Basic confidence based on keyword coverage
    confidence = matched_keywords / total_keywords
    
    # Boost confidence for longer documents (more context)
    text_length_factor = min(len(text) / 1000, 1.0)  # Cap at 1000 chars
    confidence = confidence * (0.7 + 0.3 * text_length_factor)
    
    return min(confidence, 1.0)