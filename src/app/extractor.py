"""Field extraction using regex and heuristic patterns"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Regex patterns for common fields
PATTERNS = {
    "id_numbers": [
        r'\b\d{9}\b',  # 9-digit ID (old format)
        r'\b\d{12}\b',  # 12-digit ID (new format)
        r'(?:cccd|cmnd|căn cước|chứng minh).*?(\d{9,12})',
    ],
    "phone_numbers": [
        r'(?:\+84|84|0)(?:3|5|7|8|9)\d{8}',
        r'\b0\d{9,10}\b'
    ],
    "dates": [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        r'ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}',
    ],
    "areas": [
        r'(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét vuông|diện tích)',
        r'diện tích.*?(\d+(?:[.,]\d+)?)',
    ],
    "addresses": [
        r'(?:địa chỉ|address).*?([A-Za-zÀ-ỹ\s\d,.-]+?)(?:\n|$|;)',
        r'(?:số|phố|đường|quận|huyện|tỉnh|thành phố).*?([A-Za-zÀ-ỹ\s\d,.-]+)',
    ],
    "money_amounts": [
        r'(\d+(?:[.,]\d+)?)\s*(?:đồng|vnd|vnđ|triệu|tỷ)',
        r'(?:giá|tiền|số tiền|thành tiền).*?(\d+(?:[.,]\d+)?)',
    ]
}


def extract_fields(text: str, doc_type: str) -> Dict[str, Any]:
    """
    Extract key fields from document text based on document type.
    
    Args:
        text: OCR text from the document
        doc_type: Document type (contract, certificate, etc.)
        
    Returns:
        Dictionary of extracted fields
    """
    if not text:
        return {}
    
    fields = {}
    
    # Extract common fields for all document types
    fields.update(_extract_common_fields(text))
    
    # Extract fields specific to document type
    if doc_type == "contract":
        fields.update(_extract_contract_fields(text))
    elif doc_type == "certificate":
        fields.update(_extract_certificate_fields(text))
    elif doc_type == "receipt":
        fields.update(_extract_receipt_fields(text))
    elif doc_type == "application":
        fields.update(_extract_application_fields(text))
    elif doc_type == "id_document":
        fields.update(_extract_id_document_fields(text))
    
    # Clean up empty fields
    fields = {k: v for k, v in fields.items() if v}
    
    logger.info(f"Extracted {len(fields)} fields from {doc_type} document")
    
    return fields


def _extract_common_fields(text: str) -> Dict[str, Any]:
    """Extract fields common to all document types"""
    fields = {}
    
    # Extract ID numbers
    id_numbers = _extract_by_patterns(text, PATTERNS["id_numbers"])
    if id_numbers:
        fields["id_numbers"] = id_numbers
    
    # Extract phone numbers
    phone_numbers = _extract_by_patterns(text, PATTERNS["phone_numbers"])
    if phone_numbers:
        fields["phone_numbers"] = phone_numbers
    
    # Extract dates
    dates = _extract_by_patterns(text, PATTERNS["dates"])
    if dates:
        fields["dates"] = dates
    
    # Extract addresses
    addresses = _extract_by_patterns(text, PATTERNS["addresses"])
    if addresses:
        fields["addresses"] = [addr.strip() for addr in addresses if len(addr.strip()) > 10]
    
    return fields


def _extract_contract_fields(text: str) -> Dict[str, Any]:
    """Extract fields specific to contracts"""
    fields = {}
    
    # Extract parties (simple heuristic)
    parties = _extract_parties(text)
    if parties:
        fields["parties"] = parties
    
    # Extract money amounts
    amounts = _extract_by_patterns(text, PATTERNS["money_amounts"])
    if amounts:
        fields["contract_value"] = amounts
    
    # Extract areas
    areas = _extract_by_patterns(text, PATTERNS["areas"])
    if areas:
        fields["area_m2"] = areas
    
    # Extract notary office
    notary = _extract_notary_office(text)
    if notary:
        fields["notary_office"] = notary
    
    return fields


def _extract_certificate_fields(text: str) -> Dict[str, Any]:
    """Extract fields specific to certificates"""
    fields = {}
    
    # Extract parcel number
    parcel_patterns = [
        r'(?:thửa|parcel).*?(\d+)',
        r'số thửa.*?(\d+)',
        r'bản đồ số.*?(\d+)',
    ]
    parcel_numbers = _extract_by_patterns(text, parcel_patterns)
    if parcel_numbers:
        fields["parcel_number"] = parcel_numbers
    
    # Extract areas
    areas = _extract_by_patterns(text, PATTERNS["areas"])
    if areas:
        fields["area_m2"] = areas
    
    # Extract land use purpose
    purpose_patterns = [
        r'(?:mục đích|purpose).*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$|;)',
    ]
    purposes = _extract_by_patterns(text, purpose_patterns)
    if purposes:
        fields["land_use_purpose"] = purposes
    
    return fields


def _extract_receipt_fields(text: str) -> Dict[str, Any]:
    """Extract fields specific to receipts"""
    fields = {}
    
    # Extract money amounts
    amounts = _extract_by_patterns(text, PATTERNS["money_amounts"])
    if amounts:
        fields["amount"] = amounts
    
    # Extract fee types
    fee_patterns = [
        r'(?:lệ phí|phí|fee).*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$|;)',
    ]
    fees = _extract_by_patterns(text, fee_patterns)
    if fees:
        fields["fee_types"] = fees
    
    return fields


def _extract_application_fields(text: str) -> Dict[str, Any]:
    """Extract fields specific to applications"""
    fields = {}
    
    # Extract application type
    app_patterns = [
        r'(?:đơn xin|application for).*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$|;)',
    ]
    app_types = _extract_by_patterns(text, app_patterns)
    if app_types:
        fields["application_type"] = app_types
    
    return fields


def _extract_id_document_fields(text: str) -> Dict[str, Any]:
    """Extract fields specific to ID documents"""
    fields = {}
    
    # Extract names
    name_patterns = [
        r'(?:họ tên|name|full name).*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$|;)',
    ]
    names = _extract_by_patterns(text, name_patterns)
    if names:
        fields["full_name"] = names
    
    # Extract birth dates
    birth_patterns = [
        r'(?:sinh|born|date of birth).*?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
    ]
    birth_dates = _extract_by_patterns(text, birth_patterns)
    if birth_dates:
        fields["birth_date"] = birth_dates
    
    return fields


def _extract_by_patterns(text: str, patterns: List[str]) -> List[str]:
    """Extract values using multiple regex patterns"""
    results = []
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        if matches:
            results.extend([match.strip() for match in matches if match.strip()])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique_results.append(item)
    
    return unique_results


def _extract_parties(text: str) -> List[str]:
    """Extract party names from contract text"""
    party_patterns = [
        r'bên a.*?([A-Za-zÀ-ỹ\s]+?)(?:\n|bên b)',
        r'bên b.*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$)',
        r'(?:party|bên).*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$)',
    ]
    
    parties = _extract_by_patterns(text, party_patterns)
    return [party for party in parties if len(party) > 5]  # Filter short matches


def _extract_notary_office(text: str) -> Optional[str]:
    """Extract notary office information"""
    notary_patterns = [
        r'(?:văn phòng công chứng|notary office).*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$)',
        r'công chứng viên.*?([A-Za-zÀ-ỹ\s]+?)(?:\n|$)',
    ]
    
    notaries = _extract_by_patterns(text, notary_patterns)
    return notaries[0] if notaries else None