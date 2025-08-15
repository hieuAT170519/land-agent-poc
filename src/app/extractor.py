"""Field extraction using regex and heuristics for Vietnamese documents."""
import re
from typing import Dict, List, Optional


def extract_fields(text: str, doc_type: str) -> Dict[str, Optional[str]]:
    """
    Extract fields from document text based on document type.
    
    Args:
        text: OCR extracted text
        doc_type: Document type from classifier
        
    Returns:
        Dictionary with extracted fields, omitting keys when not found
    """
    if not text:
        return {}
    
    extracted = {}
    
    # Common field extractors
    parties = extract_parties(text)
    if parties:
        extracted["parties"] = parties
    
    id_numbers = extract_id_numbers(text)
    if id_numbers:
        extracted["id_numbers"] = id_numbers
        
    addresses = extract_addresses(text)
    if addresses:
        extracted["addresses"] = addresses
        
    parcel_number = extract_parcel_number(text)
    if parcel_number:
        extracted["parcel_number"] = parcel_number
        
    area_m2 = extract_area(text)
    if area_m2:
        extracted["area_m2"] = area_m2
        
    date = extract_date(text)
    if date:
        extracted["date"] = date
        
    notary_office = extract_notary_office(text)
    if notary_office:
        extracted["notary_office"] = notary_office
    
    # Document type specific fields
    if doc_type == "contract":
        contract_value = extract_contract_value(text)
        if contract_value:
            extracted["contract_value"] = contract_value
            
    return extracted


def extract_parties(text: str) -> Optional[List[str]]:
    """Extract party names from document."""
    parties = []
    
    # Common patterns for parties in Vietnamese documents
    patterns = [
        r"(?:bên bán|người bán|bên a):\s*([^\n\r]+)",
        r"(?:bên mua|người mua|bên b):\s*([^\n\r]+)",
        r"(?:họ tên|tên):\s*([a-zA-ZÀ-ỹ\s]+)",
        r"(?:ông|bà|anh|chị)\s+([a-zA-ZÀ-ỹ\s]+)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            if len(name) > 3 and name not in parties:  # Basic validation
                parties.append(name)
    
    return parties if parties else None


def extract_id_numbers(text: str) -> Optional[List[str]]:
    """Extract ID/passport numbers from document."""
    id_numbers = []
    
    # Vietnamese ID card patterns (CMND: 9-12 digits, CCCD: 12 digits)
    patterns = [
        r"(?:cmnd|cccd|số|id):\s*(\d{9,12})",
        r"(?:chứng minh|căn cước).*?(\d{9,12})",
        r"\b(\d{9})\b",  # 9-digit CMND
        r"\b(\d{12})\b"  # 12-digit CCCD
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match not in id_numbers:
                id_numbers.append(match)
    
    return id_numbers if id_numbers else None


def extract_addresses(text: str) -> Optional[List[str]]:
    """Extract addresses from document."""
    addresses = []
    
    # Address patterns
    patterns = [
        r"(?:địa chỉ|đc):\s*([^\n\r]+)",
        r"(?:thường trú|tại):\s*([^\n\r]+)",
        r"(?:phường|xã|quận|huyện|tỉnh|tp)[\s\.]([^\n\r,]+)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            addr = match.strip()
            if len(addr) > 5 and addr not in addresses:
                addresses.append(addr)
    
    return addresses if addresses else None


def extract_parcel_number(text: str) -> Optional[str]:
    """Extract land parcel number."""
    patterns = [
        r"(?:thửa|thửa đất|số thửa):\s*(\d+[a-zA-Z]*)",
        r"(?:tờ bản đồ|bản đồ):\s*(\d+)",
        r"thửa\s+(\d+[a-zA-Z]*)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_area(text: str) -> Optional[str]:
    """Extract land area in square meters."""
    patterns = [
        r"(?:diện tích|dt):\s*([\d,\.]+)\s*(?:m2|m²|mét vuông)",
        r"([\d,\.]+)\s*(?:m2|m²|mét vuông)",
        r"(?:diện tích).*?([\d,\.]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            area = match.group(1).replace(",", ".")
            try:
                float(area)  # Validate it's a number
                return area
            except ValueError:
                continue
    
    return None


def extract_date(text: str) -> Optional[str]:
    """Extract date from document."""
    patterns = [
        r"(?:ngày|date):\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})",
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})",
        r"(?:ngày)\s+(\d{1,2})\s+(?:tháng)\s+(\d{1,2})\s+(?:năm)\s+(\d{4})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 3:  # Vietnamese format
                day, month, year = match.groups()
                return f"{day}/{month}/{year}"
            else:
                return match.group(1)
    
    return None


def extract_notary_office(text: str) -> Optional[str]:
    """Extract notary office name."""
    patterns = [
        r"(?:phòng công chứng|văn phòng công chứng)\s*([^\n\r]+)",
        r"(?:công chứng viên):\s*([a-zA-ZÀ-ỹ\s]+)",
        r"(?:tại|ở)\s+(?:phòng|văn phòng)\s*([^\n\r]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            office = match.group(1).strip()
            if len(office) > 3:
                return office
    
    return None


def extract_contract_value(text: str) -> Optional[str]:
    """Extract contract value in VND."""
    patterns = [
        r"(?:giá|giá trị|tổng giá|thành tiền):\s*([\d,\.]+)(?:\s*(?:vnđ|vnd|đồng))?",
        r"([\d,\.]+)\s*(?:vnđ|vnd|đồng)",
        r"(?:tổng cộng|tổng).*?([\d,\.]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", "").replace(".", "")
            try:
                int(value)  # Validate it's a number
                return value
            except ValueError:
                continue
    
    return None