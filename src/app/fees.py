"""Fee calculation for land transfer transactions"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default nationwide fee rates (in VND)
DEFAULT_RATES = {
    "registration_fee_rate": 0.005,  # 0.5% of contract value
    "registration_fee_min": 100000,  # Minimum 100,000 VND
    "registration_fee_max": 50000000,  # Maximum 50,000,000 VND
    "personal_income_tax_rate": 0.02,  # 2% of contract value for individuals
    "notary_fee_base": 500000,  # Base notary fee
    "notary_fee_rate": 0.003,  # 0.3% of contract value
    "admin_fee": 50000,  # Flat administrative fee
}

# Province-specific overrides
PROVINCE_OVERRIDES = {
    "ho_chi_minh": {
        "registration_fee_rate": 0.006,  # Higher rate for HCMC
        "admin_fee": 75000,
    },
    "ha_noi": {
        "registration_fee_rate": 0.006,  # Higher rate for Hanoi
        "admin_fee": 75000,
    },
    "da_nang": {
        "registration_fee_rate": 0.0055,
        "admin_fee": 60000,
    },
    "can_tho": {
        "registration_fee_rate": 0.0045,
        "admin_fee": 40000,
    },
    "hai_phong": {
        "registration_fee_rate": 0.0055,
        "admin_fee": 60000,
    }
}

# Normalize province names
PROVINCE_MAPPING = {
    "hcm": "ho_chi_minh",
    "ho chi minh": "ho_chi_minh",
    "saigon": "ho_chi_minh",
    "tp hcm": "ho_chi_minh",
    "hanoi": "ha_noi",
    "ha noi": "ha_noi",
    "danang": "da_nang",
    "da nang": "da_nang",
    "cantho": "can_tho",
    "can tho": "can_tho",
    "haiphong": "hai_phong",
    "hai phong": "hai_phong",
}


def calculate_fees(
    province: Optional[str] = None,
    contract_value: Optional[float] = None,
    extracted_fields: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate indicative fees for land transfer transaction.
    
    Args:
        province: Province name for location-specific rates
        contract_value: Contract value in VND
        extracted_fields: Additional fields from document extraction
        
    Returns:
        Dictionary containing calculated fees and details
    """
    fees = {}
    
    # Normalize province name
    normalized_province = _normalize_province(province) if province else None
    
    # Get applicable rates
    rates = _get_rates(normalized_province)
    
    # Try to get contract value from extracted fields if not provided
    if contract_value is None and extracted_fields:
        contract_value = _extract_contract_value(extracted_fields)
    
    # Default contract value if still not available
    if contract_value is None:
        contract_value = 1000000000  # 1 billion VND default
        fees["contract_value_estimated"] = True
    else:
        fees["contract_value_estimated"] = False
    
    # Calculate registration fee
    registration_fee = contract_value * rates["registration_fee_rate"]
    registration_fee = max(registration_fee, rates["registration_fee_min"])
    registration_fee = min(registration_fee, rates["registration_fee_max"])
    fees["registration_fee"] = int(registration_fee)
    
    # Calculate personal income tax (for seller)
    personal_income_tax = contract_value * rates["personal_income_tax_rate"]
    fees["personal_income_tax"] = int(personal_income_tax)
    
    # Calculate notary fee (bracket-based)
    notary_fee = _calculate_notary_fee(contract_value, rates)
    fees["notary_fee"] = int(notary_fee)
    
    # Administrative fee (flat)
    fees["admin_fee"] = int(rates["admin_fee"])
    
    # Calculate total
    total_fees = (
        fees["registration_fee"] + 
        fees["personal_income_tax"] + 
        fees["notary_fee"] + 
        fees["admin_fee"]
    )
    fees["total"] = int(total_fees)
    
    # Add metadata
    fees["meta"] = {
        "province": normalized_province,
        "contract_value": int(contract_value),
        "rates_source": "province_specific" if normalized_province in PROVINCE_OVERRIDES else "default",
        "currency": "VND",
        "calculation_date": "2024-08-15",  # Static for POC
        "disclaimer": "These are indicative fees only. Actual fees may vary."
    }
    
    logger.info(f"Calculated fees for {normalized_province or 'default'}: total {total_fees:,.0f} VND")
    
    return fees


def _normalize_province(province: str) -> Optional[str]:
    """Normalize province name to standard format"""
    if not province:
        return None
    
    province_lower = province.lower().strip()
    
    # Direct mapping
    if province_lower in PROVINCE_MAPPING:
        return PROVINCE_MAPPING[province_lower]
    
    # Partial matching
    for key, value in PROVINCE_MAPPING.items():
        if key in province_lower or province_lower in key:
            return value
    
    # If no match found, return the cleaned input
    return province_lower.replace(" ", "_")


def _get_rates(province: Optional[str]) -> Dict[str, float]:
    """Get applicable fee rates for a province"""
    rates = DEFAULT_RATES.copy()
    
    if province and province in PROVINCE_OVERRIDES:
        rates.update(PROVINCE_OVERRIDES[province])
        logger.info(f"Using province-specific rates for {province}")
    else:
        logger.info("Using default nationwide rates")
    
    return rates


def _extract_contract_value(extracted_fields: Dict[str, Any]) -> Optional[float]:
    """Extract contract value from extracted fields"""
    # Look for contract value in various field names
    value_fields = ["contract_value", "amount", "money_amounts"]
    
    for field_name in value_fields:
        if field_name in extracted_fields:
            field_value = extracted_fields[field_name]
            
            # Handle list of values
            if isinstance(field_value, list) and field_value:
                field_value = field_value[0]  # Take first value
            
            # Try to parse as number
            try:
                if isinstance(field_value, str):
                    # Clean up string (remove currency symbols, commas)
                    cleaned = field_value.replace(",", "").replace(".", "")
                    cleaned = "".join(c for c in cleaned if c.isdigit())
                    if cleaned:
                        value = float(cleaned)
                        
                        # Convert if it looks like it's in millions/billions
                        if value < 1000:  # Likely in millions
                            value = value * 1000000
                        elif value < 100000:  # Likely in hundreds of thousands
                            value = value * 100000
                        
                        return value
                elif isinstance(field_value, (int, float)):
                    return float(field_value)
            except (ValueError, TypeError):
                continue
    
    return None


def _calculate_notary_fee(contract_value: float, rates: Dict[str, float]) -> float:
    """Calculate notary fee using bracket system"""
    base_fee = rates["notary_fee_base"]
    rate = rates["notary_fee_rate"]
    
    # Simple bracket system for POC
    if contract_value <= 500000000:  # Up to 500M VND
        return base_fee + (contract_value * rate)
    elif contract_value <= 2000000000:  # 500M - 2B VND
        return base_fee + (500000000 * rate) + ((contract_value - 500000000) * rate * 0.8)
    else:  # Above 2B VND
        return base_fee + (500000000 * rate) + (1500000000 * rate * 0.8) + ((contract_value - 2000000000) * rate * 0.6)


def get_fee_breakdown_explanation() -> Dict[str, str]:
    """Get explanations for each fee type"""
    return {
        "registration_fee": "Phí đăng ký quyền sử dụng đất (0.5% giá trị hợp đồng)",
        "personal_income_tax": "Thuế thu nhập cá nhân (2% giá trị hợp đồng)",
        "notary_fee": "Phí công chứng (theo biểu phí)",
        "admin_fee": "Phí hành chính (cố định)",
        "total": "Tổng các loại phí và thuế"
    }


def estimate_processing_time(province: Optional[str] = None) -> Dict[str, Any]:
    """Estimate processing time for land transfer"""
    base_days = 15  # Default processing time
    
    # Province-specific adjustments
    if province in ["ho_chi_minh", "ha_noi"]:
        base_days = 20  # Longer in major cities
    elif province in ["can_tho", "hai_phong"]:
        base_days = 12  # Faster in smaller cities
    
    return {
        "estimated_days": base_days,
        "working_days_only": True,
        "factors": [
            "Document completeness",
            "Property verification",
            "Administrative processing",
            "Payment confirmation"
        ]
    }