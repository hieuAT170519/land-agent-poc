"""Fee calculation for land transfer transactions with province overrides."""
from typing import Dict, Optional, Tuple


# Province-specific overrides for fees
PROVINCE_OVERRIDES = {
    "hanoi": {
        "admin_fee": 150000,
        "name": "Hà Nội"
    },
    "ho_chi_minh": {
        "admin_fee": 120000, 
        "name": "TP.HCM"
    },
    "hcm": {  # Alternative name
        "admin_fee": 120000,
        "name": "TP.HCM"
    },
    "da_nang": {
        "admin_fee": 110000,
        "name": "Đà Nẵng"
    }
}

# Default fee structure
DEFAULT_FEES = {
    "registration_fee": 500000,  # Fixed registration fee
    "admin_fee": 100000,  # Default admin fee
    "personal_income_tax_rate": 0.02  # 2% of contract value
}


def calculate_fees(contract_value: Optional[float] = None, province: Optional[str] = None) -> Dict:
    """
    Calculate all fees for land transfer transaction.
    
    Args:
        contract_value: Contract value in VND (optional)
        province: Province name for fee overrides (optional)
        
    Returns:
        Dictionary with calculated fees and metadata
    """
    result = {
        "registration_fee": DEFAULT_FEES["registration_fee"],
        "personal_income_tax": 0,
        "notary_fee": 0,
        "admin_fee": DEFAULT_FEES["admin_fee"],
        "total": 0,
        "meta": {}
    }
    
    # Apply province overrides if specified
    if province:
        province_key = province.lower().replace(" ", "_").replace(".", "")
        if province_key in PROVINCE_OVERRIDES:
            override = PROVINCE_OVERRIDES[province_key]
            result["admin_fee"] = override["admin_fee"]
            result["meta"]["province"] = override["name"]
            result["meta"]["province_override_applied"] = True
        else:
            result["meta"]["province"] = province
            result["meta"]["province_override_applied"] = False
    
    # Calculate value-based fees if contract value is provided
    if contract_value and contract_value > 0:
        # Personal income tax (2% of contract value)
        result["personal_income_tax"] = int(contract_value * DEFAULT_FEES["personal_income_tax_rate"])
        
        # Notary fee based on contract value brackets
        result["notary_fee"] = calculate_notary_fee(contract_value)
        
        result["meta"]["contract_value"] = contract_value
    else:
        result["meta"]["message"] = "Contract value not provided - value-based fees set to zero"
    
    # Calculate total
    result["total"] = (
        result["registration_fee"] + 
        result["personal_income_tax"] + 
        result["notary_fee"] + 
        result["admin_fee"]
    )
    
    return result


def calculate_notary_fee(contract_value: float) -> int:
    """
    Calculate notary fee based on contract value brackets.
    
    Args:
        contract_value: Contract value in VND
        
    Returns:
        Notary fee in VND
    """
    if contract_value <= 0:
        return 0
    
    # Notary fee brackets (simplified structure)
    if contract_value <= 100_000_000:  # Up to 100M VND
        return int(contract_value * 0.003)  # 0.3%
    elif contract_value <= 500_000_000:  # 100M - 500M VND
        base = 100_000_000 * 0.003
        excess = (contract_value - 100_000_000) * 0.002  # 0.2%
        return int(base + excess)
    elif contract_value <= 1_000_000_000:  # 500M - 1B VND
        base = 100_000_000 * 0.003 + 400_000_000 * 0.002
        excess = (contract_value - 500_000_000) * 0.0015  # 0.15%
        return int(base + excess)
    else:  # Above 1B VND
        base = 100_000_000 * 0.003 + 400_000_000 * 0.002 + 500_000_000 * 0.0015
        excess = (contract_value - 1_000_000_000) * 0.001  # 0.1%
        return int(base + excess)


def get_supported_provinces() -> Dict[str, str]:
    """
    Get list of provinces with fee overrides.
    
    Returns:
        Dictionary mapping province keys to display names
    """
    return {key: data["name"] for key, data in PROVINCE_OVERRIDES.items()}


def format_fee_summary(fees: Dict) -> str:
    """
    Format fee calculation as human-readable summary.
    
    Args:
        fees: Fee calculation result from calculate_fees()
        
    Returns:
        Formatted summary string
    """
    lines = [
        "=== PHÍ VÀ LỆ PHÍ CHUYỂN NHƯỢNG ĐẤT ===",
        f"Lệ phí đăng ký: {fees['registration_fee']:,} VND",
        f"Thuế TNCN (2%): {fees['personal_income_tax']:,} VND",
        f"Phí công chứng: {fees['notary_fee']:,} VND",
        f"Phí hành chính: {fees['admin_fee']:,} VND",
        f"TỔNG CỘNG: {fees['total']:,} VND"
    ]
    
    if "province" in fees.get("meta", {}):
        lines.append(f"Tỉnh/TP: {fees['meta']['province']}")
    
    if "contract_value" in fees.get("meta", {}):
        lines.append(f"Giá trị hợp đồng: {fees['meta']['contract_value']:,} VND")
    
    return "\n".join(lines)