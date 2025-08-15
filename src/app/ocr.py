"""OCR functionality using pytesseract."""
import logging
from io import BytesIO
from typing import Optional

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text_from_image(image_data: bytes) -> str:
    """
    Extract text from image using OCR.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Extracted text string, empty string if error occurs
    """
    try:
        # Open image from bytes
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using pytesseract
        # Use Vietnamese language if available, fallback to English
        try:
            text = pytesseract.image_to_string(image, lang='vie+eng')
        except pytesseract.TesseractError:
            # Fallback to English only if Vietnamese is not available
            text = pytesseract.image_to_string(image, lang='eng')
        
        return text.strip()
        
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return ""


def is_tesseract_available() -> bool:
    """Check if tesseract is available and working."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False