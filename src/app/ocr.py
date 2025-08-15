"""OCR functionality using pytesseract"""

import logging
from PIL import Image
import pytesseract
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text string, empty string if OCR fails
    """
    try:
        # Open and process image
        with Image.open(image_path) as image:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='vie+eng')
            
            # Clean up text
            text = text.strip()
            
            logger.info(f"OCR extracted {len(text)} characters from {image_path}")
            return text
            
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {str(e)}")
        return ""


def preprocess_image(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Preprocess image for better OCR results.
    
    Args:
        image_path: Path to input image
        output_path: Path to save preprocessed image (optional)
        
    Returns:
        Path to preprocessed image
    """
    try:
        with Image.open(image_path) as image:
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize if image is too small
            width, height = image.size
            if width < 800 or height < 600:
                scale_factor = max(800 / width, 600 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save preprocessed image
            if output_path:
                image.save(output_path)
                return output_path
            else:
                # Save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    image.save(tmp.name)
                    return tmp.name
                    
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image_path  # Return original path if preprocessing fails