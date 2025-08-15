"""FastAPI main application for Land Transfer AI Agent POC."""
import logging
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.ocr import extract_text_from_image, is_tesseract_available
from app.classifier import classify_document, get_classification_confidence
from app.extractor import extract_fields
from app.fees import calculate_fees
from app.rag.indexer import build_index, is_index_available, get_index_stats
from app.rag.retriever import query_documents, is_retriever_ready

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Land Transfer AI Agent POC",
    description="OCR → classify → extract → fee calculation → RAG for Vietnamese land transfer documents",
    version="0.1.0"
)


# Pydantic models
class RAGQuery(BaseModel):
    question: str
    top_k: Optional[int] = 5


class ProcessResponse(BaseModel):
    ocr_text: str
    doc_type: str
    extracted: Dict[str, Any]
    fees: Dict[str, Any]
    meta: Dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    tesseract_ok = is_tesseract_available()
    rag_ready = is_retriever_ready()
    index_available = is_index_available()
    
    return {
        "status": "ok",
        "components": {
            "tesseract": "available" if tesseract_ok else "unavailable",
            "rag_index": "available" if index_available else "not_built",
            "rag_retriever": "ready" if rag_ready else "not_ready"
        }
    }


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    province: Optional[str] = Form(None),
    contract_value: Optional[float] = Form(None)
):
    """
    Process uploaded document through OCR → classify → extract → fee calculation.
    
    Args:
        file: Uploaded image file
        province: Province name for fee calculation (optional)
        contract_value: Contract value in VND (optional)
        
    Returns:
        ProcessResponse with all processing results
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # OCR extraction
        logger.info(f"Processing file: {file.filename}")
        ocr_text = extract_text_from_image(file_content)
        
        # Document classification
        doc_type = classify_document(ocr_text)
        confidence_scores = get_classification_confidence(ocr_text)
        
        # Field extraction
        extracted_fields = extract_fields(ocr_text, doc_type)
        
        # Fee calculation
        fees = calculate_fees(contract_value, province)
        
        # Build response
        response = ProcessResponse(
            ocr_text=ocr_text,
            doc_type=doc_type,
            extracted=extracted_fields,
            fees=fees,
            meta={
                "filename": file.filename,
                "file_size": len(file_content),
                "classification_confidence": confidence_scores,
                "ocr_available": is_tesseract_available(),
                "processing_status": "success"
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/rag/index")
async def build_rag_index():
    """
    Build TF-IDF index from legal texts in data/legal_texts directory.
    
    Returns:
        Status of indexing operation
    """
    try:
        logger.info("Starting RAG index build")
        
        success = build_index()
        
        if success:
            stats = get_index_stats()
            return {
                "status": "success",
                "message": "Index built successfully",
                "stats": stats
            }
        else:
            return JSONResponse(
                status_code=200,  # Don't fail CI if no texts
                content={
                    "status": "warning",
                    "message": "No legal texts found to index",
                    "stats": {"available": False}
                }
            )
            
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


@app.post("/rag/query")
async def query_rag(query: RAGQuery):
    """
    Query the RAG index for relevant passages.
    
    Args:
        query: RAG query with question and optional top_k
        
    Returns:
        Top-K relevant passages with sources and scores
    """
    try:
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not is_index_available():
            raise HTTPException(
                status_code=400, 
                detail="RAG index not available. Please build index first using POST /rag/index"
            )
        
        results = query_documents(query.question, query.top_k or 5)
        
        return {
            "question": query.question,
            "top_k": query.top_k or 5,
            "results": results,
            "total_found": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/rag/status")
async def rag_status():
    """
    Get status of RAG system.
    
    Returns:
        RAG system status and statistics
    """
    index_stats = get_index_stats()
    retriever_ready = is_retriever_ready()
    
    return {
        "index": index_stats,
        "retriever_ready": retriever_ready,
        "status": "ready" if index_stats.get("available") and retriever_ready else "not_ready"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Land Transfer AI Agent POC")
    
    # Check tesseract availability
    if is_tesseract_available():
        logger.info("Tesseract OCR is available")
    else:
        logger.warning("Tesseract OCR is not available")
    
    # Create cache directory
    os.makedirs(".cache/rag", exist_ok=True)
    
    logger.info("Application startup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)