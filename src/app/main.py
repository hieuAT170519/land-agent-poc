"""Main FastAPI application for Land Transfer AI Agent"""

from typing import Optional, Dict, Any
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import logging

from .ocr import extract_text_from_image
from .classifier import classify_document
from .extractor import extract_fields
from .fees import calculate_fees
from .rag.indexer import build_index
from .rag.retriever import query_documents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Land Transfer AI Agent POC",
    description="OCR → classify → extract → fee calculation → RAG",
    version="0.1.0"
)


class HealthResponse(BaseModel):
    status: str


class ProcessResponse(BaseModel):
    ocr_text: str
    doc_type: str
    extracted: Dict[str, Any]
    fees: Dict[str, Any]
    meta: Dict[str, Any]


class RAGQueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class RAGQueryResponse(BaseModel):
    passages: list
    meta: Dict[str, Any]


class RAGIndexResponse(BaseModel):
    files_processed: int
    tokens_count: int
    status: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok")


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    province: Optional[str] = Form(None),
    contract_value: Optional[float] = Form(None)
):
    """Process an uploaded document image"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # OCR
            ocr_text = extract_text_from_image(tmp_path)
            
            # Classification
            doc_type = classify_document(ocr_text)
            
            # Field extraction
            extracted = extract_fields(ocr_text, doc_type)
            
            # Fee calculation
            fees = calculate_fees(province, contract_value, extracted)
            
            # Prepare response
            response = ProcessResponse(
                ocr_text=ocr_text,
                doc_type=doc_type,
                extracted=extracted,
                fees=fees,
                meta={
                    "province": province,
                    "contract_value": contract_value
                }
            )
            
            return response
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/rag/index", response_model=RAGIndexResponse)
async def index_documents():
    """Build TF-IDF index from legal texts"""
    try:
        files_processed, tokens_count = build_index()
        return RAGIndexResponse(
            files_processed=files_processed,
            tokens_count=tokens_count,
            status="completed"
        )
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Query the RAG system"""
    try:
        passages = query_documents(request.question, request.top_k)
        return RAGQueryResponse(
            passages=passages,
            meta={
                "question": request.question,
                "top_k": request.top_k,
                "results_count": len(passages)
            }
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="Index not found. Please run /rag/index first to build the index."
        )
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)