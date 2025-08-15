# Land Transfer AI Agent POC

Complete proof-of-concept for Vietnamese Land Transfer AI Agent featuring OCR → document classification → field extraction → fee calculation → RAG-based legal document search.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Git

### Windows/macOS/Linux

1. **Clone and run:**
   ```bash
   git clone https://github.com/hieuAT170519/land-agent-poc.git
   cd land-agent-poc
   docker compose up -d --build
   ```

2. **Wait for service to start** (30-60 seconds), then test:
   ```bash
   curl http://localhost:8000/health
   ```
   Expected response: `{"status":"ok"}`

3. **Build RAG index** (optional, uses sample legal texts):
   ```bash
   curl -X POST http://localhost:8000/rag/index
   ```

## API Endpoints

### Health Check
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed component status
curl http://localhost:8000/health/detailed
```

### Document Processing
Upload an image for OCR → classification → extraction → fee calculation:

```bash
# Basic processing
curl -X POST http://localhost:8000/process \
  -F "file=@your-document.jpg"

# With province and contract value
curl -X POST http://localhost:8000/process \
  -F "file=@contract.jpg" \
  -F "province=hanoi" \
  -F "contract_value=500000000"
```

**Response structure:**
```json
{
  "ocr_text": "extracted text...",
  "doc_type": "contract|certificate|receipt|unknown",
  "extracted": {
    "parties": ["Nguyễn Văn A", "Trần Thị B"],
    "id_numbers": ["123456789"],
    "addresses": ["123 Main St, Hanoi"],
    "parcel_number": "123A",
    "area_m2": "100.5",
    "date": "01/12/2023",
    "notary_office": "Phòng công chứng ABC"
  },
  "fees": {
    "registration_fee": 500000,
    "personal_income_tax": 10000000,
    "notary_fee": 1500000,
    "admin_fee": 150000,
    "total": 12150000
  },
  "meta": {
    "filename": "contract.jpg",
    "classification_confidence": {...}
  }
}
```

### RAG (Legal Document Search)

```bash
# Build search index from legal texts
curl -X POST http://localhost:8000/rag/index

# Query legal documents
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "phí đăng ký quyền sở hữu", "top_k": 3}'

# Check RAG system status
curl http://localhost:8000/rag/status
```

## Supported Features

### Document Types
- **Contracts** (Hợp đồng chuyển nhượng)
- **Certificates** (Giấy chứng nhận, Sổ đỏ)
- **Receipts** (Biên lai, Hóa đơn)

### Extracted Fields
- Party names and ID numbers
- Addresses
- Land parcel numbers
- Area in square meters
- Dates and notary offices
- Contract values

### Fee Calculation
- Registration fees: 500,000 VND
- Personal income tax: 2% of contract value
- Notary fees: Tiered based on contract value
- Administrative fees: Province-specific
  - Default: 100,000 VND
  - Hanoi: 150,000 VND
  - Ho Chi Minh City: 120,000 VND
  - Da Nang: 110,000 VND

### RAG Search
- TF-IDF based document indexing
- Cosine similarity retrieval
- Vietnamese legal text support
- Passage-level search with source tracking

## Development

### Using Makefile
```bash
# Build and run
make run

# Run tests
make test

# View logs
make logs

# Stop services
make stop

# Clean up
make clean
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires tesseract-ocr)
make dev
```

## Architecture

```
├── src/app/
│   ├── main.py           # FastAPI application
│   ├── ocr.py           # Tesseract OCR integration
│   ├── classifier.py    # Document type classification
│   ├── extractor.py     # Field extraction
│   ├── fees.py          # Fee calculation
│   └── rag/
│       ├── indexer.py   # TF-IDF index building
│       └── retriever.py # Document retrieval
├── data/legal_texts/    # Legal documents for RAG
├── .cache/rag/          # Cached index artifacts
├── Dockerfile           # Python 3.11 + tesseract
├── docker-compose.yml   # Service orchestration
└── .github/workflows/   # CI pipeline
```

## Dependencies

- **FastAPI**: Web framework
- **pytesseract**: OCR processing
- **scikit-learn**: TF-IDF vectorization
- **Pillow**: Image processing
- **uvicorn**: ASGI server

## Troubleshooting

### Service won't start
```bash
# Check logs
docker compose logs

# Rebuild
docker compose up -d --build --force-recreate
```

### OCR not working
- Ensure tesseract-ocr is installed in container
- Check supported image formats (JPG, PNG, etc.)
- Verify image quality and text clarity

### RAG queries fail
```bash
# Check if index exists
curl http://localhost:8000/rag/status

# Rebuild index
curl -X POST http://localhost:8000/rag/index
```

## License

MIT License - see LICENSE file for details.
