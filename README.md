# Land Transfer AI Agent POC

A Proof of Concept for an AI-powered land transfer document processing system that performs OCR, document classification, field extraction, fee calculation, and provides a lightweight RAG (Retrieval-Augmented Generation) system for legal text queries.

## Architecture Overview

```
Upload Image → OCR → Classify → Extract Fields → Calculate Fees
                                     ↓
Legal Texts → TF-IDF Index → RAG Query System
```

### Key Components

- **OCR**: pytesseract with Vietnamese and English language support
- **Document Classification**: Heuristic keyword-based classification
- **Field Extraction**: Regex and pattern-based extraction
- **Fee Calculation**: Province-specific fee computation with nationwide defaults
- **RAG System**: TF-IDF vectorization with cosine similarity search

## API Endpoints

### Health Check
```bash
GET /health
```
Returns: `{"status": "ok"}`

### Document Processing
```bash
POST /process
Content-Type: multipart/form-data

Parameters:
- file: image file (jpg/png)
- province: optional string
- contract_value: optional number
```

Returns:
```json
{
  "ocr_text": "extracted text from image",
  "doc_type": "contract|certificate|receipt|application|id_document|unknown",
  "extracted": {
    "id_numbers": ["123456789"],
    "dates": ["15/08/2024"],
    "addresses": ["123 Main St, District 1, HCMC"],
    "area_m2": ["100.5"],
    "parties": ["Nguyen Van A", "Tran Thi B"]
  },
  "fees": {
    "registration_fee": 5000000,
    "personal_income_tax": 20000000,
    "notary_fee": 3000000,
    "admin_fee": 75000,
    "total": 28075000,
    "meta": {
      "province": "ho_chi_minh",
      "contract_value": 1000000000,
      "currency": "VND"
    }
  },
  "meta": {
    "province": "ho_chi_minh",
    "contract_value": 1000000000
  }
}
```

### RAG System

#### Build Index
```bash
POST /rag/index
```
Builds TF-IDF index from all `.txt` files in `data/legal_texts/`

Returns:
```json
{
  "files_processed": 5,
  "tokens_count": 15000,
  "status": "completed"
}
```

#### Query Documents
```bash
POST /rag/query
Content-Type: application/json

{
  "question": "What are the requirements for land transfer?",
  "top_k": 5
}
```

Returns:
```json
{
  "passages": [
    {
      "passage": "Relevant text passage...",
      "source_file": "land_law_2024.txt",
      "passage_index": 5,
      "similarity_score": 0.85,
      "rank": 1
    }
  ],
  "meta": {
    "question": "What are the requirements for land transfer?",
    "top_k": 5,
    "results_count": 3
  }
}
```

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and build**:
```bash
git clone <repository-url>
cd land-agent-poc
docker compose build
```

2. **Start the service**:
```bash
docker compose up -d
```

3. **Test the health endpoint**:
```bash
curl http://localhost:8000/health
```

4. **Test document processing** (with a sample image):
```bash
curl -X POST http://localhost:8000/process \
  -F "file=@path/to/your/image.jpg" \
  -F "province=ha_noi" \
  -F "contract_value=1000000000"
```

5. **Add legal texts and build index**:
```bash
# Add your .txt files to data/legal_texts/
cp your_legal_document.txt data/legal_texts/

# Build the index
curl -X POST http://localhost:8000/rag/index
```

6. **Query the RAG system**:
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "land transfer fees", "top_k": 3}'
```

### Using Makefile

```bash
# Build and start
make build
make up

# Run smoke tests
make test

# View logs
make logs

# Stop services
make down

# Clean up
make clean
```

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install tesseract-ocr** (Ubuntu/Debian):
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-vie libtesseract-dev
```

3. **Run development server**:
```bash
make dev
# or
PYTHONPATH=. uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Adding Legal Texts

1. Place `.txt` files in the `data/legal_texts/` directory
2. Call the `/rag/index` endpoint to rebuild the index
3. The system will automatically split documents into passages and create TF-IDF vectors

Example legal text structure:
```
data/legal_texts/
├── land_law_2024.txt
├── registration_procedures.txt
├── fee_regulations.txt
└── transfer_requirements.txt
```

## Document Types Supported

- **Contract** (`contract`): Land transfer agreements, purchase contracts
- **Certificate** (`certificate`): Land use rights certificates, ownership documents
- **Receipt** (`receipt`): Fee receipts, payment confirmations
- **Application** (`application`): Registration applications, declarations
- **ID Document** (`id_document`): Identity cards, passports
- **Unknown** (`unknown`): Unclassified documents

## Fee Calculation

### Default Rates (Nationwide)
- Registration fee: 0.5% of contract value (min: 100K VND, max: 50M VND)
- Personal income tax: 2% of contract value
- Notary fee: Bracket-based calculation
- Administrative fee: 50K VND

### Province-Specific Rates
- **Ho Chi Minh City & Hanoi**: Higher registration rate (0.6%), admin fee 75K VND
- **Da Nang & Hai Phong**: Medium rate (0.55%), admin fee 60K VND
- **Can Tho**: Lower rate (0.45%), admin fee 40K VND

## Development

### Project Structure
```
├── .github/workflows/ci.yml    # CI/CD pipeline
├── Dockerfile                  # Container definition
├── docker-compose.yml         # Service orchestration
├── Makefile                   # Common commands
├── requirements.txt           # Python dependencies
├── src/app/                   # Application code
│   ├── main.py               # FastAPI application
│   ├── ocr.py               # OCR functionality
│   ├── classifier.py        # Document classification
│   ├── extractor.py         # Field extraction
│   ├── fees.py              # Fee calculation
│   └── rag/                 # RAG system
│       ├── indexer.py       # TF-IDF indexing
│       └── retriever.py     # Document retrieval
└── data/legal_texts/        # Legal documents for RAG
```

### Running Tests

The CI pipeline automatically tests:
- Docker image builds successfully
- Health endpoint returns 200 with correct JSON
- RAG indexing works (when legal texts are available)
- Document processing endpoint accepts requests

### Dependencies

Core dependencies:
- `fastapi`: Web framework
- `uvicorn[standard]`: ASGI server
- `pydantic`: Data validation
- `python-multipart`: File upload support
- `pillow`: Image processing
- `pytesseract`: OCR engine
- `scikit-learn`: TF-IDF vectorization
- `joblib`: Model persistence
- `numpy`: Numerical operations

## Limitations & Notes

- This is a POC with simplified heuristic-based classification and extraction
- OCR accuracy depends on image quality and Vietnamese text recognition
- Fee calculations are indicative only; actual fees may vary
- RAG system uses basic TF-IDF; production systems might use more advanced embeddings
- No authentication or rate limiting implemented
- Error handling is basic; production systems need more robust error management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure tests pass: `make test`
5. Submit a pull request

## License

This is a Proof of Concept for demonstration purposes.
