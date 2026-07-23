import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.audit.auditor import run_audit
from app.audit.requirements import DEFAULT_REQUIREMENTS
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import get_doc_metadata, store_chunks
from app.ingestion.loader import load_pdf
from app.models import AuditRequest, ComplianceScorecard, HealthResponse, UploadResponse

app = FastAPI(title="RAG Financial Compliance Auditor")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    filing_year: int = Form(...),
) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid4())
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        docs = load_pdf(tmp_path)
        chunks = chunk_documents(docs)
        chunks_stored = store_chunks(chunks, company_name, filing_year, file.filename, doc_id)
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(
        doc_id=doc_id,
        doc_name=file.filename,
        company=company_name,
        filing_year=filing_year,
        chunks_stored=chunks_stored,
    )


@app.post("/audit", response_model=ComplianceScorecard)
def audit(request: AuditRequest) -> ComplianceScorecard:
    metadata = get_doc_metadata(request.doc_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="doc_id not found. Upload a filing first.")

    return run_audit(
        doc_id=request.doc_id,
        company=metadata["company_name"],
        filing_year=metadata["filing_year"],
        doc_name=metadata["doc_name"],
        requirements=request.requirements or DEFAULT_REQUIREMENTS,
    )
