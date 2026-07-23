from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app
from app.models import ComplianceScorecard

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_rejects_non_pdf():
    resp = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"company_name": "Acme Corp", "filing_year": 2023},
    )
    assert resp.status_code == 400


def test_upload_stores_chunks(sample_pdf_path):
    with open(sample_pdf_path, "rb") as f, \
         patch("app.api.main.store_chunks", return_value=5) as mock_store:
        resp = client.post(
            "/upload",
            files={"file": ("acme-10k.pdf", f, "application/pdf")},
            data={"company_name": "Acme Corp", "filing_year": 2023},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["company"] == "Acme Corp"
    assert body["filing_year"] == 2023
    assert body["chunks_stored"] == 5
    mock_store.assert_called_once()


def test_audit_returns_404_for_unknown_doc():
    with patch("app.api.main.get_doc_metadata", return_value=None):
        resp = client.post("/audit", json={"doc_id": "missing"})
    assert resp.status_code == 404


def test_audit_returns_scorecard_for_known_doc():
    metadata = {"company_name": "Acme Corp", "filing_year": 2023, "doc_name": "acme-10k.pdf"}
    scorecard = ComplianceScorecard(
        id="report-1",
        company="Acme Corp",
        filing_year=2023,
        doc_name="acme-10k.pdf",
        score=1.0,
        findings=[],
        created_at=datetime.now(timezone.utc),
    )
    with patch("app.api.main.get_doc_metadata", return_value=metadata), \
         patch("app.api.main.run_audit", return_value=scorecard):
        resp = client.post("/audit", json={"doc_id": "doc-1"})

    assert resp.status_code == 200
    assert resp.json()["company"] == "Acme Corp"
