import json
from unittest.mock import patch

from langchain_core.documents import Document

from app.audit.auditor import run_audit


class FakeLLM:
    def __init__(self, response: dict):
        self._response = response

    def invoke(self, _messages):
        return type("FakeMessage", (), {"content": json.dumps(self._response)})()


def _fake_docs():
    return [Document(page_content="Revenue increased 12 percent year-over-year.", metadata={"page_number": 7})]


def test_finding_kept_when_evidence_is_grounded():
    response = {
        "requirement": "Revenue growth must be disclosed",
        "status": "Pass",
        "confidence": 0.9,
        "page": 7,
        "evidence": "Revenue increased 12 percent year-over-year.",
    }
    with patch("app.audit.auditor.retrieve", return_value=_fake_docs()), \
         patch("app.audit.auditor.get_llm", return_value=FakeLLM(response)):
        scorecard = run_audit(
            doc_id="doc-1",
            company="Acme Corp",
            filing_year=2023,
            doc_name="acme-10k.pdf",
            requirements=["Revenue growth must be disclosed"],
        )

    assert scorecard.score == 1.0
    assert scorecard.findings[0].status == "Pass"
    assert scorecard.findings[0].evidence == "Revenue increased 12 percent year-over-year."


def test_finding_downgraded_when_evidence_is_not_grounded():
    response = {
        "requirement": "Revenue growth must be disclosed",
        "status": "Pass",
        "confidence": 0.9,
        "page": 7,
        "evidence": "The company acquired three competitors this year.",
    }
    with patch("app.audit.auditor.retrieve", return_value=_fake_docs()), \
         patch("app.audit.auditor.get_llm", return_value=FakeLLM(response)):
        scorecard = run_audit(
            doc_id="doc-1",
            company="Acme Corp",
            filing_year=2023,
            doc_name="acme-10k.pdf",
            requirements=["Revenue growth must be disclosed"],
        )

    finding = scorecard.findings[0]
    assert finding.status == "Insufficient Evidence"
    assert finding.confidence == 0.0
    assert finding.evidence == ""
    assert scorecard.score == 0.0


def test_score_is_zero_for_no_requirements():
    with patch("app.audit.auditor.retrieve", return_value=_fake_docs()):
        scorecard = run_audit(
            doc_id="doc-1",
            company="Acme Corp",
            filing_year=2023,
            doc_name="acme-10k.pdf",
            requirements=[],
        )

    assert scorecard.score == 0.0
    assert scorecard.findings == []
