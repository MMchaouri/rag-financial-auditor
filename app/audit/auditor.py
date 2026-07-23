import json
from datetime import datetime, timezone
from uuid import uuid4

from thefuzz import fuzz

from app.config import settings
from app.llm.client import get_llm
from app.llm.prompts import compliance_prompt
from app.models import ComplianceFinding, ComplianceScorecard
from app.retrieval.context import assemble_context
from app.retrieval.retriever import retrieve


def _evaluate_requirement(requirement: str, doc_id: str) -> ComplianceFinding:
    docs = retrieve(requirement, doc_id=doc_id)
    context = assemble_context(docs)

    llm = get_llm()
    messages = compliance_prompt.format_messages(requirement=requirement, context=context)
    raw = llm.invoke(messages)
    data = json.loads(raw.content)
    finding = ComplianceFinding(**data)

    if finding.evidence and fuzz.partial_ratio(finding.evidence, context) < settings.grounding_min_ratio:
        finding = finding.model_copy(update={
            "status": "Insufficient Evidence",
            "confidence": 0.0,
            "evidence": "",
        })

    return finding


def run_audit(
    doc_id: str,
    company: str,
    filing_year: int,
    doc_name: str,
    requirements: list[str],
) -> ComplianceScorecard:
    findings = [_evaluate_requirement(req, doc_id) for req in requirements]

    passed = sum(1 for f in findings if f.status == "Pass")
    score = passed / len(findings) if findings else 0.0

    return ComplianceScorecard(
        id=str(uuid4()),
        company=company,
        filing_year=filing_year,
        doc_name=doc_name,
        score=score,
        findings=findings,
        created_at=datetime.now(timezone.utc),
    )
