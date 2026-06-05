from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class ComplianceFinding(BaseModel):
    requirement: str
    status: Literal["Pass", "Fail", "Insufficient Evidence"]
    confidence: float = Field(ge=0.0, le=1.0)
    page: int | None = None
    evidence: str


class ComplianceScorecard(BaseModel):
    id: str
    company: str
    filing_year: int
    doc_name: str
    score: float
    findings: list[ComplianceFinding]
    created_at: datetime


class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    company: str
    filing_year: int
    chunks_stored: int


class AuditRequest(BaseModel):
    doc_id: str


class AuditResponse(BaseModel):
    report_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
