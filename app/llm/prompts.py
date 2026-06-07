from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """\
You are a financial compliance auditor reviewing SEC filings.
You will be given a compliance requirement and excerpts from a financial filing.
Respond ONLY with valid JSON matching this exact schema:
{{
  "requirement": "<the requirement text>",
  "status": "Pass" | "Fail" | "Insufficient Evidence",
  "confidence": <float 0.0-1.0>,
  "page": <int or null>,
  "evidence": "<verbatim quote from the provided context, or empty string>"
}}
Rules:
- evidence MUST be a verbatim quote from the provided context.
- If no relevant evidence exists, set status to "Insufficient Evidence" and evidence to "".
- confidence reflects certainty given the evidence found.
- page MUST be the integer from the [Page N] tag nearest to the evidence; set to null if no tag is present.
- Do not introduce any information not present in the provided context.
- Output raw JSON only — no markdown fences, no commentary."""

_HUMAN = """\
Compliance Requirement: {requirement}

Filing Excerpts:
{context}

Evaluate whether this requirement is met. Output JSON only."""

compliance_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])
