import pytest
from fpdf import FPDF
from pathlib import Path


@pytest.fixture
def sample_pdf_path(tmp_path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    lines = [
        "ITEM 1A. RISK FACTORS",
        "The company faces significant market, operational, and regulatory risks.",
        "Competition in our industry is intense and may reduce our margins.",
        "ITEM 3. LEGAL PROCEEDINGS",
        "The company is not party to any material legal proceedings as of this date.",
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS",
        "Revenue increased 12 percent year-over-year driven by strong product demand.",
        "Operating income improved due to cost discipline and scale efficiencies.",
        "LIQUIDITY AND CAPITAL RESOURCES",
        "As of December 31 2023 cash and equivalents totaled 5.2 billion dollars.",
        "We believe our current liquidity is sufficient to fund operations for 12 months.",
        "REVENUE RECOGNITION",
        "Revenue is recognized when performance obligations are satisfied per ASC 606.",
        "We allocate transaction price to each distinct performance obligation.",
        "INDEPENDENT AUDITORS REPORT",
        "Ernst and Young LLP served as independent registered public accounting firm.",
        "In our opinion the financial statements present fairly in all material respects.",
    ]
    for line in lines:
        pdf.cell(0, 8, text=line, new_x="LMARGIN", new_y="NEXT")
    out = tmp_path / "test_filing.pdf"
    pdf.output(str(out))
    return out
