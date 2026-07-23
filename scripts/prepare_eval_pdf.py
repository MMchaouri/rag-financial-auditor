"""
Fetch a real SEC EDGAR filing (HTML) and convert it to a plain PDF so it can
be run through the existing /upload pipeline, which only reads PDF.

Usage:
    python scripts/prepare_eval_pdf.py <edgar_url> <output.pdf>

Example:
    python scripts/prepare_eval_pdf.py \\
        https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm \\
        data/eval/aapl-10k.pdf
"""
import html
import re
import sys
import urllib.request

from fpdf import FPDF
from fpdf.enums import XPos, YPos

SEC_USER_AGENT = "research (contact: replace-with-your-email@example.com)"

REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}

JUNK_PATTERNS = [
    re.compile(r"^(aapl|us-gaap|dei|xbrli|iso4217|srt|ecd):"),
    re.compile(r"^https?://"),
    re.compile(r"^\d{10}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
]


def fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def strip_tags(raw_html: str) -> list[str]:
    raw_html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    raw_html = re.sub(r"<[^>]+>", "\n", raw_html)
    text = html.unescape(raw_html)
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_junk(line: str) -> bool:
    if " " not in line and len(line) > 30:
        return True
    return any(pattern.match(line) for pattern in JUNK_PATTERNS)


def sanitize(line: str) -> str:
    for target, replacement in REPLACEMENTS.items():
        line = line.replace(target, replacement)
    return line.encode("latin-1", "replace").decode("latin-1")


def build_pdf(lines: list[str], output_path: str) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)

    for line in lines:
        pdf.multi_cell(0, 5, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(output_path)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    url, output_path = sys.argv[1], sys.argv[2]
    raw_html = fetch_html(url)
    lines = [line for line in strip_tags(raw_html) if not is_junk(line)]
    build_pdf(lines, output_path)
    print(f"wrote {len(lines)} lines to {output_path}")


if __name__ == "__main__":
    main()
