from pathlib import Path
from typing import Union
from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(file_path: Union[str, Path]) -> list[Document]:
    reader = PdfReader(str(file_path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page_number": i + 1, "source": str(file_path)},
            ))
    return docs
