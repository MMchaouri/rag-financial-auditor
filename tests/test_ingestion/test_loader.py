from app.ingestion.loader import load_pdf


def test_load_pdf_returns_documents(sample_pdf_path):
    docs = load_pdf(sample_pdf_path)
    assert len(docs) >= 1


def test_load_pdf_documents_have_page_number(sample_pdf_path):
    docs = load_pdf(sample_pdf_path)
    for doc in docs:
        assert "page_number" in doc.metadata
        assert isinstance(doc.metadata["page_number"], int)


def test_load_pdf_documents_have_source(sample_pdf_path):
    docs = load_pdf(sample_pdf_path)
    for doc in docs:
        assert "source" in doc.metadata


def test_load_pdf_content_is_nonempty(sample_pdf_path):
    docs = load_pdf(sample_pdf_path)
    for doc in docs:
        assert doc.page_content.strip() != ""
