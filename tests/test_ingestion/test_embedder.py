from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.ingestion.embedder import store_chunks


def _make_chunks(n: int = 3) -> list[Document]:
    return [
        Document(
            page_content=f"chunk content {i}",
            metadata={"page_number": 1, "source": "test.pdf", "chunk_id": f"test_{i:04d}"},
        )
        for i in range(n)
    ]


@patch("app.ingestion.embedder.get_vector_store")
def test_store_chunks_returns_count(mock_vs_factory):
    mock_vs = MagicMock()
    mock_vs_factory.return_value = mock_vs
    chunks = _make_chunks(3)
    result = store_chunks(chunks, company_name="Acme", filing_year=2023, doc_name="acme_10k.pdf", doc_id="abc123")
    assert result == 3


@patch("app.ingestion.embedder.get_vector_store")
def test_store_chunks_calls_add_documents(mock_vs_factory):
    mock_vs = MagicMock()
    mock_vs_factory.return_value = mock_vs
    chunks = _make_chunks(2)
    store_chunks(chunks, company_name="Acme", filing_year=2023, doc_name="acme_10k.pdf", doc_id="abc123")
    mock_vs.add_documents.assert_called_once()


@patch("app.ingestion.embedder.get_vector_store")
def test_store_chunks_enriches_metadata(mock_vs_factory):
    mock_vs = MagicMock()
    mock_vs_factory.return_value = mock_vs
    chunks = _make_chunks(1)
    store_chunks(chunks, company_name="Acme", filing_year=2023, doc_name="acme_10k.pdf", doc_id="abc123")
    added_docs = mock_vs.add_documents.call_args[0][0]
    assert added_docs[0].metadata["company_name"] == "Acme"
    assert added_docs[0].metadata["filing_year"] == 2023
    assert added_docs[0].metadata["doc_id"] == "abc123"
