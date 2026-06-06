from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.retrieval.retriever import retrieve


def _make_doc(text: str, doc_id: str = "abc123") -> Document:
    return Document(page_content=text, metadata={"doc_id": doc_id, "page_number": 1})


@patch("app.retrieval.retriever.get_vector_store")
def test_retrieve_returns_documents(mock_vs_factory):
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [_make_doc("risk factors text")]
    mock_vs_factory.return_value = mock_vs
    results = retrieve("risk factors", "abc123")
    assert len(results) == 1
    assert results[0].page_content == "risk factors text"


@patch("app.retrieval.retriever.get_vector_store")
def test_retrieve_passes_doc_id_filter(mock_vs_factory):
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    mock_vs_factory.return_value = mock_vs
    retrieve("revenue recognition", "xyz789")
    call_kwargs = mock_vs.similarity_search.call_args[1]
    assert call_kwargs["filter"] == {"doc_id": "xyz789"}


@patch("app.retrieval.retriever.get_vector_store")
def test_retrieve_uses_configured_top_k(mock_vs_factory):
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    mock_vs_factory.return_value = mock_vs
    retrieve("liquidity", "abc123")
    call_kwargs = mock_vs.similarity_search.call_args[1]
    assert call_kwargs["k"] == 5  # default from settings
