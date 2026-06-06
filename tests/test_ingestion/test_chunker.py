from langchain_core.documents import Document
from app.ingestion.chunker import chunk_documents


def _make_doc(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"page_number": page, "source": "test.pdf"})


def test_chunk_documents_returns_nonempty_list():
    docs = [_make_doc("word " * 600)]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


def test_chunks_have_chunk_id():
    docs = [_make_doc("word " * 600)]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert isinstance(chunk.metadata["chunk_id"], str)


def test_chunks_preserve_source_metadata():
    docs = [_make_doc("word " * 600)]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"


def test_single_short_doc_produces_one_chunk():
    docs = [_make_doc("short text")]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
