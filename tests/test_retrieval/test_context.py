from langchain_core.documents import Document
from app.retrieval.context import assemble_context


def test_assemble_context_includes_page_numbers():
    docs = [
        Document(page_content="risk disclosure text", metadata={"page_number": 12}),
        Document(page_content="liquidity text", metadata={"page_number": 45}),
    ]
    result = assemble_context(docs)
    assert "[Page 12]" in result
    assert "[Page 45]" in result


def test_assemble_context_includes_content():
    docs = [Document(page_content="revenue recognition policy", metadata={"page_number": 1})]
    result = assemble_context(docs)
    assert "revenue recognition policy" in result


def test_assemble_context_separates_chunks():
    docs = [
        Document(page_content="chunk one", metadata={"page_number": 1}),
        Document(page_content="chunk two", metadata={"page_number": 2}),
    ]
    result = assemble_context(docs)
    assert "---" in result


def test_assemble_context_handles_missing_page():
    docs = [Document(page_content="some text", metadata={})]
    result = assemble_context(docs)
    assert "[Page ?]" in result
