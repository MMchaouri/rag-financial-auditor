from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import settings


def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name="filings",
        embedding_function=get_embedding_function(),
        persist_directory=settings.chroma_persist_dir,
    )


def store_chunks(
    chunks: list[Document],
    company_name: str,
    filing_year: int,
    doc_name: str,
    doc_id: str,
) -> int:
    for chunk in chunks:
        chunk.metadata.update({
            "company_name": company_name,
            "filing_year": filing_year,
            "doc_name": doc_name,
            "doc_id": doc_id,
        })
    vs = get_vector_store()
    vs.add_documents(chunks)
    return len(chunks)
