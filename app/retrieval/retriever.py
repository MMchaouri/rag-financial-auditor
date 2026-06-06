from langchain_core.documents import Document
from app.ingestion.embedder import get_vector_store
from app.config import settings


def retrieve(query: str, doc_id: str, top_k: int | None = None) -> list[Document]:
    k = top_k if top_k is not None else settings.retrieval_top_k
    vs = get_vector_store()
    return vs.similarity_search(query, k=k, filter={"doc_id": doc_id})
