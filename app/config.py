from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chroma_persist_dir: str = "./data/chroma"
    reports_dir: str = "./data/reports"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_model: str = "mistral:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5
    confidence_threshold: float = 0.6
    grounding_min_ratio: int = 75


settings = Settings()
