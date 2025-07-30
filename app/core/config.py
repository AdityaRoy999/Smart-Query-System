from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    max_tokens: int = 200
    batch_size: int = 100
    top_k_results: int = 5
    embedding_model: str = "models/embedding-001"
    generation_model: str = "models/gemini-2.0"

    class Config:
        env_file = ".env"

settings= Settings()
