from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Milvus configuration
    milvus_uri: str = Field("http://localhost:19530", description="Milvus URI, e.g. http://localhost:19530")
    milvus_token: Optional[str] = Field(None, description="Milvus token or API key if authentication is enabled")
    milvus_collection: str = Field("novels", description="Default Milvus collection name")
    milvus_database: str = Field("default", description="Milvus database name")
    milvus_consistency_level: str = Field("Bounded", description="Milvus consistency level")
    milvus_metric_type: str = Field("COSINE", description="Vector similarity metric type")

    # Embedding configuration
    embedding_model_path: Path = Field(Path("./models/qwen"), description="Local path to the Qwen embedding model directory")
    embedding_dim: int = Field(1536, description="Embedding dimension for the chosen model")
    chunk_size: int = Field(800, description="Number of characters per chunk inside a chapter")
    chunk_overlap: int = Field(120, description="Number of overlapping characters between chunks")

    # LLM configuration
    llm_base_url: str = Field("https://api.example.com/v1", description="Base URL for the OpenAPI compatible chat completion endpoint")
    llm_model_name: str = Field("qwen-max", description="Model name for the chat completion endpoint")
    llm_api_key: str = Field("changeme", description="API key for the chat completion endpoint")
    llm_temperature: float = Field(0.3, description="Sampling temperature for the chat model")
    llm_max_tokens: int = Field(512, description="Maximum tokens to generate per response")

    # Logging and service configuration
    log_directory: Path = Field(Path("logs"), description="Directory where interaction logs will be written")
    max_history_turns: int = Field(6, description="Maximum number of history turns to keep per session")


settings = Settings()
