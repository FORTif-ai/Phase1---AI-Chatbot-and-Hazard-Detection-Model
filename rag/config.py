"""
Configuration management for Fortif.ai RAG API.
Uses Pydantic Settings for environment variable validation.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
import os

# Get the directory of the current file (config.py), which ensures the path is relative to the config file, not the CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env") # Joins the directory path with the filename ".env"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # === Weaviate Configuration ===
    weaviate_host: str = Field(default="localhost", description="Weaviate server host")
    weaviate_port: int = Field(default=8080, description="Weaviate HTTP port")
    weaviate_grpc_port: int = Field(default=50051, description="Weaviate gRPC port")
    weaviate_collection_name: str = Field(
        default="FortifAiMasterMemory",
        description="Weaviate collection name"
    )

    # === Google AI Configuration ===
    google_api_key: str = Field(..., description="Google AI API key (required)")
    embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Google embedding model"
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Google LLM model for generation"
    )
    vector_dimension: int = Field(
        default=768,
        description="Embedding vector dimension"
    )

    # === RAG Configuration ===
    default_retrieval_limit: int = Field(
        default=10,
        ge=1,
        le=10,
        description="Default number of documents to retrieve"
    )
    llm_temperature: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0.25 = consistent, safer for healthcare)"
    )
    max_context_length: int = Field(
        default=2000,
        description="Maximum context length in characters"
    )

    # === API Configuration ===
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (optional for development)"
    )
    api_title: str = Field(
        default="Fortif.ai RAG API",
        description="API title"
    )
    api_version: str = Field(
        default="2.0.0",
        description="API version"
    )
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Allowed CORS origins (comma-separated)"
    )
    api_url: str = Field(
        default="http://localhost:8000",
        description="API URL"
    )

    # === Ingestion Configuration ===
    chunk_size: int = Field(default=500, description="Text chunk size")
    chunk_overlap: int = Field(default=50, description="Text chunk overlap")
    default_batch_size: int = Field(default=100, description="Batch size for ingestion")

    # === Retry Configuration ===
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=2, description="Retry delay in seconds")

    @field_validator("google_api_key")
    @classmethod
    def validate_api_key(cls, v):
        """Ensure Google API key is provided."""
        if not v or v.strip() == "":
            raise ValueError("GOOGLE_API_KEY environment variable must be set")
        return v

    def get_cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()
