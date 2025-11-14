"""
Pydantic models for Fortif.ai RAG API request/response validation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for querying patient memories."""

    patient_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique patient identifier",
        example="patient_123"
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Patient's question or conversation prompt",
        example="Tell me about my granddaughter's birthday"
    )
    include_sensitive: bool = Field(
        default=False,
        description="Whether to include sensitive memories in retrieval"
    )
    emotion_filter: Optional[str] = Field(
        default=None,
        description="Filter memories by emotion type",
        example="positive"
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of memories to retrieve"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        """Ensure question is not empty after stripping."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only")
        return cleaned

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, v):
        """Ensure patient_id is properly formatted."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Patient ID cannot be empty")
        return cleaned

    @field_validator("emotion_filter")
    @classmethod
    def validate_emotion(cls, v):
        """Validate emotion filter value."""
        if v is not None:
            valid_emotions = ["positive", "negative", "neutral", "mixed", "unknown"]
            if v not in valid_emotions:
                raise ValueError(
                    f"Emotion must be one of: {', '.join(valid_emotions)}"
                )
        return v


class SourceDocument(BaseModel):
    """A single retrieved source document."""

    text: str = Field(..., description="Memory text content")
    topic: str = Field(..., description="Memory topic category")
    emotion: str = Field(..., description="Associated emotion")
    source: str = Field(..., description="Source of the memory")
    distance: float = Field(..., description="Vector similarity distance (lower = more relevant)")
    chunk_index: int = Field(..., description="Chunk index if memory was split")
    total_chunks: int = Field(..., description="Total number of chunks for this memory")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    response: str = Field(
        ...,
        description="Generated empathetic response from Fortif.ai"
    )
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Retrieved source memories used to generate response"
    )
    patient_id: str = Field(..., description="Patient identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model used, retrieval count, etc.)"
    )


class IngestRequest(BaseModel):
    """Request model for ingesting new patient memories."""

    patient_id: str = Field(..., min_length=1, description="Patient identifier")
    raw_text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Raw memory text to ingest"
    )
    source: str = Field(
        ...,
        description="Source of the memory (e.g., 'family_questionnaire', 'ehr_note')",
        example="family_questionnaire"
    )
    topic: str = Field(
        ...,
        description="Topic category (e.g., 'daily_routine', 'positive_memory')",
        example="positive_memory"
    )
    is_sensitive: bool = Field(
        default=False,
        description="Whether this memory contains sensitive information"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Named entities mentioned (people, places, things)",
        example=["Sarah (granddaughter)", "oak tree park"]
    )
    emotion: str = Field(
        ...,
        description="Emotional valence of the memory",
        example="positive"
    )

    @field_validator("raw_text")
    @classmethod
    def validate_text(cls, v):
        """Validate text content."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Memory text cannot be empty")
        if len(cleaned) < 10:
            raise ValueError("Memory text must be at least 10 characters")
        return cleaned

    @field_validator("emotion")
    @classmethod
    def validate_emotion(cls, v):
        """Validate emotion value."""
        valid_emotions = ["positive", "negative", "neutral", "mixed", "unknown"]
        if v not in valid_emotions:
            raise ValueError(
                f"Emotion must be one of: {', '.join(valid_emotions)}"
            )
        return v


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""

    status: str = Field(..., description="Ingestion status")
    patient_id: str = Field(..., description="Patient identifier")
    chunks_created: int = Field(..., description="Number of chunks created")
    objects_upserted: int = Field(..., description="Number of objects added to Weaviate")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Overall API status")
    weaviate_connected: bool = Field(..., description="Weaviate connection status")
    collection_exists: bool = Field(default=True, description="Whether collection exists")
    collection_count: Optional[int] = Field(
        None,
        description="Number of objects in collection"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Detailed error message")
    path: Optional[str] = Field(None, description="Request path that caused error")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
