from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class HealthResponse(BaseModel):
    status: str
    message: str


class IngestResponse(BaseModel):
    chunks_added: int
    table_name: str
    match_function: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question or query")
    k: int = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve")
    filter: Dict[str, Any] = Field(default_factory=dict, description="Metadata filter")
    match_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    model: Optional[str] = Field(default=None)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class SourceChunk(BaseModel):
    content_preview: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
