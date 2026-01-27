from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid as uuid_lib


class HealthResponse(BaseModel):
    status: str
    message: str


class IngestResponse(BaseModel):
    chunks_added: int
    table_name: str
    match_function: str


class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid_lib.uuid4()))
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    rag_sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="RAG sources used for this message")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    k: int = Field(4, ge=1, le=20)
    filter: Optional[Dict[str, Any]] = None
    match_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    model: Optional[str] = None
    max_output_tokens: Optional[int] = Field(None, ge=50, le=4000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class SourceChunk(BaseModel):
    content_preview: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    message: Message = Field(..., description="Assistant's response message with UUID")
    session_id: str
    total_messages_in_session: int


class SessionCreateResponse(BaseModel):
    session_id: str
    message: str


class SessionInfoResponse(BaseModel):
    session_id: str
    messages: List[Message]
    total_messages: int
    created_at: datetime
    last_accessed: datetime
