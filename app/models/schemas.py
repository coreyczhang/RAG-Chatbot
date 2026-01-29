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
    rag_sources: Optional[List[Dict[str, Any]]] = Field(default=None)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation")
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
    message: Message
    session_id: str
    total_messages_in_session: int


class SessionCreateRequest(BaseModel):
    title: Optional[str] = Field(None, description="Optional session title")


class SessionCreateResponse(BaseModel):
    session_id: str
    title: str
    message: str


class SessionListItem(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionListItem]
    total: int


class SessionInfoResponse(BaseModel):
    session_id: str
    title: str
    messages: List[Message]
    total_messages: int
    created_at: datetime
    updated_at: datetime


class SessionUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
