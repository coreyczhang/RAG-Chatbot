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


class Agent(BaseModel):
    agent_id: str
    name: str
    description: Optional[str] = None
    system_prompt: str
    model: str = 'gpt-4o-mini'
    temperature: float = 0.4
    max_output_tokens: int = 400
    avatar_emoji: Optional[str] = None
    created_at: datetime


class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    system_prompt: str = Field(..., min_length=10)
    model: str = Field('gpt-4o-mini')
    temperature: float = Field(0.4, ge=0.0, le=2.0)
    max_output_tokens: int = Field(400, ge=50, le=4000)
    avatar_emoji: Optional[str] = None


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    system_prompt: Optional[str] = Field(None, min_length=10)
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(None, ge=50, le=4000)
    avatar_emoji: Optional[str] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(None)
    agent_id: Optional[str] = Field(None, description="Agent to use for this conversation")
    k: int = Field(4, ge=1, le=20)
    filter: Optional[Dict[str, Any]] = None
    match_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Optional overrides (use agent settings if not provided)
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


class SourceChunk(BaseModel):
    content_preview: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    message: Message
    session_id: str
    agent: Agent
    total_messages_in_session: int


class SessionCreateRequest(BaseModel):
    title: Optional[str] = Field(None)
    agent_id: Optional[str] = Field(None, description="Agent to use for this session")


class SessionCreateResponse(BaseModel):
    session_id: str
    title: str
    agent: Optional[Agent] = None
    message: str


class SessionListItem(BaseModel):
    session_id: str
    title: str
    agent: Optional[Agent] = None
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionListItem]
    total: int


class SessionInfoResponse(BaseModel):
    session_id: str
    title: str
    agent: Optional[Agent] = None
    messages: List[Message]
    total_messages: int
    created_at: datetime
    updated_at: datetime


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    agent_id: Optional[str] = None
