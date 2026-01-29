from fastapi import APIRouter, HTTPException, Query
from datetime import datetime

from app.core.config import settings
from app.models.schemas import (
    ChatRequest, 
    ChatResponse, 
    SourceChunk, 
    SessionCreateRequest,
    SessionCreateResponse,
    SessionListResponse,
    SessionListItem,
    SessionInfoResponse,
    SessionUpdateRequest,
    Message
)

from app.clients.openai_client import get_openai_client
from app.clients.supabase_client import get_supabase_client
from app.clients.embeddings import get_embeddings

from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService
from app.services.session_service import SessionService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/session", response_model=SessionCreateResponse)
def create_session(req: SessionCreateRequest):
    """Create a new permanent chat session"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    
    session_id = session_service.create_session(title=req.title)
    session = session_service.get_session(session_id)
    
    return SessionCreateResponse(
        session_id=session_id,
        title=session['title'],
        message="Session created successfully"
    )


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all chat sessions (like ChatGPT sidebar)"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    
    sessions = session_service.list_sessions(limit=limit, offset=offset)
    
    # Add message count to each session
    sessions_with_count = []
    for session in sessions:
        msg_count = session_service.get_message_count(session['session_id'])
        sessions_with_count.append(SessionListItem(
            session_id=session['session_id'],
            title=session['title'],
            created_at=datetime.fromisoformat(session['created_at']),
            updated_at=datetime.fromisoformat(session['updated_at']),
            message_count=msg_count
        ))
    
    return SessionListResponse(
        sessions=sessions_with_count,
        total=len(sessions_with_count)
    )


@router.get("/session/{session_id}", response_model=SessionInfoResponse)
def get_session_info(session_id: str):
    """Get full session with all messages"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session_service.get_all_messages(session_id)
    
    return SessionInfoResponse(
        session_id=session['session_id'],
        title=session['title'],
        messages=messages,
        total_messages=len(messages),
        created_at=datetime.fromisoformat(session['created_at']),
        updated_at=datetime.fromisoformat(session['updated_at'])
    )


@router.patch("/session/{session_id}", response_model=dict)
def update_session(session_id: str, req: SessionUpdateRequest):
    """Update session title"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    
    success = session_service.update_session_title(session_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session updated successfully"}


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session permanently"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    
    success = session_service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat with RAG and permanent session storage
    """
    try:
        supabase = get_supabase_client()
        session_service = SessionService(supabase)
        
        # Get or create session
        if req.session_id:
            session = session_service.get_session(req.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            session_id = req.session_id
        else:
            # Auto-create session with timestamp
            session_id = session_service.create_session()
        
        # Store user message
        user_message = Message(
            role="user",
            content=req.query,
            timestamp=datetime.now()
        )
        session_service.add_message(session_id, user_message)
        
        # Get conversation history
        all_messages = session_service.get_all_messages(session_id)
        conversation_history = all_messages[:-1]  # Exclude current message
        conversation_history_for_llm = [
            {"role": msg.role, "content": msg.content} 
            for msg in conversation_history
        ]

        # Retrieve RAG chunks
        embeddings = get_embeddings()
        retrieval = RetrievalService(supabase=supabase, embeddings=embeddings)

        retrieved_docs = retrieval.similarity_search(
            query=req.query,
            k=min(req.k, settings.max_k),
            filter=req.filter,
            match_threshold=req.match_threshold,
        )

        sources = [
            SourceChunk(
                content_preview=(d.page_content or "")[:500],
                metadata=d.metadata or {},
            )
            for d in retrieved_docs
        ]
        
        sources_dict = [
            {
                "content_preview": (d.page_content or "")[:500],
                "metadata": d.metadata or {}
            }
            for d in retrieved_docs
        ]

        # Generate answer
        openai_client = get_openai_client()
        chat_service = ChatService(openai_client=openai_client)

        model = req.model or settings.openai_model
        max_output_tokens = req.max_output_tokens or settings.default_max_output_tokens
        temperature = req.temperature if req.temperature is not None else settings.default_temperature

        answer_text = chat_service.answer(
            query=req.query,
            retrieved_docs=retrieved_docs,
            conversation_history=conversation_history_for_llm,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        
        # Store assistant message
        assistant_message = Message(
            role="assistant",
            content=answer_text,
            timestamp=datetime.now(),
            rag_sources=sources_dict
        )
        session_service.add_message(session_id, assistant_message)

        total_messages = session_service.get_message_count(session_id)

        return ChatResponse(
            message=assistant_message,
            session_id=session_id,
            total_messages_in_session=total_messages
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
