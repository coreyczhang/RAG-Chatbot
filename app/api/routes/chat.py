from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.core.config import settings
from app.models.schemas import (
    ChatRequest, 
    ChatResponse, 
    SourceChunk, 
    SessionCreateResponse,
    SessionInfoResponse,
    Message
)

from app.clients.openai_client import get_openai_client
from app.clients.supabase_client import get_supabase_client
from app.clients.embeddings import get_embeddings

from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService
from app.services.session_service import session_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/session", response_model=SessionCreateResponse)
def create_session():
    """Create a new chat session with UUID"""
    session_id = session_service.create_session()
    return SessionCreateResponse(
        session_id=session_id,
        message="Session created successfully"
    )


@router.get("/session/{session_id}", response_model=SessionInfoResponse)
def get_session_info(session_id: str):
    """Get ALL messages in session by session UUID"""
    info = session_service.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return SessionInfoResponse(**info)


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and all its messages"""
    session_service.clear_session(session_id)
    return {"message": "Session deleted successfully"}


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    RAG Chat with Session Memory Flow:
    1. Get or create session by UUID
    2. Store user input with UUID
    3. Retrieve ALL previous messages from session
    4. Retrieve relevant RAG chunks from Supabase
    5. Send conversation history + RAG context to LLM
    6. Store assistant output with UUID
    7. Return response
    """
    try:
        # STEP 1: Get or create session
        if req.session_id:
            session = session_service.get_session(req.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or expired")
            session_id = req.session_id
        else:
            session_id = session_service.create_session()
            session = session_service.get_session(session_id)
        
        # STEP 2: Store user input with UUID
        user_message = Message(
            role="user",
            content=req.query,
            timestamp=datetime.now()
        )
        user_message_id = session_service.add_message(session_id, user_message)
        
        # STEP 3: Retrieve ALL previous messages from session (before current input)
        all_previous_messages = session_service.get_all_messages(session_id)
        # Exclude the message we just added
        conversation_history = all_previous_messages[:-1]
        # Convert to LLM format
        conversation_history_for_llm = [
            {"role": msg.role, "content": msg.content} 
            for msg in conversation_history
        ]

        # STEP 4: Retrieve relevant RAG chunks from Supabase
        supabase = get_supabase_client()
        embeddings = get_embeddings()
        retrieval = RetrievalService(supabase=supabase, embeddings=embeddings)

        retrieved_docs = retrieval.similarity_search(
            query=req.query,
            k=min(req.k, settings.max_k),
            filter=req.filter,
            match_threshold=req.match_threshold,
        )

        # Build sources for response
        sources = [
            SourceChunk(
                content_preview=(d.page_content or "")[:500],
                metadata=d.metadata or {},
            )
            for d in retrieved_docs
        ]
        
        # Convert sources to dict format for storage
        sources_dict = [
            {
                "content_preview": (d.page_content or "")[:500],
                "metadata": d.metadata or {}
            }
            for d in retrieved_docs
        ]

        # STEP 5: Send conversation history + RAG context to LLM
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
        
        # STEP 6: Store assistant output with UUID and RAG sources
        assistant_message = Message(
            role="assistant",
            content=answer_text,
            timestamp=datetime.now(),
            rag_sources=sources_dict
        )
        assistant_message_id = session_service.add_message(session_id, assistant_message)

        # STEP 7: Return response
        return ChatResponse(
            message=assistant_message,
            session_id=session_id,
            total_messages_in_session=session.get_message_count()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
