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
    Message,
    Agent
)

from app.clients.openai_client import get_openai_client
from app.clients.supabase_client import get_supabase_client
from app.clients.embeddings import get_embeddings

from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService
from app.services.session_service import SessionService
from app.services.agent_service import AgentService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/session", response_model=SessionCreateResponse)
def create_session(req: SessionCreateRequest):
    """Create a new permanent chat session"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    agent_service = AgentService(supabase)
    
    session_id = session_service.create_session(title=req.title, agent_id=req.agent_id)
    session = session_service.get_session(session_id)
    
    agent_obj = None
    if req.agent_id:
        agent_data = agent_service.get_agent(req.agent_id)
        if agent_data:
            agent_obj = Agent(
                agent_id=agent_data['agent_id'],
                name=agent_data['name'],
                description=agent_data.get('description'),
                system_prompt=agent_data['system_prompt'],
                model=agent_data['model'],
                temperature=agent_data['temperature'],
                max_output_tokens=agent_data['max_output_tokens'],
                avatar_emoji=agent_data.get('avatar_emoji'),
                created_at=datetime.fromisoformat(agent_data['created_at'])
            )
    
    return SessionCreateResponse(
        session_id=session_id,
        title=session['title'],
        agent=agent_obj,
        message="Session created successfully"
    )


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all chat sessions"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    
    sessions = session_service.list_sessions(limit=limit, offset=offset)
    
    sessions_with_count = []
    for session in sessions:
        msg_count = session_service.get_message_count(session['session_id'])
        
        agent_obj = None
        if session.get('agents'):
            agent_data = session['agents']
            agent_obj = Agent(
                agent_id=agent_data['agent_id'],
                name=agent_data['name'],
                description=agent_data.get('description'),
                system_prompt=agent_data['system_prompt'],
                model=agent_data['model'],
                temperature=agent_data['temperature'],
                max_output_tokens=agent_data['max_output_tokens'],
                avatar_emoji=agent_data.get('avatar_emoji'),
                created_at=datetime.fromisoformat(agent_data['created_at'])
            )
        
        sessions_with_count.append(SessionListItem(
            session_id=session['session_id'],
            title=session['title'],
            agent=agent_obj,
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
    
    agent_obj = None
    if session.get('agents'):
        agent_data = session['agents']
        agent_obj = Agent(
            agent_id=agent_data['agent_id'],
            name=agent_data['name'],
            description=agent_data.get('description'),
            system_prompt=agent_data['system_prompt'],
            model=agent_data['model'],
            temperature=agent_data['temperature'],
            max_output_tokens=agent_data['max_output_tokens'],
            avatar_emoji=agent_data.get('avatar_emoji'),
            created_at=datetime.fromisoformat(agent_data['created_at'])
        )
    
    return SessionInfoResponse(
        session_id=session['session_id'],
        title=session['title'],
        agent=agent_obj,
        messages=messages,
        total_messages=len(messages),
        created_at=datetime.fromisoformat(session['created_at']),
        updated_at=datetime.fromisoformat(session['updated_at'])
    )


@router.patch("/session/{session_id}", response_model=dict)
def update_session(session_id: str, req: SessionUpdateRequest):
    """Update session title or agent"""
    supabase = get_supabase_client()
    session_service = SessionService(supabase)
    
    updates = {}
    if req.title:
        updates['title'] = req.title
    if req.agent_id:
        updates['agent_id'] = req.agent_id
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    success = session_service.update_session(session_id, **updates)
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
    """Chat with RAG using selected agent"""
    try:
        supabase = get_supabase_client()
        session_service = SessionService(supabase)
        agent_service = AgentService(supabase)
        
        # Get or create session
        if req.session_id:
            session = session_service.get_session(req.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            session_id = req.session_id
            
            # Get agent from session or request
            if req.agent_id:
                agent_id = req.agent_id
            elif session.get('agent_id'):
                agent_id = session['agent_id']
            else:
                agent_id = None
        else:
            # Create new session with agent
            session_id = session_service.create_session(agent_id=req.agent_id)
            session = session_service.get_session(session_id)
            agent_id = req.agent_id
        
        # Get agent configuration
        if agent_id:
            agent_data = agent_service.get_agent(agent_id)
            if not agent_data:
                raise HTTPException(status_code=404, detail="Agent not found")
        else:
            # Default agent data
            agent_data = {
                'agent_id': 'default',
                'name': 'Default Assistant',
                'description': None,
                'system_prompt': 'You are a helpful AI assistant.',
                'model': 'gpt-4o-mini',
                'temperature': 0.4,
                'max_output_tokens': 400,
                'avatar_emoji': '🤖',
                'created_at': datetime.now().isoformat()
            }
        
        # Store user message
        user_message = Message(role="user", content=req.query, timestamp=datetime.now())
        session_service.add_message(session_id, user_message)
        
        # Get conversation history
        all_messages = session_service.get_all_messages(session_id)
        conversation_history_for_llm = [
            {"role": msg.role, "content": msg.content} 
            for msg in all_messages[:-1]
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

        sources_dict = [
            {"content_preview": (d.page_content or "")[:500], "metadata": d.metadata or {}}
            for d in retrieved_docs
        ]

        # Generate answer with agent configuration
        openai_client = get_openai_client()
        chat_service = ChatService(openai_client=openai_client)

        # Use request overrides or agent defaults
        model = req.model or agent_data['model']
        temperature = req.temperature if req.temperature is not None else agent_data['temperature']
        max_tokens = req.max_output_tokens or agent_data['max_output_tokens']

        answer_text = chat_service.answer(
            query=req.query,
            retrieved_docs=retrieved_docs,
            conversation_history=conversation_history_for_llm,
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_prompt=agent_data['system_prompt'],
        )
        
        # Store assistant message
        assistant_message = Message(
            role="assistant",
            content=answer_text,
            timestamp=datetime.now(),
            rag_sources=sources_dict
        )
        session_service.add_message(session_id, assistant_message)

        # Build Agent response object
        agent_obj = Agent(
            agent_id=agent_data['agent_id'],
            name=agent_data['name'],
            description=agent_data.get('description'),
            system_prompt=agent_data['system_prompt'],
            model=agent_data['model'],
            temperature=agent_data['temperature'],
            max_output_tokens=agent_data['max_output_tokens'],
            avatar_emoji=agent_data.get('avatar_emoji'),
            created_at=datetime.fromisoformat(agent_data['created_at']) if isinstance(agent_data['created_at'], str) else agent_data['created_at']
        )

        return ChatResponse(
            message=assistant_message,
            session_id=session_id,
            agent=agent_obj,
            total_messages_in_session=session_service.get_message_count(session_id)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
