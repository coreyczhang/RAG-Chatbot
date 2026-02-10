from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.models.schemas import (
    Agent,
    AgentCreateRequest,
    AgentUpdateRequest
)
from app.clients.supabase_client import get_supabase_client
from app.services.agent_service import AgentService

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=list[Agent])
def list_agents():
    """List all available agents"""
    supabase = get_supabase_client()
    agent_service = AgentService(supabase)
    
    agents = agent_service.list_agents()
    return [
        Agent(
            agent_id=a['agent_id'],
            name=a['name'],
            description=a.get('description'),
            system_prompt=a['system_prompt'],
            model=a['model'],
            temperature=a['temperature'],
            max_output_tokens=a['max_output_tokens'],
            avatar_emoji=a.get('avatar_emoji'),
            created_at=datetime.fromisoformat(a['created_at'])
        )
        for a in agents
    ]


@router.get("/{agent_id}", response_model=Agent)
def get_agent(agent_id: str):
    """Get specific agent"""
    supabase = get_supabase_client()
    agent_service = AgentService(supabase)
    
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return Agent(
        agent_id=agent['agent_id'],
        name=agent['name'],
        description=agent.get('description'),
        system_prompt=agent['system_prompt'],
        model=agent['model'],
        temperature=agent['temperature'],
        max_output_tokens=agent['max_output_tokens'],
        avatar_emoji=agent.get('avatar_emoji'),
        created_at=datetime.fromisoformat(agent['created_at'])
    )


@router.post("", response_model=Agent)
def create_agent(req: AgentCreateRequest):
    """Create new agent"""
    supabase = get_supabase_client()
    agent_service = AgentService(supabase)
    
    agent_id = agent_service.create_agent(
        name=req.name,
        description=req.description,
        system_prompt=req.system_prompt,
        model=req.model,
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
        avatar_emoji=req.avatar_emoji
    )
    
    return get_agent(agent_id)


@router.patch("/{agent_id}", response_model=Agent)
def update_agent(agent_id: str, req: AgentUpdateRequest):
    """Update agent configuration"""
    supabase = get_supabase_client()
    agent_service = AgentService(supabase)
    
    # Build update dict
    updates = {}
    for field, value in req.model_dump(exclude_unset=True).items():
        if value is not None:
            updates[field] = value
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    success = agent_service.update_agent(agent_id, **updates)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return get_agent(agent_id)


@router.delete("/{agent_id}")
def delete_agent(agent_id: str):
    """Delete an agent"""
    supabase = get_supabase_client()
    agent_service = AgentService(supabase)
    
    success = agent_service.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {"message": "Agent deleted successfully"}
