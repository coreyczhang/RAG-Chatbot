from typing import List, Optional, Dict
from supabase import Client


class AgentService:
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent by ID"""
        result = self.supabase.table('agents').select('*').eq('agent_id', agent_id).execute()
        return result.data[0] if result.data else None
    
    def list_agents(self) -> List[Dict]:
        """List all available agents"""
        result = self.supabase.table('agents').select('*').order('name').execute()
        return result.data
    
    def create_agent(self, name: str, system_prompt: str, description: Optional[str] = None,
                    model: str = 'gpt-4o-mini', temperature: float = 0.4, 
                    max_output_tokens: int = 400, avatar_emoji: Optional[str] = None) -> str:
        """Create a new agent"""
        result = self.supabase.table('agents').insert({
            'name': name,
            'description': description,
            'system_prompt': system_prompt,
            'model': model,
            'temperature': temperature,
            'max_output_tokens': max_output_tokens,
            'avatar_emoji': avatar_emoji
        }).execute()
        return result.data[0]['agent_id']
    
    def update_agent(self, agent_id: str, **kwargs) -> bool:
        """Update agent settings"""
        result = self.supabase.table('agents').update(kwargs).eq('agent_id', agent_id).execute()
        return len(result.data) > 0
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        result = self.supabase.table('agents').delete().eq('agent_id', agent_id).execute()
        return len(result.data) > 0
