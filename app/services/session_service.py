from typing import List, Optional, Dict
from datetime import datetime
from supabase import Client
from app.models.schemas import Message


class SessionService:
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    def create_session(self, title: Optional[str] = None, agent_id: Optional[str] = None) -> str:
        """Create a new session with optional agent"""
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        data = {'title': title}
        if agent_id:
            data['agent_id'] = agent_id
        
        result = self.supabase.table('chat_sessions').insert(data).execute()
        return result.data[0]['session_id']
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session with agent info"""
        result = self.supabase.table('chat_sessions')\
            .select('*, agents(*)')\
            .eq('session_id', session_id)\
            .execute()
        return result.data[0] if result.data else None
    
    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List all sessions with agent info"""
        result = self.supabase.table('chat_sessions')\
            .select('session_id, title, agent_id, agents(*), created_at, updated_at')\
            .order('updated_at', desc=True)\
            .limit(limit)\
            .offset(offset)\
            .execute()
        return result.data
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session (title or agent)"""
        result = self.supabase.table('chat_sessions')\
            .update(kwargs)\
            .eq('session_id', session_id)\
            .execute()
        return len(result.data) > 0
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        result = self.supabase.table('chat_sessions')\
            .delete()\
            .eq('session_id', session_id)\
            .execute()
        return len(result.data) > 0
    
    def add_message(self, session_id: str, message: Message) -> str:
        """Add message to session"""
        result = self.supabase.table('chat_messages').insert({
            'message_id': message.message_id,
            'session_id': session_id,
            'role': message.role,
            'content': message.content,
            'rag_sources': message.rag_sources
        }).execute()
        return result.data[0]['message_id']
    
    def get_all_messages(self, session_id: str) -> List[Message]:
        """Get all messages for session"""
        result = self.supabase.table('chat_messages')\
            .select('*')\
            .eq('session_id', session_id)\
            .order('created_at', desc=False)\
            .execute()
        
        return [
            Message(
                message_id=msg['message_id'],
                role=msg['role'],
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg['created_at']),
                rag_sources=msg.get('rag_sources')
            )
            for msg in result.data
        ]
    
    def get_message_count(self, session_id: str) -> int:
        """Get message count"""
        result = self.supabase.table('chat_messages')\
            .select('message_id', count='exact')\
            .eq('session_id', session_id)\
            .execute()
        return result.count
