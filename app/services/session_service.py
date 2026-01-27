from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
from app.models.schemas import Message


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: Dict[str, Message] = {}  # message_id -> Message
        self.message_order: List[str] = []  # Ordered list of message_ids
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    def add_message(self, message: Message) -> str:
        """Add a Message object and return its UUID"""
        self.messages[message.message_id] = message
        self.message_order.append(message.message_id)
        self.last_accessed = datetime.now()
        return message.message_id
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """Get a specific message by UUID"""
        return self.messages.get(message_id)
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages in chronological order"""
        return [self.messages[mid] for mid in self.message_order if mid in self.messages]
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get all messages in OpenAI format (role + content only)"""
        messages = self.get_all_messages()
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def get_message_count(self) -> int:
        """Get total number of messages in session"""
        return len(self.message_order)


class SessionService:
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self) -> str:
        """Create a new session and return session_id"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID, return None if expired or doesn't exist"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session expired
        if datetime.now() - session.last_accessed > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        return session
    
    def add_message(self, session_id: str, message: Message) -> str:
        """Add a Message object to session, return message UUID"""
        session = self.get_session(session_id)
        if session:
            return session.add_message(message)
        return None
    
    def get_message(self, session_id: str, message_id: str) -> Optional[Message]:
        """Get a specific message by UUID"""
        session = self.get_session(session_id)
        if session:
            return session.get_message(message_id)
        return None
    
    def get_all_messages(self, session_id: str) -> List[Message]:
        """Get all messages for a session"""
        session = self.get_session(session_id)
        if session:
            return session.get_all_messages()
        return []
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get full session information"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "messages": session.get_all_messages(),
            "total_messages": session.get_message_count(),
            "created_at": session.created_at,
            "last_accessed": session.last_accessed
        }
    
    def clear_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_accessed > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]


# Global session service instance
session_service = SessionService(session_timeout_minutes=30)
