from typing import List, Dict

from openai import OpenAI
from langchain_core.documents import Document

from app.core.config import settings


class ChatService:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client

    def _build_context_from_docs(self, docs: List[Document]) -> str:
        """Build context string from retrieved documents"""
        return "\n\n---\n\n".join(d.page_content for d in docs if d.page_content)

    def answer(
        self,
        query: str,
        retrieved_docs: List[Document],
        conversation_history: List[Dict[str, str]],
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        """
        Generate answer using:
        1. Full conversation history from session
        2. RAG context from retrieved documents
        """
        # Build RAG context
        context = self._build_context_from_docs(retrieved_docs)

        # System message
        system_content = (
            "You are an AI assistant. "
            "Use the provided context to answer questions accurately. "
            "You have access to conversation history and retrieved document context."
        )

        # Build messages array
        messages = [{"role": "system", "content": system_content}]
        
        # Add ALL previous conversation history
        messages.extend(conversation_history)

        # Add current query WITH RAG context
        user_content_with_context = (
            f"Retrieved Context:\n{context}\n\n"
            f"Question: {query}"
        )
        messages.append({"role": "user", "content": user_content_with_context})

        # Call OpenAI
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )

        return resp.choices[0].message.content
