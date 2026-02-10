from typing import List, Dict, Optional

from openai import OpenAI
from langchain_core.documents import Document


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
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate answer using agent-specific system prompt
        """
        # Build RAG context
        context = self._build_context_from_docs(retrieved_docs)

        # Use agent's system prompt or default
        if not system_prompt:
            system_prompt = (
                "You are an AI assistant. "
                "Use the provided context to answer questions accurately. "
                "You have access to conversation history and retrieved document context."
            )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(conversation_history)

        # Add current query with RAG context
        user_content = f"Retrieved Context:\n{context}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_content})

        # Call OpenAI
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )

        return resp.choices[0].message.content
