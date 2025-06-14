"""
Response models for the Vivum RAG Backend
"""
from pydantic import BaseModel

class TopicResponse(BaseModel):
    topic_id: str
    message: str
    status: str

class ChatResponse(BaseModel):
    response: str
    conversation_id: str