from pydantic import BaseModel
from typing import List, Dict
from uuid import UUID


class ChatCreate(BaseModel):
    chat_id: str


class QueryRequest(BaseModel):
    chat_id: UUID
    query: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: List[Dict]
    evaluation: dict
