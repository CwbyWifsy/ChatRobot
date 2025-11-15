from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class NovelUploadResult(BaseModel):
    book_title: str
    file_path: str
    file_hash: str
    chunks_indexed: int
    skipped: bool = False


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the conversation session")
    query: str = Field(..., description="User question")
    top_k: int = Field(4, ge=1, le=10, description="Number of documents to retrieve")


class DocumentCitation(BaseModel):
    book_title: str
    chapter_title: str
    chunk_index: int
    source_path: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[DocumentCitation]


class CollectionInfo(BaseModel):
    name: str
    novels: List[str]


class CollectionList(BaseModel):
    collections: List[CollectionInfo]
    active_collection: str


class UploadConfirmation(BaseModel):
    file_path: str
    book_title: str
    confirm: bool
    collection: Optional[str] = None


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "DocumentCitation",
    "NovelUploadResult",
    "CollectionInfo",
    "CollectionList",
    "UploadConfirmation",
]
