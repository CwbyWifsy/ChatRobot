from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter

from ..models.api import ChatRequest, ChatResponse, CollectionList, DocumentCitation
from ..services.chat_history import ChatSessionManager
from ..services.rag import RAGService

logger = logging.getLogger(__name__)


router = APIRouter()
rag_service = RAGService()
chat_sessions = ChatSessionManager()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    requested_collection = payload.collection
    active_collection = (
        requested_collection
        or chat_sessions.get_collection(payload.session_id)
        or rag_service.vector_store.collection_name
    )
    chat_sessions.set_collection(payload.session_id, active_collection)

    history = chat_sessions.get_history(payload.session_id)
    documents = rag_service.retrieve(
        payload.query,
        top_k=payload.top_k,
        collection_name=active_collection,
    )
    answer = rag_service.generate(payload.query, documents, history)
    chat_sessions.append(payload.session_id, payload.query, answer)

    citations: List[DocumentCitation] = [
        DocumentCitation(
            book_title=doc["book_title"],
            chapter_title=doc["chapter_title"],
            chunk_index=doc["chunk_index"],
            source_path=doc["source_path"],
            score=float(doc["score"]),
        )
        for doc in documents
    ]

    logger.info(
        "Session %s | Collection %s | User: %s | Answer length: %s",
        payload.session_id,
        active_collection,
        payload.query,
        len(answer),
    )

    return ChatResponse(answer=answer, citations=citations)


@router.get("/collections", response_model=CollectionList)
async def list_collections() -> CollectionList:
    collections = []
    for name in rag_service.vector_store.list_collections():
        try:
            novels = rag_service.vector_store.list_books(name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to read books from collection %s: %s", name, exc)
            novels = []
        collections.append({"name": name, "novels": novels})

    return CollectionList(collections=collections, active_collection=rag_service.vector_store.collection_name)


__all__ = ["router"]
