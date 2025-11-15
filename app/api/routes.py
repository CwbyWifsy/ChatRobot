from __future__ import annotations
import os
import logging
from typing import List

from fastapi import APIRouter

from ..models.api import ChatRequest, ChatResponse, CollectionList, DocumentCitation, ModelList, ModelInfo
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


@router.get("/models", response_model=ModelList)
async def list_models() -> ModelList:
    """
    读取 .env 中的 LLM_MODELS / LLM_DEFAULT_MODEL，
    返回当前可用模型列表和默认模型。
    """
    raw = os.getenv("LLM_MODELS", "")  # 例如 "qwen2.5-72b-instruct,gpt-4.1-mini"
    default_model = (
        os.getenv("LLM_DEFAULT_MODEL")
        or os.getenv("LLM_MODEL_NAME", "")  # 兼容你原来单模型的配置
    )

    # 拆分 + 去空格 / 空字符串
    names = [m.strip() for m in raw.split(",") if m.strip()]

    # 如果没配置 DEFAULT，但有列表，就默认第一个
    if not default_model and names:
        default_model = names[0]

    return ModelList(
        models=[ModelInfo(name=n) for n in names],
        active_model=default_model,
    )

__all__ = ["router"]
