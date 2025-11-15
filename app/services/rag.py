from __future__ import annotations

import logging
from typing import Dict, List

from openai import OpenAI

from ..config import settings
from .embedding import EmbeddingService
from .vector_store import MilvusVectorStore, VectorRecord

logger = logging.getLogger(__name__)


class RAGService:
    """High level retrieval augmented generation pipeline."""

    def __init__(self, vector_store: MilvusVectorStore | None = None, embedding_service: EmbeddingService | None = None) -> None:
        self.vector_store = vector_store or MilvusVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    def index_records(self, records: List[VectorRecord], collection_name: str | None = None) -> None:
        self.vector_store.insert_records(records, collection_name)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        embedding = self.embedding_service.embed_documents([query])[0]
        results = self.vector_store.search(embedding, top_k=top_k)
        documents: List[Dict[str, str]] = []
        for hit in results:
            documents.append(
                {
                    "content": hit.entity.get("content"),
                    "book_title": hit.entity.get("book_title"),
                    "chapter_title": hit.entity.get("chapter_title"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "source_path": hit.entity.get("source_path"),
                    "score": hit.distance,
                }
            )
        return documents

    def generate(self, query: str, context_documents: List[Dict[str, str]], history: List[Dict[str, str]]) -> str:
        context_text = "\n\n".join(
            f"【{doc['book_title']}·{doc['chapter_title']}·chunk {doc['chunk_index']}】\n{doc['content']}"
            for doc in context_documents
        )
        messages = []
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        system_prompt = (
            "你是一个小说问答助手。你将基于提供的参考内容回答用户的问题，"
            "回答时引用相关的章节和来源，保持语言简洁准确。"
        )
        messages = (
            [{"role": "system", "content": system_prompt}] + messages + [
                {
                    "role": "user",
                    "content": (
                        "参考内容：\n" + context_text + "\n\n" + "用户问题：" + query if context_text else query
                    ),
                }
            ]
        )
        response = self.client.responses.create(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            input=messages,
        )

        text_fragments: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text_fragments.append(getattr(content, "text", ""))

        generated = "".join(text_fragments).strip()
        if not generated:
            logger.warning("Empty response from LLM, returning fallback message")
            return "抱歉，我暂时无法生成回答。"
        return generated


__all__ = ["RAGService"]
