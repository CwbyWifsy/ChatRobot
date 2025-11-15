from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    db,
    utility,
)

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    content: str
    embedding: List[float]
    book_title: str
    chapter_title: str
    chunk_index: int
    source_path: str
    file_hash: str


class MilvusVectorStore:
    """Wrapper around Milvus collection management and operations."""

    def __init__(self, collection_name: str | None = None) -> None:
        self.collection_name = collection_name or settings.milvus_collection
        self._connect()
        self._ensure_database()
        self.collection = self._ensure_collection()

    def _connect(self) -> None:
        logger.info("Connecting to Milvus at %s", settings.milvus_uri)
        connections.connect(alias="default", uri=settings.milvus_uri, token=settings.milvus_token)

    def _ensure_database(self) -> None:
        try:
            db.using_database(settings.milvus_database)
        except MilvusException:
            logger.info("Creating Milvus database %s", settings.milvus_database)
            db.create_database(settings.milvus_database)
            db.using_database(settings.milvus_database)

    def _ensure_collection(self) -> Collection:
        existing_collections = set(utility.list_collections())
        if self.collection_name in existing_collections:
            logger.info("Using existing collection %s", self.collection_name)
            collection = Collection(self.collection_name)
        else:
            logger.info("Creating collection %s", self.collection_name)
            schema = CollectionSchema(
                fields=[
                    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema("book_title", DataType.VARCHAR, max_length=256),
                    FieldSchema("chapter_title", DataType.VARCHAR, max_length=2048),
                    FieldSchema("chunk_index", DataType.INT64),
                    FieldSchema("source_path", DataType.VARCHAR, max_length=256),
                    FieldSchema("file_hash", DataType.VARCHAR, max_length=128),
                    FieldSchema("content", DataType.VARCHAR, max_length=8192),
                    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
                ]
            )
            collection = Collection(self.collection_name, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": settings.milvus_metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                },
            )
        collection.load()
        return collection

    def list_collections(self) -> List[str]:
        return sorted(utility.list_collections())

    def use_collection(self, collection_name: str) -> None:
        if collection_name == self.collection_name:
            return
        self.collection_name = collection_name
        self.collection = self._ensure_collection()

    def list_books(self, collection_name: str | None = None) -> List[str]:
        collection = Collection(collection_name or self.collection_name)
        results = collection.query(
            expr="book_title != ''",
            output_fields=["book_title"],
            consistency_level=settings.milvus_consistency_level,
            limit=16384,
        )
        return sorted({row["book_title"] for row in results})

    def has_file(self, file_hash: str, collection_name: str | None = None) -> bool:
        collection = Collection(collection_name or self.collection_name)
        safe_hash = file_hash.replace("\"", "\\\"")
        try:
            results = collection.query(
                expr=f"file_hash == \"{safe_hash}\"",
                output_fields=["file_hash"],
                consistency_level=settings.milvus_consistency_level,
            )
        except MilvusException:
            return False
        return len(results) > 0

    def insert_records(self, records, collection_name=None):
        if not records:
            return

        collection = Collection(collection_name or self.collection_name)
        rows = []
        for r in records:
            rows.append({
                "book_title": r.book_title,
                "chapter_title": r.chapter_title,
                "chunk_index": r.chunk_index,
                "source_path": r.source_path,
                "file_hash": r.file_hash,
                "content": r.content,
                "embedding": r.embedding,
            })

        collection.insert(rows, timeout=120)
        collection.flush()

    def search(self, embedding: List[float], top_k: int = 4, collection_name: str | None = None):
        collection = Collection(collection_name or self.collection_name)
        search_params = {"metric_type": settings.milvus_metric_type, "params": {"nprobe": 32}}
        try:
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["book_title", "chapter_title", "chunk_index", "content", "source_path"],
            )
        except MilvusException:
            return []
        if not results:
            return []
        return results[0]


__all__ = ["MilvusVectorStore", "VectorRecord"]
