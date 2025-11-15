from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Iterable, List

from app.services.embedding import EmbeddingService
from app.services.hashing import NovelHasher
from app.logger import configure_logging
from app.services.text_splitter import ChapterTextSplitter, Chunk
from app.services.vector_store import MilvusVectorStore, VectorRecord

logger = logging.getLogger(__name__)


def iter_text_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("**/*.txt")):
        if path.is_file():
            yield path


async def load_file(path: Path) -> str:
    return await asyncio.to_thread(path.read_text, "utf-8")


async def process_file(
    path: Path,
    embedding_service: EmbeddingService,
    vector_store: MilvusVectorStore,
    splitter: ChapterTextSplitter,
    hasher: NovelHasher,
    collection_name: str,
    force: bool,
) -> None:
    logger.info("Reading %s", path)
    content = await load_file(path)

    book_title = input(f"为文件 {path.name} 输入小说名称（直接回车使用文件名）: ") or path.stem

    file_hash = hasher.hash_file(path, extra_values=[book_title])
    if not force and vector_store.has_file(file_hash, collection_name):
        confirm = input(f"检测到 {path} 已上传过，是否重新上传? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            logger.info("跳过 %s", path)
            return

    chunks: List[Chunk] = list(splitter.split(content, book_title=book_title, source_path=path))
    texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.embed_documents(texts)

    records = [
        VectorRecord(
            content=chunk.content,
            embedding=embedding,
            book_title=chunk.book_title,
            chapter_title=chunk.chapter_title,
            chunk_index=chunk.chunk_index,
            source_path=str(chunk.source_path),
            file_hash=file_hash,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    vector_store.insert_records(records, collection_name)
    logger.info("已向集合 %s 写入 %d 个分片", collection_name, len(records))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Upload UTF-8 novels into Milvus vector store")
    parser.add_argument("directory", type=Path, help="Directory containing .txt novel files")
    parser.add_argument("--collection", type=str, default=None, help="Target collection name")
    parser.add_argument("--force", action="store_true", help="Upload even if file hash already exists")
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.exists():
        raise SystemExit(f"目录 {directory} 不存在")

    configure_logging()

    embedding_service = EmbeddingService()
    vector_store = MilvusVectorStore(collection_name=args.collection)

    target_collection = args.collection or vector_store.collection_name
    if not args.collection:
        collections = vector_store.list_collections()
        if collections:
            logger.info("当前共有 %d 个集合：", len(collections))
            for name in collections:
                try:
                    novels = vector_store.list_books(name)
                except Exception as exc:  # pragma: no cover - 运行环境可能缺少权限
                    logger.warning("读取集合 %s 的小说列表失败：%s", name, exc)
                    novels = []
                if novels:
                    logger.info("  - %s (%d 本)：%s", name, len(novels), ", ".join(novels))
                else:
                    logger.info("  - %s (暂无小说)", name)
        else:
            logger.info("当前没有集合，将创建默认集合 %s", target_collection)

        chosen = input(f"请输入目标集合名称（直接回车使用 {target_collection}）: ").strip()
        if chosen:
            target_collection = chosen

    vector_store.use_collection(target_collection)
    logger.info("上传目标集合：%s", target_collection)

    splitter = ChapterTextSplitter()
    hasher = NovelHasher()

    for file_path in iter_text_files(directory):
        await process_file(
            file_path,
            embedding_service=embedding_service,
            vector_store=vector_store,
            splitter=splitter,
            hasher=hasher,
            collection_name=vector_store.collection_name,
            force=args.force,
        )


if __name__ == "__main__":
    asyncio.run(main())
