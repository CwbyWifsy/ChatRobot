from __future__ import annotations
from scripts.utils import make_collection_name_from_path
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List
from tqdm import tqdm

from app.services.embedding import EmbeddingService
from app.services.hashing import NovelHasher
from app.logger import configure_logging
from app.services.text_splitter import ChapterTextSplitter, Chunk
from app.services.vector_store import MilvusVectorStore, VectorRecord

logger = logging.getLogger(__name__)


def iter_text_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("**/*.txt")):
        if path.is_file():
            # convert_to_utf8(path)
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
        extra_collection_name: str | None,
        force: bool,
) -> None:
    logger.info("Reading %s", path)

    extra_store = None
    if extra_collection_name:
        logger.info("为文件 %s 使用独立集合 %s", path.name, extra_collection_name)
        print(f"本书独立集合名：{path.name} -> {extra_collection_name}")
        extra_store = MilvusVectorStore(collection_name=extra_collection_name)

    content = await load_file(path)

    book_title = path.stem
    logger.info("当前文件书名自动设置为：%s", book_title)

    file_hash = hasher.hash_file(path, extra_values=[book_title])
    if not force and vector_store.has_file(file_hash, collection_name):
        logger.info("检测到 %s 已上传过，未指定 --force，自动跳过", path)
        return

    # 1⃣️ 切分文本 —— 带进度条
    logger.info("正在切分章节…")
    raw_chunks = list(splitter.split(content, book_title=book_title, source_path=path))
    chunks = []

    for c in tqdm(raw_chunks, desc="📑 分片处理中"):
        chunks.append(c)

    # 2⃣️ 分批 embedding + 上传，三个真实进度条
    BATCH_SIZE = 1000  # 每批处理多少条，可以按机器情况改

    total = len(chunks)
    logger.info("正在分批生成向量并写入 Milvus...")

    MAX_BOOK_TITLE_LEN = 256
    MAX_CHAPTER_TITLE_LEN = 512
    MAX_SOURCE_PATH_LEN = 256
    MAX_FILE_HASH_LEN = 128
    MAX_CONTENT_LEN = 8192

    # ① 总体进度条：整本小说的分片总进度
    with tqdm(total=total, desc="📦 总体进度", unit="chunk") as pbar_total:
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_chunks = chunks[start:end]
            batch_texts = [c.content for c in batch_chunks]

            # ② 当前批次 embedding 进度条（嵌入 n 条）
            batch_embeddings: List[List[float]] = []
            for text in tqdm(batch_texts, desc="🧠 本批 embedding", leave=False):
                vec = embedding_service.embed_documents([text])[0]
                batch_embeddings.append(vec)

            # 组装当前批次记录
            batch_records: list[VectorRecord] = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                chapter_title = chunk.chapter_title
                if len(chapter_title) > MAX_CHAPTER_TITLE_LEN:
                    logger.warning("跳过一条记录：chapter_title_len=%d, title=%r", len(chapter_title),
                                   chapter_title[:80])
                    continue
                batch_records.append(
                    VectorRecord(
                        content=chunk.content,
                        embedding=embedding,
                        book_title=chunk.book_title,
                        chapter_title=chunk.chapter_title,
                        chunk_index=chunk.chunk_index,
                        source_path=str(chunk.source_path),
                        file_hash=file_hash,
                    )
                )

            vector_store.insert_records(batch_records, collection_name)
            # 如果用户启用了 single_collection，再写入新集合
            if extra_store is not None:
                extra_store.insert_records(batch_records)
            # 更新总体进度
            pbar_total.update(len(batch_chunks))

    logger.info("已向集合 %s 写入 %d 个分片", collection_name, total)
    if extra_collection_name:
        logger.info("已向独立集合 %s 额外写入 %d 个分片", extra_collection_name, total)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Upload UTF-8 novels into Milvus vector store")
    parser.add_argument("directory", type=Path, help="Directory containing .txt novel files")
    parser.add_argument("--collection", type=str, default=None, help="Target collection name")
    parser.add_argument("--force", action="store_true", help="Upload even if file hash already exists")
    parser.add_argument("--single_collection", action="store_true",
                        help="为当前上传额外创建并写入一个新集合")
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
    else:
        logger.info(f"公共集合:{args.collection}")
    vector_store.use_collection(target_collection)
    logger.info("上传目标集合：%s", target_collection)

    splitter = ChapterTextSplitter()
    hasher = NovelHasher()

    # 先把所有要处理的 txt 文件拿出来
    all_files = list(iter_text_files(directory))
    if not all_files:
        logger.info("目录 %s 下没有找到 txt 文件", directory)
        return

    # 如果需要每本小说单独建 collection，先列出全部名字让你确认
    per_file_extra: dict[Path, str] = {}
    if args.single_collection:
        print("你启用了 --single_collection，将为每本小说创建独立集合。")
        print("即将使用如下映射：")

        for f in all_files:
            cname = make_collection_name_from_path(f)
            per_file_extra[f] = cname
            print(f"  {f.name}  ->  {cname}")

        confirm = input("确认以上集合名映射无误后继续？[y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            raise SystemExit("已取消上传。")
    # sys.exit()
    for file_path in iter_text_files(directory):
        extra_name = per_file_extra.get(file_path) if args.single_collection else None
        await process_file(
            file_path,
            embedding_service=embedding_service,
            vector_store=vector_store,
            splitter=splitter,
            hasher=hasher,
            collection_name=vector_store.collection_name,
            extra_collection_name=extra_name,
            force=args.force,
        )


if __name__ == "__main__":
    asyncio.run(main())
