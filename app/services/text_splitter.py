from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..config import settings


CHAPTER_PATTERN = re.compile(
    r"(?P<title>第?[\d一二三四五六七八九十百千]+[章回节卷篇部分章节：:．\.\s].*?)\n",
    flags=re.IGNORECASE,
)


@dataclass
class Chunk:
    book_title: str
    chapter_title: str
    chunk_index: int
    content: str
    source_path: Path


class ChapterTextSplitter:
    """Split novel text into chapter-aware overlapping chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def _split_chapter(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def split(self, content: str, *, book_title: str, source_path: Path) -> Iterable[Chunk]:
        matches = list(CHAPTER_PATTERN.finditer(content))
        if not matches:
            # Fallback to naive chunking using placeholder chapter name
            for idx, chunk in enumerate(self._split_chapter(content)):
                yield Chunk(book_title, "章节未知", idx, chunk.strip(), source_path)
            return

        boundaries = [match.start() for match in matches] + [len(content)]
        titles = [match.group("title").strip() for match in matches]

        # Handle prologue text before first chapter heading
        first_start = boundaries[0]
        if first_start > 0:
            preface = content[:first_start].strip()
            if preface:
                for chunk_index, chunk in enumerate(self._split_chapter(preface)):
                    yield Chunk(book_title, "序章", chunk_index, chunk.strip(), source_path)

        for idx, (title, start, end) in enumerate(zip(titles, boundaries, boundaries[1:])):
            chapter_body = content[start:end].strip()
            if not chapter_body:
                continue
            for chunk_index, chunk in enumerate(self._split_chapter(chapter_body)):
                yield Chunk(book_title, title, chunk_index, chunk.strip(), source_path)


__all__ = ["Chunk", "ChapterTextSplitter"]
