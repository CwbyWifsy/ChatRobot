from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


class NovelHasher:
    """Compute hash values for novel files combining metadata and content."""

    def __init__(self, algorithms: Iterable[str] | None = None) -> None:
        self.algorithms = list(algorithms or ["sha256"])

    def hash_file(self, path: Path, extra_values: Iterable[str] | None = None) -> str:
        path = Path(path)
        hashers = [hashlib.new(algo) for algo in self.algorithms]

        extra = "".join(extra_values or [])
        for hasher in hashers:
            hasher.update(path.name.encode("utf-8"))
            if extra:
                hasher.update(extra.encode("utf-8"))

        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(8192), b""):
                for hasher in hashers:
                    hasher.update(chunk)

        return ":".join(hasher.hexdigest() for hasher in hashers)


__all__ = ["NovelHasher"]
