from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModel, AutoTokenizer

from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Load a local embedding model and create vector representations."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = Path(model_path or settings.embedding_model_path)
        logger.info("Loading embedding model from %s", path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        expected_dim = settings.embedding_dim
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)
            outputs = self.model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                raise ValueError("Model output does not contain last_hidden_state")
            pooled = hidden_states.mean(dim=1)
            vector = pooled.squeeze(0).tolist()
            if len(vector) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {len(vector)}"
                )
            embeddings.append(vector)
        return embeddings


__all__ = ["EmbeddingService"]
