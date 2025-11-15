from __future__ import annotations

from fastapi import FastAPI

from .api.routes import router as api_router
from .logger import configure_logging

app = FastAPI(title="Novel RAG Service", version="0.1.0")
configure_logging()
app.include_router(api_router, prefix="/api")


__all__ = ["app"]
