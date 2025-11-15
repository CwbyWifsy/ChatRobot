from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router as api_router
from .logger import configure_logging

app = FastAPI(title="Novel RAG Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 开发环境直接全放开
    allow_credentials=False,  # 我们没带 cookie，可以关掉
    allow_methods=["*"],
    allow_headers=["*"],
)
configure_logging()
app.include_router(api_router, prefix="/api")


__all__ = ["app"]
