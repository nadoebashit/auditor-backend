"""FastAPI application entrypoint."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html

# Ensure the project root is on the import path when running as ``python app/main.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core.db import Base, engine, _import_models
from app.core.logging import configure_logging, get_logger
from app.modules.auth.router import router as auth_router
from app.modules.files.router import router as files_router

configure_logging()
logger = get_logger(__name__)

_import_models()
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="OSON Document Intelligence",
    description="Авторизация для административной панели и Telegram Mini App",
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.modules.chats.router import router as chats_router
from app.modules.customers.router import router as customers_router
from app.modules.prompts.router import router as prompts_router
from app.modules.rag.router import router as rag_router
from app.modules.actions.router import router as actions_router
from app.modules.projects.router import router as projects_router
from app.modules.chat_api.router import router as chat_api_router
from app.modules.knowledge.router import router as knowledge_router

app.include_router(auth_router)
app.include_router(files_router)
app.include_router(customers_router)
app.include_router(chats_router)
app.include_router(rag_router)
app.include_router(prompts_router)

# API v1 (docs-compatible) without breaking existing legacy routes
api_v1 = FastAPI(
    title="OSON Document Intelligence API v1",
    description="Versioned API surface (docs-compatible).",
    swagger_ui_parameters={"persistAuthorization": True},
)

api_v1.include_router(auth_router)
api_v1.include_router(files_router)
api_v1.include_router(customers_router)
api_v1.include_router(chats_router)
api_v1.include_router(rag_router)
api_v1.include_router(prompts_router)
api_v1.include_router(projects_router)
api_v1.include_router(actions_router)
api_v1.include_router(chat_api_router)
api_v1.include_router(knowledge_router)

app.mount("/api/v1", api_v1)


@app.get("/docs-v1", include_in_schema=False)
async def swagger_v1():
    return get_swagger_ui_html(
        openapi_url="/api/v1/openapi.json",
        title="OSON Document Intelligence API v1",
    )


@app.get("/redoc-v1", include_in_schema=False)
async def redoc_v1():
    return get_redoc_html(
        openapi_url="/api/v1/openapi.json",
        title="OSON Document Intelligence API v1",
    )


@app.get("/health", tags=["health"])
async def health_check():
    """Проверка работоспособности сервера."""
    return {"status": "ok", "service": "auditor-backend"}
