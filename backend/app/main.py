import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import agents, chat, connections, conversations, datasets, evals, flows, jobs, mcp_server, models, openai_compat, providers, questionsets, rag, system, triggers
from .config import ROOT, settings
from .db import init_db
from .services import jobs as job_service
from .services import backup
from .services import rag as rag_service
from .services import vllm_engine
from .services.inference import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Before a training run, free the inference model's VRAM
    job_service.on_before_gpu_job(manager.unload)
    job_service.on_before_gpu_job(rag_service.unload_models)
    job_service.start()
    vllm_engine.kill_orphans()  # leftovers from a previous start
    # Preload in the background so the first RAG query doesn't take ~30 s
    threading.Thread(target=rag_service.warmup, daemon=True, name="rag-warmup").start()
    backup.start_scheduler()
    from .services import triggers as _trig
    _trig.start_all()   # Telegram bots and scheduled tasks
    async with mcp_server.server.session_manager.run():
        yield
    # Orderly shutdown: if a vLLM server was left, it's stopped with the app
    manager.unload()


app = FastAPI(title="LLM Trainer", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

for r in (datasets, jobs, models, chat, conversations, rag, evals, agents, flows, providers, questionsets, connections, triggers, system, openai_compat):
    app.include_router(r.router)
# MCP server for external agents (it must be mounted before the frontend's catch-all route)
app.mount("/mcp", mcp_server.app)

# Serve the built frontend if it exists (production)
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")


def run():
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()

