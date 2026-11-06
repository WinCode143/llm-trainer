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
    # Antes de un entrenamiento, libera la VRAM del modelo de inferencia
    job_service.on_before_gpu_job(manager.unload)
    job_service.on_before_gpu_job(rag_service.unload_models)
    job_service.start()
    vllm_engine.kill_orphans()  # restos de un arranque anterior
    # Precarga en segundo plano para que la primera consulta RAG no tarde ~30 s
    threading.Thread(target=rag_service.warmup, daemon=True, name="rag-warmup").start()
    backup.start_scheduler()
    from .services import triggers as _trig
    _trig.arrancar_todo()   # bots de Telegram y tareas programadas
    async with mcp_server.server.session_manager.run():
        yield
    # Apagado ordenado: si quedó un servidor vLLM, se detiene con la app
    manager.unload()


app = FastAPI(title="LLM Trainer", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

for r in (datasets, jobs, models, chat, conversations, rag, evals, agents, flows, providers, questionsets, connections, triggers, system, openai_compat):
    app.include_router(r.router)
# Servidor MCP para agentes externos (debe montarse antes de la ruta comodín del frontend)
app.mount("/mcp", mcp_server.app)

# Sirve el frontend compilado si existe (producción)
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

