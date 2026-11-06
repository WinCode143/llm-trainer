import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..config import settings
from ..db import Model, engine, get_session
from ..services import tasks
from ..services.inference import manager
from ..services.jobs import is_gpu_busy

router = APIRouter(prefix="/api/models", tags=["models"])

SUGGESTED = [
    {"repo": "Qwen/Qwen3-0.6B", "size": "0.6B", "note": "Muy rápido; ideal para probar el pipeline"},
    {"repo": "Qwen/Qwen3-4B-Instruct-2507", "size": "4B", "note": "Gran relación calidad/VRAM"},
    {"repo": "Qwen/Qwen3-8B", "size": "8B", "note": "QLoRA cómodo en 24 GB"},
    {"repo": "meta-llama/Llama-3.1-8B-Instruct", "size": "8B", "note": "Requiere aceptar licencia en HF"},
    {"repo": "google/gemma-3-4b-it", "size": "4B", "note": "Multilingüe, requiere licencia"},
    {"repo": "mistralai/Mistral-Nemo-Instruct-2407", "size": "12B", "note": "QLoRA 4-bit en 24 GB"},
    {"repo": "Qwen/Qwen3-14B", "size": "14B", "note": "QLoRA 4-bit al límite; batch 1, seq corta"},
    {"repo": "Qwen/Qwen2.5-VL-3B-Instruct", "size": "3B", "note": "Visión: chatear con imágenes y describir las de los PDF"},
    {"repo": "Qwen/Qwen3-VL-8B-Instruct", "size": "8B", "note": "Visión más capaz (usa carga en 4-bit)"},
]


@router.get("")
def list_models(s: Session = Depends(get_session)):
    return s.exec(select(Model).order_by(Model.created_at.desc())).all()


@router.get("/suggested")
def suggested():
    return SUGGESTED


class RegisterBody(BaseModel):
    name: str
    path: str  # repo de HF o ruta local
    kind: str = "hf"
    base_model: str | None = None


@router.post("")
def register(body: RegisterBody, s: Session = Depends(get_session)):
    if s.exec(select(Model).where(Model.name == body.name)).first():
        raise HTTPException(409, "Ya existe un modelo con ese nombre")
    kind = body.kind
    p = Path(body.path).expanduser()
    if p.suffix == ".gguf":
        kind = "gguf"
    elif p.exists() and (p / "adapter_config.json").exists():
        kind = "lora"
        if not body.base_model:
            import json
            body.base_model = json.loads((p / "adapter_config.json").read_text()).get("base_model_name_or_path")
    m = Model(name=body.name, kind=kind, path=str(p) if p.exists() else body.path, base_model=body.base_model)
    s.add(m)
    s.commit()
    s.refresh(m)
    return m


class DownloadBody(BaseModel):
    repo: str
    name: str | None = None
    allow_patterns: list[str] | None = None


def _nombre_libre(s: Session, base: str) -> str:
    """Primer nombre sin usar de la forma base, base-2, base-3…

    Comprobar una sola vez no bastaba: dos descargas simultáneas del mismo repo pasaban las dos la
    comprobación y la segunda reventaba contra la restricción de unicidad de la tabla."""
    nombre, n = base, 1
    while s.exec(select(Model).where(Model.name == nombre)).first():
        n += 1
        nombre = f"{base}-{n}"
    return nombre


def _barra(task, repo: str):
    """Barra de progreso de Hugging Face conectada a nuestra tarea.

    Hace dos cosas que no había: enseña cuánto lleva descargado, y permite cancelar de verdad.
    Antes «cancelar» sólo marcaba una bandera que `snapshot_download` no miraba, así que la descarga
    seguía hasta el final y luego chocaba al intentar registrarse."""
    from tqdm.auto import tqdm as _tqdm

    corto = repo.split("/")[-1]

    class Barra(_tqdm):
        def update(self, n=1):
            r = super().update(n)
            if getattr(task, "cancel_requested", False):
                raise RuntimeError(f"Descarga de {corto} cancelada")
            if self.total:
                # Hugging Face abre varias barras: una cuenta archivos y otras bytes de cada uno.
                # Sin mirar la unidad, la de archivos salía como «0.0 / 0.0 GB».
                if getattr(self, "unit", "") == "B":
                    texto = f"{corto} · {self.n / 1e9:.1f} / {self.total / 1e9:.1f} GB"
                else:
                    texto = f"{corto} · archivo {self.n} de {self.total}"
                task.update(self.n / self.total, texto)
            return r

    return Barra


@router.post("/download")
def download(body: DownloadBody):
    """Descarga un repo de HF a data/models y lo registra."""

    def work(task):
        from huggingface_hub import snapshot_download

        task.update(message=f"Descargando {body.repo}…")
        local = settings.models_dir / body.repo.replace("/", "__")
        snapshot_download(body.repo, local_dir=local, token=settings.hf_token,
                          allow_patterns=body.allow_patterns, tqdm_class=_barra(task, body.repo))
        ggufs = list(local.glob("*.gguf"))
        base = body.name or body.repo.split("/")[-1]
        ruta = str(ggufs[0]) if ggufs else str(local)
        with Session(engine) as s:
            # Si ya está registrada esa misma ruta, no se duplica: es la misma descarga repetida
            ya = s.exec(select(Model).where(Model.path == ruta)).first()
            if ya:
                task.update(message=f"{base} ya estaba registrado")
                return ya.model_dump(mode="json")
            m = Model(name=_nombre_libre(s, base), kind="gguf" if ggufs else "hf", path=ruta)
            s.add(m)
            s.commit()
            s.refresh(m)
            return m.model_dump(mode="json")

    return tasks.submit("download", f"Descargar {body.repo}", work).to_dict()


@router.get("/inference/vram")
def vram():
    """Quién está usando la VRAM: nuestro proceso, los modelos del RAG y el resto del sistema."""
    import os

    from ..services import rag as rag_svc
    from ..services.inference import _vram_libre_gb

    v = _vram_libre_gb()
    propia = 0.0
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        mio = str(os.getpid())
        for linea in out.strip().splitlines():
            pid, mb = (x.strip() for x in linea.split(","))
            if pid == mio:
                propia = float(mb) / 1024
    except Exception:  # noqa: BLE001
        pass
    return {
        "libre_gb": round(v[0], 1) if v else None,
        "total_gb": round(v[1], 1) if v else None,
        "nuestro_gb": round(propia, 1),
        "modelo": manager.current.name if manager.current else None,
        "rag_modelos": rag_svc.loaded_model_names(),
    }


@router.post("/{mid}/ollama")
def a_ollama(mid: int, nombre: str | None = None, s: Session = Depends(get_session)):
    """Registra un GGUF en Ollama con `ollama create`, sin copiar el archivo.

    El Modelfile ya se generaba al exportar, pero había que abrir una terminal y acordarse del
    comando. Esto lo ejecuta y deja el modelo listo para `ollama run`."""
    import shutil
    import subprocess

    m = s.get(Model, mid)
    if not m:
        raise HTTPException(404)
    if m.kind != "gguf":
        raise HTTPException(400, "Ollama sólo admite GGUF. Exporta antes el modelo a GGUF desde esta pantalla.")
    if not shutil.which("ollama"):
        raise HTTPException(400, "Ollama no está instalado o no está en el PATH.")
    gguf = Path(m.path)
    if not gguf.exists():
        raise HTTPException(400, f"No se encuentra el archivo {m.path}")

    destino = (nombre or m.name).lower().replace(" ", "-")
    modelfile = gguf.parent / f"Modelfile.{destino}"
    modelfile.write_text(f"FROM {gguf}" + chr(10), encoding="utf-8")
    try:
        r = subprocess.run(["ollama", "create", destino, "-f", str(modelfile)],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Ollama tardó demasiado en registrar el modelo") from None
    if r.returncode != 0:
        raise HTTPException(500, f"ollama create falló: {(r.stderr or r.stdout)[:300]}")
    return {"ok": True, "nombre": destino, "comando": f"ollama run {destino}"}


@router.get("/sizes")
def sizes(s: Session = Depends(get_session)):
    """Lo que ocupa cada modelo en disco, y si algo lo está usando.

    Va en su propia llamada porque recorrer las carpetas es lento y la lista de modelos se pide a
    menudo; así la tabla aparece al instante y los tamaños llegan después."""
    from ..db import Agent, RagCollection

    agentes = {a.model_id: a.name for a in s.exec(select(Agent)).all() if a.model_id}
    embed_en_uso = {c.embed_model: c.name for c in s.exec(select(RagCollection)).all()}
    out = {}
    for m in s.exec(select(Model)).all():
        # El uso se calcula siempre, también para los que viven en la caché de Hugging Face:
        # un modelo base sin archivos propios puede ser justo el que usan tus agentes.
        usos = []
        if m.id in agentes:
            usos.append(f"agente «{agentes[m.id]}»")
        if m.path in embed_en_uso:
            usos.append(f"colección «{embed_en_uso[m.path]}»")
        p = Path(m.path)
        if not p.exists():
            out[m.id] = {"mb": None, "cached": True, "used_by": usos}
            continue
        mb = p.stat().st_size / 1e6 if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
        out[m.id] = {"mb": round(mb, 1), "cached": False, "used_by": usos}
    return out


def _carpetas_gestionadas() -> list[Path]:
    """Carpetas cuyos archivos puede borrar la app, ya resueltas.

    Se resuelven porque `data/outputs` puede ser un enlace a otro disco: comparar sin resolver
    hacía que el borrado no encontrara nunca la carpeta y no llegara a borrar nada."""
    return [d.resolve() for d in (settings.data_dir, settings.models_dir, settings.outputs_dir) if d.exists()]


def _borrar_si_es_nuestro(p: Path) -> bool:
    """Borra la ruta sólo si está dentro de las carpetas de la app. Devuelve si borró algo."""
    if not p.exists():
        return True   # no hay nada que borrar: no es un error
    real = p.resolve()
    if not any(raiz == real or raiz in real.parents for raiz in _carpetas_gestionadas()):
        return False
    shutil.rmtree(real) if real.is_dir() else real.unlink()
    return True


@router.delete("/{mid}")
def delete(mid: int, delete_files: bool = False, s: Session = Depends(get_session)):
    m = s.get(Model, mid)
    if not m:
        raise HTTPException(404)
    if manager.current and manager.current.model_id == mid:
        manager.unload()
    if delete_files:
        borrado = _borrar_si_es_nuestro(Path(m.path))
        if not borrado:
            # Antes esto fallaba en silencio: la fila desaparecía y los archivos se quedaban en disco
            raise HTTPException(400, f"No se pueden borrar los archivos de {m.path}: están fuera de las carpetas "
                                     "que gestiona la aplicación. Bórralos a mano si quieres recuperar el espacio.")
    s.delete(m)
    s.commit()
    return {"ok": True}


# ---------------- inferencia ----------------
class LoadBody(BaseModel):
    model_id: int
    quantization: str = "none"  # none | 8bit | 4bit
    n_ctx: int = 8192
    engine: str = "auto"  # auto (transformers / llama.cpp) | vllm
    gpu_util: float = 0.6


@router.get("/inference/status")
def inference_status():
    return manager.status()


@router.post("/inference/load")
def load(body: LoadBody, s: Session = Depends(get_session)):
    if is_gpu_busy():
        raise HTTPException(409, "Hay un entrenamiento/exportación en marcha usando la GPU")
    m = s.get(Model, body.model_id)
    if not m:
        raise HTTPException(404)
    if m.kind == "embedding":
        raise HTTPException(400, "Es un modelo de embeddings: úsalo en una colección RAG (Ajustes → Re-indexar)")

    def work(task):
        task.update(message=f"Cargando {m.name}…")
        return manager.load(model_id=m.id, name=m.name, kind=m.kind, path=m.path, base_model=m.base_model,
                            quantization=body.quantization, n_ctx=body.n_ctx, engine=body.engine,
                            gpu_util=body.gpu_util)

    return tasks.submit("load", f"Cargar {m.name}", work).to_dict()


@router.post("/inference/unload")
def unload(everything: bool = True):
    """Saca de la VRAM lo que tengamos cargado.

    `everything` incluye los modelos de embeddings y el reranker del RAG, que se quedan cacheados
    entre búsquedas y son ~3 GB: sin ellos no se recupera todo el espacio y la descarga parece que
    no ha servido de nada."""
    from ..services.inference import _vram_libre_gb

    antes = _vram_libre_gb()
    cargado = manager.current.name if manager.current else None
    manager.unload()
    rag_gb = 0.0
    if everything:
        from ..services import rag as rag_svc
        rag_gb = rag_svc.unload_models() or 0.0
    despues = _vram_libre_gb()
    liberado = round((despues[0] - antes[0]), 1) if antes and despues else None
    return {"ok": True, "modelo": cargado, "rag_gb": rag_gb, "liberado_gb": liberado,
            "libre_gb": round(despues[0], 1) if despues else None}
