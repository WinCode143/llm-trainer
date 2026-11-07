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
    {"repo": "Qwen/Qwen3-0.6B", "size": "0.6B", "note": "Very fast; ideal for testing the pipeline"},
    {"repo": "Qwen/Qwen3-4B-Instruct-2507", "size": "4B", "note": "Great quality per GB of VRAM"},
    {"repo": "Qwen/Qwen3-8B", "size": "8B", "note": "Comfortable QLoRA on 24 GB"},
    {"repo": "meta-llama/Llama-3.1-8B-Instruct", "size": "8B", "note": "Requires accepting the license on HF"},
    {"repo": "google/gemma-3-4b-it", "size": "4B", "note": "Multilingual, requires a license"},
    {"repo": "mistralai/Mistral-Nemo-Instruct-2407", "size": "12B", "note": "QLoRA 4-bit en 24 GB"},
    {"repo": "Qwen/Qwen3-14B", "size": "14B", "note": "4-bit QLoRA at the limit; batch 1, short sequences"},
    {"repo": "Qwen/Qwen2.5-VL-3B-Instruct", "size": "3B", "note": "Vision: chat with images and describe the ones in PDFs"},
    {"repo": "Qwen/Qwen3-VL-8B-Instruct", "size": "8B", "note": "A more capable vision model (loads in 4-bit)"},
]


@router.get("")
def list_models(s: Session = Depends(get_session)):
    return s.exec(select(Model).order_by(Model.created_at.desc())).all()


@router.get("/suggested")
def suggested():
    return SUGGESTED


class RegisterBody(BaseModel):
    name: str
    path: str  # HF repo or local path
    kind: str = "hf"
    base_model: str | None = None


@router.post("")
def register(body: RegisterBody, s: Session = Depends(get_session)):
    if s.exec(select(Model).where(Model.name == body.name)).first():
        raise HTTPException(409, "There's already a model with that name")
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


def _free_name(s: Session, base: str) -> str:
    """First unused name of the form base, base-2, base-3…

    Checking once wasn't enough: two simultaneous downloads of the same repo both passed the
    check and the second one crashed against the table's uniqueness constraint."""
    name, n = base, 1
    while s.exec(select(Model).where(Model.name == name)).first():
        n += 1
        name = f"{base}-{n}"
    return name


def _barra(task, repo: str):
    """Hugging Face progress bar wired to our task.

    It does two things that were missing: it shows how much has been downloaded, and it really allows cancelling.
    “Cancel” used to only set a flag that `snapshot_download` never checked, so the download
    went on to the end and then crashed while registering."""
    from tqdm.auto import tqdm as _tqdm

    short = repo.split("/")[-1]

    class Barra(_tqdm):
        def update(self, n=1):
            r = super().update(n)
            if getattr(task, "cancel_requested", False):
                raise RuntimeError(f"Download of {short} cancelled")
            if self.total:
                # Hugging Face opens several bars: one counts files and others the bytes of each file.
                # Without checking the unit, the files one showed up as “0.0 / 0.0 GB”.
                if getattr(self, "unit", "") == "B":
                    text = f"{short} · {self.n / 1e9:.1f} / {self.total / 1e9:.1f} GB"
                else:
                    text = f"{short} · file {self.n} of {self.total}"
                task.update(self.n / self.total, text)
            return r

    return Barra


@router.post("/download")
def download(body: DownloadBody):
    """Downloads an HF repo to data/models and registers it."""

    def work(task):
        from huggingface_hub import snapshot_download

        task.update(message=f"Downloading {body.repo}…")
        local = settings.models_dir / body.repo.replace("/", "__")
        snapshot_download(body.repo, local_dir=local, token=settings.hf_token,
                          allow_patterns=body.allow_patterns, tqdm_class=_barra(task, body.repo))
        ggufs = list(local.glob("*.gguf"))
        base = body.name or body.repo.split("/")[-1]
        path_ = str(ggufs[0]) if ggufs else str(local)
        with Session(engine) as s:
            # If that same path is already registered, it isn't duplicated: it's the same download repeated
            existing = s.exec(select(Model).where(Model.path == path_)).first()
            if existing:
                task.update(message=f"{base} was already registered")
                return existing.model_dump(mode="json")
            m = Model(name=_free_name(s, base), kind="gguf" if ggufs else "hf", path=path_)
            s.add(m)
            s.commit()
            s.refresh(m)
            return m.model_dump(mode="json")

    return tasks.submit("download", f"Download {body.repo}", work).to_dict()


@router.get("/inference/vram")
def vram():
    """Who is using VRAM: our process, the RAG models and the rest of the system."""
    import os

    from ..services import rag as rag_svc
    from ..services.inference import _vram_free_gb

    v = _vram_free_gb()
    own = 0.0
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        mio = str(os.getpid())
        for line in out.strip().splitlines():
            pid, mb = (x.strip() for x in line.split(","))
            if pid == mio:
                own = float(mb) / 1024
    except Exception:  # noqa: BLE001
        pass
    return {
        "libre_gb": round(v[0], 1) if v else None,
        "total_gb": round(v[1], 1) if v else None,
        "nuestro_gb": round(own, 1),
        "modelo": manager.current.name if manager.current else None,
        "rag_modelos": rag_svc.loaded_model_names(),
    }


@router.post("/{mid}/ollama")
def to_ollama(mid: int, name: str | None = None, s: Session = Depends(get_session)):
    """Registers a GGUF in Ollama with `ollama create`, without copying the file.

    The Modelfile was already generated on export, but you had to open a terminal and remember the
    command. This runs it and leaves the model ready for `ollama run`."""
    import shutil
    import subprocess

    m = s.get(Model, mid)
    if not m:
        raise HTTPException(404)
    if m.kind != "gguf":
        raise HTTPException(400, "Ollama only supports GGUF. Export the model to GGUF from this screen first.")
    if not shutil.which("ollama"):
        raise HTTPException(400, "Ollama isn't installed or isn't on the PATH.")
    gguf = Path(m.path)
    if not gguf.exists():
        raise HTTPException(400, f"Can't find the file {m.path}")

    target = (name or m.name).lower().replace(" ", "-")
    modelfile = gguf.parent / f"Modelfile.{target}"
    modelfile.write_text(f"FROM {gguf}" + chr(10), encoding="utf-8")
    try:
        r = subprocess.run(["ollama", "create", target, "-f", str(modelfile)],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "Ollama took too long to register the model") from None
    if r.returncode != 0:
        raise HTTPException(500, f"ollama create failed: {(r.stderr or r.stdout)[:300]}")
    return {"ok": True, "nombre": target, "comando": f"ollama run {target}"}


@router.get("/sizes")
def sizes(s: Session = Depends(get_session)):
    """How much disk each model takes, and whether something is using it.

    It's a separate call because walking the folders is slow and the model list is requested
    often; this way the table shows up instantly and the sizes arrive afterwards."""
    from ..db import Agent, RagCollection

    agents = {a.model_id: a.name for a in s.exec(select(Agent)).all() if a.model_id}
    embed_in_use = {c.embed_model: c.name for c in s.exec(select(RagCollection)).all()}
    out = {}
    for m in s.exec(select(Model)).all():
        # Usage is always computed, also for models living in the Hugging Face cache:
        # a base model with no files of its own may be exactly the one your agents use.
        uses = []
        if m.id in agents:
            uses.append(f"agent “{agents[m.id]}”")
        if m.path in embed_in_use:
            uses.append(f"collection “{embed_in_use[m.path]}”")
        p = Path(m.path)
        if not p.exists():
            out[m.id] = {"mb": None, "cached": True, "used_by": uses}
            continue
        mb = p.stat().st_size / 1e6 if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
        out[m.id] = {"mb": round(mb, 1), "cached": False, "used_by": uses}
    return out


def _managed_dirs() -> list[Path]:
    """Folders whose files the app may delete, already resolved.

    They're resolved because `data/outputs` may be a link to another disk: comparing unresolved paths
    meant deletion never found the folder and never deleted anything."""
    return [d.resolve() for d in (settings.data_dir, settings.models_dir, settings.outputs_dir) if d.exists()]


def _delete_if_ours(p: Path) -> bool:
    """Deletes the path only if it's inside the app's folders. Returns whether it deleted anything."""
    if not p.exists():
        return True   # there's nothing to delete: that's not an error
    real = p.resolve()
    if not any(root == real or root in real.parents for root in _managed_dirs()):
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
        deleted = _delete_if_ours(Path(m.path))
        if not deleted:
            # This used to fail silently: the row disappeared and the files stayed on disk
            raise HTTPException(400, f"Can't delete the files of {m.path}: they're outside the folders "
                                     "the application manages. Delete them by hand if you want the space back.")
    s.delete(m)
    s.commit()
    return {"ok": True}


# ---------------- inference ----------------
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
        raise HTTPException(409, "A training run/export is using the GPU")
    m = s.get(Model, body.model_id)
    if not m:
        raise HTTPException(404)
    if m.kind == "embedding":
        raise HTTPException(400, "It's an embedding model: use it in a RAG collection (Settings → Re-index)")

    def work(task):
        task.update(message=f"Loading {m.name}…")
        return manager.load(model_id=m.id, name=m.name, kind=m.kind, path=m.path, base_model=m.base_model,
                            quantization=body.quantization, n_ctx=body.n_ctx, engine=body.engine,
                            gpu_util=body.gpu_util)

    return tasks.submit("load", f"Load {m.name}", work).to_dict()


@router.post("/inference/unload")
def unload(everything: bool = True):
    """Unloads whatever we have in VRAM.

    `everything` includes the RAG's embedding models and reranker, which stay cached
    between searches and take ~3 GB: without them not all the space is freed and the unload seems
    to have done nothing."""
    from ..services.inference import _vram_free_gb

    before = _vram_free_gb()
    loaded = manager.current.name if manager.current else None
    manager.unload()
    rag_gb = 0.0
    if everything:
        from ..services import rag as rag_svc
        rag_gb = rag_svc.unload_models() or 0.0
    after = _vram_free_gb()
    freed = round((after[0] - before[0]), 1) if before and after else None
    return {"ok": True, "modelo": loaded, "rag_gb": rag_gb, "liberado_gb": freed,
            "libre_gb": round(after[0], 1) if after else None}
