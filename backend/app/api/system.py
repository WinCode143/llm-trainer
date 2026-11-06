import shutil
import subprocess
from pathlib import Path

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import ROOT, settings
from ..services import tasks
from ..services.inference import manager
from ..services.jobs import is_gpu_busy, unsloth_available, vllm_available

router = APIRouter(prefix="/api", tags=["system"])

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:  # noqa: BLE001
    _NVML = False


@router.get("/system")
def system():
    gpus = []
    if _NVML:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpus.append({
                "index": i, "name": pynvml.nvmlDeviceGetName(h),
                "mem_used_gb": round(mem.used / 1e9, 2), "mem_total_gb": round(mem.total / 1e9, 2),
                "util": util.gpu, "temp": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000),
            })
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(settings.data_dir)
    return {
        "gpus": gpus,
        "ram_used_gb": round(vm.used / 1e9, 1), "ram_total_gb": round(vm.total / 1e9, 1),
        "cpu": psutil.cpu_percent(), "disk_free_gb": round(du.free / 1e9, 1),
        "gpu_job_running": is_gpu_busy(), "inference": manager.status(),
        "data_dir": str(settings.data_dir), "gguf_export": bool(settings.llama_cpp), "unsloth": unsloth_available(),
        "vllm": vllm_available(), "tools": tools_status(), "api_key_required": bool(settings.api_key),
    }


def tools_status() -> dict:
    import importlib.util
    return {
        "unsloth": unsloth_available(),
        "vllm": vllm_available(),
        "llama_cpp": bool(settings.llama_cpp),
        "llama_cpp_python": importlib.util.find_spec("llama_cpp") is not None,
        "tesseract": shutil.which("tesseract") is not None,
        "sandbox": __import__("app.services.sandbox", fromlist=["x"]).available(),
        "hf_token": bool(settings.hf_token),
    }


SETUP_SCRIPTS = {"unsloth": "setup-unsloth.sh", "llamacpp": "setup-llamacpp.sh", "vllm": "setup-vllm.sh"}


@router.post("/system/setup/{tool}")
def setup_tool(tool: str):
    """Instala una herramienta opcional ejecutando su script; el progreso se ve en la tarea."""
    script = SETUP_SCRIPTS.get(tool)
    if not script:
        raise HTTPException(404, f"Herramienta desconocida: {tool}")
    path = ROOT / "backend" / script

    def work(task):
        lines: list[str] = []
        proc = subprocess.Popen(["bash", str(path)], cwd=path.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                # Porcentaje de compilación de cmake: "[ 42%] Building ..."
                pct = line[1:4].strip().rstrip("%") if line.startswith("[") and "%]" in line[:6] else None
                task.update(int(pct) / 100 if pct and pct.isdigit() else None, line[-160:])
        if proc.wait() != 0:
            raise RuntimeError("\n".join(lines[-15:]))
        return {"ok": True, "log_tail": lines[-5:]}

    return tasks.submit("setup", f"Instalar {tool}", work).to_dict()


@router.get("/tasks")
def list_tasks():
    return [t.to_dict() for t in tasks.all_tasks()]


class BackupBody(BaseModel):
    note: str = ""


@router.get("/system/backups")
def list_backups():
    from ..services import backup
    return {"backups": backup.listing(), "keep": settings.backup_keep, "daily": settings.backup_daily,
            "dir": str(backup.backups_dir()), "includes": backup.INCLUDE}


@router.post("/system/backups")
def create_backup(body: BackupBody):
    """Copia de la base de datos, los datasets y el RAG (no incluye los pesos de modelos)."""
    from ..services import backup
    return tasks.submit("backup", "Copia de seguridad", lambda task: backup.create(task, body.note)).to_dict()


@router.delete("/system/backups/{name}")
def delete_backup(name: str):
    from ..services import backup
    if not backup.delete(name):
        raise HTTPException(404, "Copia no encontrada")
    return {"ok": True}


@router.post("/system/backups/{name}/verify")
def verify_backup(name: str):
    """Abre la copia en un temporal y comprueba que la base de datos es legible. No toca tus datos."""
    from ..services import backup
    return tasks.submit("verify", f"Verificar {name}", lambda task: backup.verify(name)).to_dict()


@router.get("/system/backups/{name}/restore-help")
def restore_help(name: str):
    from ..services import backup
    return {"instructions": backup.restore_instructions(name)}


@router.post("/tasks/{tid}/cancel")
def cancel_task(tid: str):
    """Pide a una tarea larga (enriquecer, resúmenes…) que pare en el siguiente paso."""
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404)
    tasks.cancel(tid)
    return {"ok": True}


@router.get("/tasks/{tid}")
def get_task(tid: str):
    t = tasks.get(tid)
    if not t:
        raise HTTPException(404)
    return t.to_dict()
