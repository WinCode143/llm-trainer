"""GPU job queue (training/export), run as isolated subprocesses."""
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from sqlmodel import Session, select

from ..config import ROOT, settings
from ..db import Job, Model, engine, now

WORKERS = {"train": "app.workers.train", "export": "app.workers.export", "embed": "app.workers.embed",
           "bench": "app.workers.bench"}
BACKEND_DIR = ROOT / "backend"
UNSLOTH_PYTHON = BACKEND_DIR / ".venv-unsloth" / "bin" / "python"
VLLM_PYTHON = BACKEND_DIR / ".venv-vllm" / "bin" / "python"
BACKEND_PYTHONS = {"unsloth": (UNSLOTH_PYTHON, "backend/setup-unsloth.sh"), "vllm": (VLLM_PYTHON, "backend/setup-vllm.sh")}


def unsloth_available() -> bool:
    return UNSLOTH_PYTHON.exists()


def vllm_available() -> bool:
    return VLLM_PYTHON.exists()


def _python_for(job: Job) -> str:
    entry = BACKEND_PYTHONS.get(job.config.get("backend")) if job.type == "train" else None
    if entry:
        py, script = entry
        if not py.exists():
            raise RuntimeError(f"Engine '{job.config['backend']}' isn't installed: run {script}")
        return str(py)
    return sys.executable

_procs: dict[int, subprocess.Popen] = {}
_wake = threading.Event()
_before_gpu_job = []  # callbacks (e.g. unload the inference model to free VRAM)


def on_before_gpu_job(fn):
    _before_gpu_job.append(fn)


def job_dir(job_id: int) -> Path:
    return settings.jobs_dir / str(job_id)


def read_events(job_id: int, offset: int = 0) -> tuple[list[dict], int]:
    """Returns events from a byte offset, and the new offset."""
    p = job_dir(job_id) / "events.jsonl"
    if not p.exists():
        return [], offset
    with p.open("rb") as f:
        f.seek(offset)
        data = f.read()
    # Complete lines only
    end = data.rfind(b"\n") + 1
    events = [json.loads(l) for l in data[:end].splitlines() if l.strip()]
    return events, offset + end


def last_progress(job_id: int) -> dict | None:
    """The last progress event (reads only the file's tail; cheap for lists)."""
    p = job_dir(job_id) / "events.jsonl"
    if not p.exists():
        return None
    with p.open("rb") as f:
        f.seek(max(0, p.stat().st_size - 16384))
        lines = f.read().splitlines()
    for line in reversed(lines):
        if b'"progress"' in line:
            try:
                ev = json.loads(line)
                return {"step": ev.get("step"), "max_steps": ev.get("max_steps"), "epoch": ev.get("epoch")}
            except ValueError:
                continue
    return None


def read_log(job_id: int, offset: int = 0) -> tuple[str, int]:
    p = job_dir(job_id) / "output.log"
    if not p.exists():
        return "", offset
    with p.open("rb") as f:
        f.seek(offset)
        data = f.read()
    return data.decode("utf-8", errors="replace"), offset + len(data)


def enqueue(name: str, type_: str, config: dict) -> Job:
    with Session(engine) as s:
        job = Job(name=name, type=type_, config=config)
        s.add(job)
        s.commit()
        s.refresh(job)
    _wake.set()
    return job


def cancel(job_id: int) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if not job:
            return
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = now()
            s.add(job)
            s.commit()
            return
    proc = _procs.get(job_id)
    if proc and proc.poll() is None:
        # SIGTERM: the worker stops training and saves
        proc.send_signal(signal.SIGTERM)


def kill(job_id: int) -> None:
    proc = _procs.get(job_id)
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _unique_model_name(s: Session, base: str) -> str:
    name, i = base, 2
    while s.exec(select(Model).where(Model.name == name)).first():
        name, i = f"{base}-{i}", i + 1
    return name


def _register_outputs(s: Session, job: Job, events: list[dict]) -> dict:
    result: dict = {}
    base = job.config.get("base_model")
    for ev in events:
        t = ev["type"]
        if t == "saved":
            kind = "lora" if ev.get("use_lora") else "merged"
            m = Model(name=_unique_model_name(s, job.name), kind=kind, path=ev["path"], base_model=base, job_id=job.id)
            s.add(m)
            result["model"] = ev["path"]
        elif t == "merged":
            m = Model(name=_unique_model_name(s, f"{job.name}-merged"), kind="merged", path=ev["path"],
                      base_model=base, job_id=job.id)
            s.add(m)
            result["merged"] = ev["path"]
        elif t == "gguf":
            m = Model(name=_unique_model_name(s, Path(ev["path"]).stem), kind="gguf", path=ev["path"],
                      base_model=base, job_id=job.id)
            s.add(m)
            result["gguf"] = ev["path"]
        elif t == "saved_embedding":
            m = Model(name=_unique_model_name(s, job.name), kind="embedding", path=ev["path"], base_model=base, job_id=job.id)
            s.add(m)
            result["model"] = ev["path"]
            result["before"], result["after"] = ev.get("before"), ev.get("after")
        elif t == "bench_result":
            result["bench"] = {k: v for k, v in ev.items() if k not in ("type", "t")}
        elif t == "train_result":
            result["metrics"] = {k: v for k, v in ev.items() if k not in ("type", "t")}
        elif t == "log":
            for k in ("eval_loss", "loss", "reward", "mean_token_accuracy"):
                if k in ev:
                    result.setdefault("final", {})[k] = ev[k]
        elif t == "done":
            result["stopped_early"] = ev.get("stopped", False)
        elif t == "error":
            result["error"] = ev.get("message")
    return result


def _run(job: Job) -> None:
    d = job_dir(job.id)
    d.mkdir(parents=True, exist_ok=True)
    cfg_path = d / "config.json"
    cfg_path.write_text(json.dumps(job.config, indent=2, ensure_ascii=False))
    for fn in _before_gpu_job:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("before_gpu_job error:", e)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token
    with (d / "output.log").open("ab") as log:
        proc = subprocess.Popen([_python_for(job), "-m", WORKERS[job.type], str(cfg_path)], cwd=BACKEND_DIR,
                                stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _procs[job.id] = proc
    with Session(engine) as s:
        j = s.get(Job, job.id)
        j.status, j.pid, j.started_at = "running", proc.pid, now()
        s.add(j)
        s.commit()
    code = proc.wait()
    _procs.pop(job.id, None)
    events, _ = read_events(job.id)
    with Session(engine) as s:
        j = s.get(Job, job.id)
        j.result = _register_outputs(s, j, events)
        j.finished_at = now()
        if code == 0:
            j.status = "cancelled" if j.result.get("stopped_early") else "completed"
        else:
            j.status = "failed" if code != -signal.SIGKILL else "cancelled"
            j.error = j.result.get("error") or f"The process ended with code {code}"
        s.add(j)
        s.commit()


def _loop() -> None:
    while True:
        with Session(engine) as s:
            job = s.exec(select(Job).where(Job.status == "queued").order_by(Job.id)).first()
        if job:
            try:
                _run(job)
            except Exception as e:  # noqa: BLE001
                with Session(engine) as s:
                    j = s.get(Job, job.id)
                    j.status, j.error, j.finished_at = "failed", str(e), now()
                    s.add(j)
                    s.commit()
            continue
        _wake.wait(timeout=5)
        _wake.clear()


def start() -> None:
    # Jobs left "running" after a backend restart are orphaned
    with Session(engine) as s:
        for j in s.exec(select(Job).where(Job.status == "running")).all():
            j.status, j.error, j.finished_at = "failed", "The backend restarted while it was running", now()
            s.add(j)
        s.commit()
    threading.Thread(target=_loop, daemon=True, name="job-queue").start()


def is_gpu_busy() -> bool:
    return any(p.poll() is None for p in _procs.values())


__all__ = ["enqueue", "cancel", "kill", "read_events", "read_log", "start", "is_gpu_busy", "on_before_gpu_job"]
