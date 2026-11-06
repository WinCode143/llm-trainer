"""Registro en memoria de tareas en segundo plano dentro del proceso (descargas, evaluaciones, ingesta RAG...)."""
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class Task:
    id: str
    kind: str
    title: str
    status: str = "running"  # running | completed | failed
    progress: float = 0.0  # 0..1
    message: str = ""
    result: Any = None
    error: str | None = None
    cancel_requested: bool = field(default=False, repr=False)

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("cancel_requested", None)
        return d


_tasks: dict[str, Task] = {}
_lock = threading.Lock()


def submit(kind: str, title: str, fn: Callable[[Task], Any]) -> Task:
    task = Task(id=uuid.uuid4().hex[:12], kind=kind, title=title)
    with _lock:
        _tasks[task.id] = task

    def run():
        try:
            resultado = fn(task)
            task.result = resultado
            task.progress = 1.0
            task.status = "completed"   # lo último: antes de esto la tarea sigue «running»
        except Exception as e:  # noqa: BLE001
            # El error se guarda ANTES de marcar el estado: al revés hay una ventana en la que
            # quien consulta ve «failed» con error a None y no puede decir qué pasó.
            task.error = f"{e}\n{traceback.format_exc(limit=5)}"
            task.status = "failed"

    threading.Thread(target=run, daemon=True, name=f"task-{task.id}").start()
    return task


def get(task_id: str) -> Task | None:
    return _tasks.get(task_id)


def all_tasks() -> list[Task]:
    return sorted(_tasks.values(), key=lambda t: t.id, reverse=True)


def cancel(task_id: str) -> bool:
    """Pide a una tarea que pare. Es una bandera: el trabajo tiene que mirarla para enterarse."""
    t = _tasks.get(task_id)
    if not t or t.status != "running":
        return False
    t.cancel_requested = True
    return True
