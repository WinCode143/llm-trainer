"""In-memory registry of background tasks inside the process (downloads, evaluations, RAG ingestion...)."""
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
            result = fn(task)
            task.result = result
            task.progress = 1.0
            task.status = "completed"   # last: until then the task is still “running”
        except Exception as e:  # noqa: BLE001
            # The error is saved BEFORE marking the status: the other way round there's a window in which
            # whoever checks sees “failed” with error None and can't tell what happened.
            task.error = f"{e}\n{traceback.format_exc(limit=5)}"
            task.status = "failed"

    threading.Thread(target=run, daemon=True, name=f"task-{task.id}").start()
    return task


def get(task_id: str) -> Task | None:
    return _tasks.get(task_id)


def all_tasks() -> list[Task]:
    return sorted(_tasks.values(), key=lambda t: t.id, reverse=True)


def cancel(task_id: str) -> bool:
    """Asks a task to stop. It's a flag: the work has to check it to find out."""
    t = _tasks.get(task_id)
    if not t or t.status != "running":
        return False
    t.cancel_requested = True
    return True
