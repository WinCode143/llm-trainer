import json
import signal
import sys
import time
from dataclasses import fields
from pathlib import Path

CHATML_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


class Emitter:
    """Writes JSONL events that the backend reads to show live progress."""

    def __init__(self, job_dir: Path):
        self.path = job_dir / "events.jsonl"
        self.f = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, type_: str, **data):
        self.f.write(json.dumps({"type": type_, "t": time.time(), **data}, ensure_ascii=False) + "\n")

    def status(self, message: str):
        print(f"[status] {message}", flush=True)
        self.emit("status", message=message)


class StopFlag:
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, *_):
        print("[worker] Stop signal received; the checkpoint will be saved and the run will end.", flush=True)
        self.stop = True


def filter_kwargs(cls, kwargs: dict) -> dict:
    """Drops kwargs that the installed version of the class (a config dataclass) doesn't accept."""
    names = {f.name for f in fields(cls)}
    dropped = [k for k in kwargs if k not in names]
    if dropped:
        print(f"[worker] Parameters not supported by {cls.__name__} (ignored): {dropped}", flush=True)
    return {k: v for k, v in kwargs.items() if k in names}


def load_config() -> tuple[dict, Path]:
    cfg_path = Path(sys.argv[1])
    return json.loads(cfg_path.read_text()), cfg_path.parent
