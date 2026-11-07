"""Backups of what can't be replaced.

It keeps what can't be downloaded again or regenerated without work: the database (datasets, registered
models, jobs, collections, agents, conversations, evaluations), the dataset files, the RAG indexes
(Chroma), the graphs and the original uploaded documents.

Model weights and checkpoints (data/outputs) are NOT kept: they're huge and can be downloaded
or trained again.
"""
import shutil
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings

INCLUDE = ["llmtrainer.db", "datasets", "rag"]
EXCLUDE_DIRS = {"__pycache__"}


def backups_dir() -> Path:
    d = settings.data_dir / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else (p.stat().st_size if p.exists() else 0)


def create(task=None, note: str = "") -> dict:
    """Creates a .tar.gz with the database, datasets and RAG. Returns the name, size and contents."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = backups_dir() / f"backup-{ts}.tar.gz"
    # The name only goes down to the second: two backups in a row overwrote each other and the first was lost
    n = 1
    while out.exists():
        n += 1
        out = backups_dir() / f"backup-{ts}-{n}.tar.gz"
    items = [(name, settings.data_dir / name) for name in INCLUDE]
    items = [(n, p) for n, p in items if p.exists()]
    total = sum(_size(p) for _, p in items) or 1
    done = 0
    t0 = time.time()
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for name, path in items:
            if task:
                task.update(done / total, f"Copying {name}…")
            tar.add(path, arcname=name, filter=lambda ti: None if Path(ti.name).name in EXCLUDE_DIRS else ti)
            done += _size(path)
    if note:
        (out.with_suffix(".txt")).write_text(note, encoding="utf-8")
    prune()
    return {"file": out.name, "size_mb": round(out.stat().st_size / 1e6, 1), "seconds": round(time.time() - t0, 1),
            "includes": [n for n, _ in items]}


def prune(keep: int | None = None) -> int:
    keep = keep or settings.backup_keep
    files = sorted(backups_dir().glob("backup-*.tar.gz"), key=lambda f: f.stat().st_mtime, reverse=True)
    removed = 0
    for f in files[keep:]:
        f.unlink(missing_ok=True)
        f.with_suffix(".txt").unlink(missing_ok=True)
        removed += 1
    return removed


def listing() -> list[dict]:
    out = []
    for f in sorted(backups_dir().glob("backup-*.tar.gz"), key=lambda f: f.stat().st_mtime, reverse=True):
        note = f.with_suffix(".txt")
        out.append({"file": f.name, "size_mb": round(f.stat().st_size / 1e6, 1),
                    "created_at": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(),
                    "note": note.read_text(encoding="utf-8")[:200] if note.exists() else ""})
    return out


def delete(name: str) -> bool:
    f = backups_dir() / Path(name).name
    if not f.exists() or not f.name.startswith("backup-"):
        return False
    f.unlink()
    f.with_suffix(".txt").unlink(missing_ok=True)
    return True


def restore_instructions(name: str) -> str:
    return (f"To restore: stop the app, move the current data/ folder to data.bak/ and extract the backup:\n"
            f"  cd '{settings.data_dir}' && tar xzf backups/{name}\n"
            "Trained models (data/outputs) aren't in the backup; if you deleted them, they have to be retrained.")


def last_backup_age_hours() -> float | None:
    files = sorted(backups_dir().glob("backup-*.tar.gz"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None
    return (time.time() - files[0].stat().st_mtime) / 3600


def start_scheduler() -> None:
    """Daily background backup (can be turned off with LLMT_BACKUP_DAILY=false)."""
    if not settings.backup_daily:
        return

    def loop():
        while True:
            try:
                age = last_backup_age_hours()
                if age is None or age >= 24:
                    info = create(note="automatic daily backup")
                    print(f"[backup] {info['file']} ({info['size_mb']} MB)")
            except Exception as e:  # noqa: BLE001
                print("[backup] error:", e)
            time.sleep(3600)

    threading.Thread(target=loop, daemon=True, name="backup").start()


def disk_free_gb() -> float:
    return shutil.disk_usage(settings.data_dir).free / 1e9

def verify(name: str) -> dict:
    """Checks that a backup works, without touching anything: it extracts it to a temp folder, opens the database
    and counts the rows. A backup nobody has ever opened isn't a backup, it's a big file."""
    import sqlite3
    import tempfile

    f = backups_dir() / Path(name).name
    if not f.exists() or not f.name.startswith("backup-"):
        return {"ok": False, "error": "That backup doesn't exist"}

    # Tables are read from the model, not from a hand-written list: when “questionset” was added the
    # fixed list fell behind and the backup was considered fine without checking the new table.
    from sqlmodel import SQLModel

    from ..db import Dataset  # noqa: F401  (imports the module so all tables get registered)
    TABLES = sorted(SQLModel.metadata.tables)
    with tempfile.TemporaryDirectory(prefix="llmt-verify-") as tmp:
        dest = Path(tmp)
        try:
            with tarfile.open(f, "r:gz") as tar:
                # filter="data" prevents absolute paths or .. inside the tar
                tar.extractall(dest, filter="data")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"The file is damaged or incomplete: {e}"}

        db = dest / "llmtrainer.db"
        if not db.exists():
            return {"ok": False, "error": "The backup doesn't contain the database"}
        rows, missing = {}, []
        try:
            con = sqlite3.connect(db)
            existing = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            for t in TABLES:
                if t in existing:
                    rows[t] = con.execute(f"select count(*) from {t}").fetchone()[0]  # noqa: S608
                else:
                    missing.append(t)
            con.execute("pragma integrity_check").fetchone()
            con.close()
        except sqlite3.DatabaseError as e:
            return {"ok": False, "error": f"The database doesn't open: {e}"}

        datasets = dest / "datasets"
        rag = dest / "rag"
        return {
            "ok": True,
            "rows": rows,
            # New tables that didn't exist yet when the backup was made: restoring it takes you back to that state
            "missing_tables": missing,
            "dataset_files": len(list(datasets.glob("*"))) if datasets.exists() else 0,
            "rag_collections": len([d for d in rag.iterdir() if d.is_dir()]) if rag.exists() else 0,
            "size_mb": round(f.stat().st_size / 1e6, 1),
        }
