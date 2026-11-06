"""Copias de seguridad de lo irreemplazable.

Se guarda lo que no se puede volver a descargar ni regenerar sin trabajo: la base de datos (datasets, modelos
registrados, trabajos, colecciones, agentes, conversaciones, evaluaciones), los archivos de dataset, los índices
de RAG (Chroma), los grafos y los documentos originales subidos.

NO se guardan los pesos de modelos ni los checkpoints (data/outputs): son enormes y se pueden volver a
descargar o entrenar.
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
    """Crea un .tar.gz con la base de datos, datasets y RAG. Devuelve nombre, tamaño y contenido."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = backups_dir() / f"backup-{ts}.tar.gz"
    # El nombre sólo llega al segundo: dos copias seguidas se pisaban y la primera se perdía
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
                task.update(done / total, f"Copiando {name}…")
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
    return (f"Para restaurar: para la app, mueve la carpeta data/ actual a data.bak/ y extrae la copia:\n"
            f"  cd '{settings.data_dir}' && tar xzf backups/{name}\n"
            "Los modelos entrenados (data/outputs) no están en la copia; si los borraste, hay que reentrenarlos.")


def last_backup_age_hours() -> float | None:
    files = sorted(backups_dir().glob("backup-*.tar.gz"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None
    return (time.time() - files[0].stat().st_mtime) / 3600


def start_scheduler() -> None:
    """Copia diaria en segundo plano (desactivable con LLMT_BACKUP_DAILY=false)."""
    if not settings.backup_daily:
        return

    def loop():
        while True:
            try:
                age = last_backup_age_hours()
                if age is None or age >= 24:
                    info = create(note="copia automática diaria")
                    print(f"[backup] {info['file']} ({info['size_mb']} MB)")
            except Exception as e:  # noqa: BLE001
                print("[backup] error:", e)
            time.sleep(3600)

    threading.Thread(target=loop, daemon=True, name="backup").start()


def disk_free_gb() -> float:
    return shutil.disk_usage(settings.data_dir).free / 1e9

def verify(name: str) -> dict:
    """Comprueba que una copia sirve, sin tocar nada: la extrae en un temporal, abre la base de datos
    y cuenta las filas. Un backup que nadie ha abierto nunca no es un backup, es un archivo grande."""
    import sqlite3
    import tempfile

    f = backups_dir() / Path(name).name
    if not f.exists() or not f.name.startswith("backup-"):
        return {"ok": False, "error": "No existe esa copia"}

    # Las tablas se leen del modelo, no de una lista escrita a mano: al añadir «questionset» la
    # lista fija se quedó atrás y la copia se daba por buena sin comprobar la tabla nueva.
    from sqlmodel import SQLModel

    from ..db import Dataset  # noqa: F401  (importa el módulo para que se registren todas)
    TABLAS = sorted(SQLModel.metadata.tables)
    with tempfile.TemporaryDirectory(prefix="llmt-verify-") as tmp:
        dest = Path(tmp)
        try:
            with tarfile.open(f, "r:gz") as tar:
                # filter="data" evita rutas absolutas o con .. dentro del tar
                tar.extractall(dest, filter="data")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"El archivo está dañado o incompleto: {e}"}

        db = dest / "llmtrainer.db"
        if not db.exists():
            return {"ok": False, "error": "La copia no contiene la base de datos"}
        filas, faltan = {}, []
        try:
            con = sqlite3.connect(db)
            existen = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            for t in TABLAS:
                if t in existen:
                    filas[t] = con.execute(f"select count(*) from {t}").fetchone()[0]  # noqa: S608
                else:
                    faltan.append(t)
            con.execute("pragma integrity_check").fetchone()
            con.close()
        except sqlite3.DatabaseError as e:
            return {"ok": False, "error": f"La base de datos no abre: {e}"}

        datasets = dest / "datasets"
        rag = dest / "rag"
        return {
            "ok": True,
            "rows": filas,
            # Tablas nuevas que aún no existían cuando se hizo la copia: restaurarla te devuelve al estado de entonces
            "missing_tables": faltan,
            "dataset_files": len(list(datasets.glob("*"))) if datasets.exists() else 0,
            "rag_collections": len([d for d in rag.iterdir() if d.is_dir()]) if rag.exists() else 0,
            "size_mb": round(f.stat().st_size / 1e6, 1),
        }
