"""Aislamiento para el código que ejecutan los agentes (herramienta «Ejecutar Python»).

Usa **bubblewrap** (bwrap): espacios de nombres de usuario, sin root y con arranque de milisegundos.
Dentro del sandbox el proceso ve:
  - /usr y /etc en sólo lectura (lo necesario para que Python arranque)
  - el intérprete y sus paquetes en sólo lectura
  - una carpeta de trabajo temporal, la única escribible
  - opcionalmente una carpeta de datos tuya en sólo lectura
No ve tu /home, ni el proyecto, ni la base de datos, ni los modelos. La red está cortada salvo que la actives.
Además se aplican límites de memoria, CPU y tamaño de archivo, y un tiempo máximo.

Si bwrap no está disponible se ejecuta sin aislamiento (con límites de recursos) y la interfaz lo advierte.
"""
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

BWRAP = shutil.which("bwrap")


@lru_cache(maxsize=1)
def available() -> bool:
    """¿Funciona bwrap en esta máquina? (algunos kernels desactivan los user namespaces)."""
    if not BWRAP:
        return False
    try:
        # Los enlaces de /lib son imprescindibles: sin ellos no existe el cargador dinámico dentro del sandbox
        p = subprocess.run([BWRAP, "--ro-bind", "/usr", "/usr",
                            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                            "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
                            "--unshare-all", "--die-with-parent", "/usr/bin/true"],
                           capture_output=True, timeout=15)
        return p.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def status() -> dict:
    return {"backend": "bubblewrap" if available() else "ninguno", "available": available(),
            "detail": ("Aísla sistema de archivos, red y procesos sin root."
                       if available() else "Instala bubblewrap (paquete `bubblewrap`) para aislar el código de los agentes.")}


def interpreter() -> tuple[str, str]:
    """(intérprete real, carpeta de paquetes). Se usa el binario resuelto: el `python` del entorno virtual es
    un enlace que pasa por rutas que no existen dentro del sandbox."""
    import sysconfig
    return str(Path(sys.executable).resolve()), sysconfig.get_paths()["purelib"]


def _interpreter_binds() -> list[str]:
    """Monta en sólo lectura el intérprete y los paquetes del entorno."""
    real, site = interpreter()
    binds: list[str] = []
    seen: set[str] = set()
    for p in (str(Path(real).parent.parent), site, str(Path(sys.prefix))):
        if p not in seen and Path(p).exists() and not p.startswith(("/usr", "/etc")):
            seen.add(p)
            binds += ["--ro-bind", p, p]
    return binds


def build_command(script: Path, workdir: Path, *, allow_network: bool = False,
                  ro_dir: str | None = None) -> list[str]:
    """Comando para ejecutar `script` aislado. Sin bwrap, ejecución directa (sin aislamiento)."""
    real, site = interpreter()
    if not available():
        return [sys.executable, str(script)]
    cmd = [
        BWRAP,
        "--ro-bind", "/usr", "/usr",
        "--ro-bind-try", "/etc/ssl", "/etc/ssl",
        "--ro-bind-try", "/etc/ca-certificates", "/etc/ca-certificates",
        "--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        *_interpreter_binds(),
        "--bind", str(workdir), "/work", "--chdir", "/work",
        "--setenv", "HOME", "/work", "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONPATH", site,  # los paquetes del entorno (numpy, pandas…)
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
        "--die-with-parent", "--new-session",
    ]
    if not allow_network:
        cmd.append("--unshare-net")
    if ro_dir:
        d = str(Path(ro_dir).expanduser().resolve())
        if Path(d).exists():
            cmd += ["--ro-bind", d, "/datos"]
    # Dentro del sandbox la carpeta de trabajo está montada en /work
    return cmd + [real, f"/work/{script.name}"]


def _limits(memory_mb: int, cpu_s: int):
    def apply():
        b = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run_python(code: str, *, timeout_s: int = 20, memory_mb: int = 2048, allow_network: bool = False,
               ro_dir: str | None = None) -> dict:
    """Ejecuta código Python aislado. Devuelve salida, si hubo aislamiento y por qué terminó."""
    with tempfile.TemporaryDirectory(prefix="llmt-sandbox-") as tmp:
        work = Path(tmp)
        script = work / "script.py"
        script.write_text(code, encoding="utf-8")
        cmd = build_command(script, work, allow_network=allow_network, ro_dir=ro_dir)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "PYTHONUNBUFFERED": "1", "LC_ALL": "C.UTF-8"}
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=tmp, env=env,
                               preexec_fn=_limits(memory_mb, timeout_s))
        except subprocess.TimeoutExpired:
            return {"output": "", "error": f"Se superó el límite de {timeout_s} s y se detuvo el proceso.",
                    "sandboxed": available(), "timeout": True}
        # Los archivos que deje el script se pierden con el directorio temporal (es intencionado)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if p.returncode != 0 and "MemoryError" not in err and not out:
            err = err or f"El proceso terminó con código {p.returncode} (¿límite de memoria o de CPU?)"
        return {"output": out, "error": err, "sandboxed": available(), "timeout": False}
