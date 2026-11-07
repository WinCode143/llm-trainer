"""Sandboxing for the code agents run (the “Run Python” tool).

It uses **bubblewrap** (bwrap): user namespaces, no root and millisecond startup.
Inside the sandbox the process sees:
    - /usr and /etc read-only (what Python needs to start)
    - the interpreter and its packages read-only
    - a temporary work folder, the only writable one
    - optionally a data folder of yours, read-only
It doesn't see your /home, the project, the database or the models. The network is cut off unless you enable it.
Memory, CPU and file-size limits are applied too, plus a maximum time.

If bwrap isn't available it runs without a sandbox (with resource limits) and the interface warns about it.
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
    """Does bwrap work on this machine? (some kernels disable user namespaces)."""
    if not BWRAP:
        return False
    try:
        # The /lib links are essential: without them the dynamic loader doesn't exist inside the sandbox
        p = subprocess.run([BWRAP, "--ro-bind", "/usr", "/usr",
                            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                            "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
                            "--unshare-all", "--die-with-parent", "/usr/bin/true"],
                           capture_output=True, timeout=15)
        return p.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def status() -> dict:
    return {"backend": "bubblewrap" if available() else "none", "available": available(),
            "detail": ("Isolates the file system, network and processes without root."
                       if available() else "Install bubblewrap (the `bubblewrap` package) to sandbox the agents' code.")}


def interpreter() -> tuple[str, str]:
    """(real interpreter, packages folder). The resolved binary is used: the virtualenv's `python` is
    a link that goes through paths that don't exist inside the sandbox."""
    import sysconfig
    return str(Path(sys.executable).resolve()), sysconfig.get_paths()["purelib"]


def _interpreter_binds() -> list[str]:
    """Mounts the environment's interpreter and packages read-only."""
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
    """Command to run `script` sandboxed. Without bwrap, a direct run (no sandbox)."""
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
        "--setenv", "PYTHONPATH", site,  # the environment's packages (numpy, pandas…)
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
        "--die-with-parent", "--new-session",
    ]
    if not allow_network:
        cmd.append("--unshare-net")
    if ro_dir:
        d = str(Path(ro_dir).expanduser().resolve())
        if Path(d).exists():
            cmd += ["--ro-bind", d, "/data"]
    # Inside the sandbox the work folder is mounted at /work
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
    """Runs Python code sandboxed. Returns the output, whether it was sandboxed and why it ended."""
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
            return {"output": "", "error": f"The {timeout_s} s limit was exceeded and the process was stopped.",
                    "sandboxed": available(), "timeout": True}
        # Files the script leaves behind go away with the temp directory (on purpose)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if p.returncode != 0 and "MemoryError" not in err and not out:
            err = err or f"The process ended with code {p.returncode} (memory or CPU limit?)"
        return {"output": out, "error": err, "sandboxed": available(), "timeout": False}
