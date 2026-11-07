"""vLLM inference engine.

vLLM lives in its own environment (.venv-vllm), so it's launched as an OpenAI server in a subprocess and the
inference manager talks to it over HTTP. Advantages over transformers: much faster generation and, above
all, batch processing (bulk evaluations and comparisons).

Limitations: it doesn't support bitsandbytes-quantized weights (it uses the model in 16 bits or a LoRA on top) and
takes ~1 min to start.
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Iterator

import httpx

from ..config import ROOT

VLLM_PYTHON = ROOT / "backend" / ".venv-vllm" / "bin" / "python"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}/v1"


def available() -> bool:
    return VLLM_PYTHON.exists()


def kill_orphans() -> int:
    """Kills this app's vLLM servers that were left running (e.g. after restarting the backend)."""
    import psutil
    killed = 0
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info.get("cmdline") or []
        if "vllm.entrypoints.openai.api_server" in " ".join(cmd) and str(PORT) in cmd:
            try:
                p.terminate()
                killed += 1
            except psutil.Error:
                pass
    if killed:
        psutil.wait_procs([p for p in psutil.process_iter() if p.pid], timeout=20)
        print(f"[vllm] stopped {killed} orphaned server(s)")
    return killed


class VllmServer:
    """vLLM server in a subprocess. `served_name` is the name the model is requested by."""

    def __init__(self, model: str, lora_path: str | None = None, max_model_len: int = 8192,
                 gpu_util: float = 0.6, log_path: Path | None = None):
        self.model = model
        self.lora_path = lora_path
        self.served_name = "lora" if lora_path else "base"
        cmd = [str(VLLM_PYTHON), "-m", "vllm.entrypoints.openai.api_server", "--model", model,
               "--port", str(PORT), "--host", "127.0.0.1", "--max-model-len", str(max_model_len),
               "--gpu-memory-utilization", str(gpu_util), "--no-enable-log-requests",
               "--served-model-name", "base"]
        if lora_path:
            cmd += ["--enable-lora", "--lora-modules", f"lora={lora_path}", "--max-lora-rank", "64"]
        kill_orphans()  # the port has to be free
        self.log = open(log_path or "/tmp/vllm-server.log", "ab", buffering=0)  # noqa: SIM115
        # Without the CUDA toolkit (nvcc) FlashInfer can't compile its kernels on the fly: we use FlashAttention
        # and PyTorch's native sampler.
        env = {**os.environ, "VLLM_LOGGING_LEVEL": "WARNING", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
               "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.client = httpx.Client(base_url=BASE, timeout=httpx.Timeout(600, connect=10))

    def wait_ready(self, timeout: float = 600) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited while starting (code {self.proc.returncode}); check /tmp/vllm-server.log")
            try:
                if self.client.get("/models", timeout=3).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        self.stop()
        raise RuntimeError("vLLM didn't start in time (check /tmp/vllm-server.log)")

    def stop(self) -> None:
        try:
            if self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        finally:
            self.client.close()
            self.log.close()

    # ---------------- generation ----------------
    def _payload(self, messages: list[dict], params, stream: bool, template_kwargs: dict | None) -> dict:
        extra = dict(template_kwargs or {})
        if params.enable_thinking is not None:
            extra.setdefault("enable_thinking", params.enable_thinking)
        body = {
            "model": self.served_name, "messages": messages, "stream": stream,
            "max_tokens": params.max_new_tokens, "temperature": params.temperature, "top_p": params.top_p,
            "repetition_penalty": params.repetition_penalty,
        }
        if params.top_k:
            body["top_k"] = params.top_k
        if params.seed is not None:
            body["seed"] = params.seed
        tools = extra.pop("tools", None)
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        if extra:
            body["chat_template_kwargs"] = extra
        return body

    def stream(self, messages: list[dict], params, cancel, template_kwargs: dict | None = None) -> Iterator[str]:
        body = self._payload(messages, params, True, template_kwargs)
        with self.client.stream("POST", "/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"vLLM: {r.read().decode()[:300]}")
            for line in r.iter_lines():
                if cancel.is_set():
                    break
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                # Reasoning models return their thinking in a separate field
                think = (choices[0].get("delta") or {}).get("reasoning_content") if choices else None
                if think:
                    yield think
                if delta:
                    yield delta

    def batch(self, prompts: list[list[dict]], params, template_kwargs: dict | None = None) -> list[str]:
        """Several conversations at once: vLLM batches them and it's much faster than one by one."""
        import concurrent.futures as cf

        def one(msgs):
            body = self._payload(msgs, params, False, template_kwargs)
            r = self.client.post("/chat/completions", json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""

        with cf.ThreadPoolExecutor(max_workers=min(16, len(prompts))) as ex:
            return list(ex.map(one, prompts))
