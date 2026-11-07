"""Inference manager: keeps ONE model loaded in VRAM and generates text (with streaming)."""
import gc
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..workers.common import CHATML_TEMPLATE


@dataclass
class GenParams:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    seed: int | None = None
    # None = whatever the template decides; False/True = turn <think> reasoning off/on (Qwen3 and similar)
    enable_thinking: bool | None = None


@dataclass
class Loaded:
    model_id: int | None
    name: str
    kind: str
    path: str
    base_model: str | None
    quantization: str
    backend: str  # transformers | llama.cpp
    obj: object = field(repr=False, default=None)
    tokenizer: object = field(repr=False, default=None)
    processor: object = field(repr=False, default=None)  # only vision models (VLM) in transformers
    # GGUFs have no processor: their vision comes from an mmproj loaded separately
    vision_gguf: bool = False
    n_ctx: int = 0            # the one that could be used, which may be smaller than requested
    loaded_at: float = field(default_factory=time.time)

    def info(self) -> dict:
        return {"model_id": self.model_id, "name": self.name, "kind": self.kind, "path": self.path,
                "base_model": self.base_model, "quantization": self.quantization, "backend": self.backend,
                "vision": self.processor is not None or self.vision_gguf, "n_ctx": self.n_ctx,
                "loaded_at": self.loaded_at}


def _vram_free_gb() -> tuple[float, float] | None:
    """(free, total) in GB according to nvidia-smi, or None if it can't be queried."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        free, total = (float(x) / 1024 for x in out.split(","))
        return free, total
    except Exception:  # noqa: BLE001
        return None


def _why_it_doesnt_fit(path: str, n_ctx: int, err: Exception, free_before: float | None) -> str:
    """Turns llama.cpp's “Failed to create llama_context” into something actionable.

    Free VRAM has to be measured BEFORE trying to load: measured at failure time, the weights are
    already in and the message blames “other applications” for using it, when it's actually
    us loading that very model."""
    from pathlib import Path as _P
    weights = _P(path).stat().st_size / 1e9 if _P(path).exists() else 0
    parts = [f"Couldn't load {_P(path).name} ({weights:.1f} GB) with {n_ctx} of context."]
    if free_before is not None:
        parts.append(f"Before trying there were {free_before:.1f} GB free.")
        if free_before < weights + 1:
            parts.append("Not even the weights fit: close whatever is using the GPU or pick a smaller model.")
        else:
            spare = free_before - weights
            parts.append(f"The weights fit ({spare:.1f} GB of headroom) but their context cache doesn't: "
                          f"lower the context, which is now {n_ctx}.")
    parts.append(f"({err})")
    return " ".join(parts)


def _for_llamacpp(messages: list[dict]) -> list[dict]:
    """Translates our multimodal format into the one llama.cpp expects.

    We use {"type": "image", "url": …} and llama.cpp uses {"type": "image_url", "image_url": {"url": …}};
    untranslated, the image travels but the model doesn't see it and answers that it can't see it."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [{"type": "image_url", "image_url": {"url": p.get("url")}} if p.get("type") == "image" else p
                 for p in c if isinstance(p, dict)]
        out.append({**m, "content": c})
    return out


def _text_only(messages: list[dict]) -> list[dict]:
    """Flattens multimodal content to text (for models without vision)."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
            imgs = sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image")
            c = " ".join(parts) + (f"\n[{imgs} image(s) omitted: the loaded model doesn't support vision]" if imgs else "")
        out.append({**m, "content": c})
    return out


class InferenceManager:
    def __init__(self):
        self.current: Loaded | None = None
        # Engine chosen by the user: automatic loads reuse it (evaluation, agents, API)
        self.preferred_engine = "auto"
        # Lock (not RLock): streaming may release it from another thread of the threadpool
        self.lock = threading.Lock()
        self.loading: str | None = None

    # ---------------- loading ----------------
    def load(self, *, model_id: int | None, name: str, kind: str, path: str, base_model: str | None,
             quantization: str = "none", n_ctx: int = 8192, engine: str = "auto", gpu_util: float = 0.6) -> dict:
        """engine: auto (transformers, or llama.cpp for GGUF) | vllm (a separate server, faster and batched)."""
        with self.lock:
            # Is that model already loaded with the same engine? (with vLLM quantization doesn't apply: always 16-bit)
            cur = self.current
            if cur and cur.path == path and (engine == "vllm") == (cur.backend == "vllm") \
                    and (engine == "vllm" or cur.quantization == quantization):
                return cur.info()
            self._unload()
            # The RAG's embedding models and reranker stay cached in VRAM between
            # searches. That's ~3 GB which, with a 17 GB model, is the difference between loading and
            # not loading, so they're released here: they come back by themselves on the next search.
            try:
                from . import rag as _rag
                _rag.unload_models()
            except Exception as e:  # noqa: BLE001
                print("[inference] while releasing the RAG models:", e)
            self.loading = name
            self.preferred_engine = engine
            try:
                if engine == "vllm":
                    self.current = self._load_vllm(model_id, name, kind, path, base_model, n_ctx, gpu_util)
                elif kind == "gguf":
                    self.current = self._load_gguf(model_id, name, path, n_ctx)
                else:
                    self.current = self._load_hf(model_id, name, kind, path, base_model, quantization)
            finally:
                self.loading = None
            return self.current.info()

    def _load_vllm(self, model_id, name, kind, path, base_model, n_ctx, gpu_util) -> Loaded:
        from . import vllm_engine
        if not vllm_engine.available():
            raise RuntimeError("vLLM isn't installed: install it from the System tab")
        if kind == "gguf":
            raise RuntimeError("vLLM doesn't load GGUF models: use the automatic engine (llama.cpp)")
        weights = base_model if kind == "lora" else path
        if kind == "lora" and not weights:
            meta = Path(path) / "llmtrainer.json"
            if meta.exists():
                weights = json.loads(meta.read_text())["base_model"]
        server = vllm_engine.VllmServer(weights, lora_path=path if kind == "lora" else None,
                                        max_model_len=n_ctx, gpu_util=gpu_util)
        server.wait_ready()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(weights)  # only for counting tokens and auxiliary templates
        return Loaded(model_id, name, kind, path, base_model, "16-bit", "vllm", server, tok)

    def _load_hf(self, model_id, name, kind, path, base_model, quantization) -> Loaded:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        qconf = None
        if quantization == "4bit":
            qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        elif quantization == "8bit":
            qconf = BitsAndBytesConfig(load_in_8bit=True)
        weights = base_model if kind == "lora" else path
        # A vision model (VLM)? Detected from the config
        try:
            conf = AutoConfig.from_pretrained(weights)
            is_vlm = hasattr(conf, "vision_config") or any("VL" in a or "Vision" in a or "Idefics" in a or "Gemma3For" in a
                                                           for a in (getattr(conf, "architectures", None) or []))
        except Exception as e:  # noqa: BLE001
            # If the config can't be read, the model loads as text-only and
            # images are dropped without warning. At least say so.
            print(f"[inference] couldn't read the config of {path} ({e}); assuming no vision")
            is_vlm = False
        if kind == "lora" and not weights:
            meta = Path(path) / "llmtrainer.json"
            if meta.exists():
                weights = json.loads(meta.read_text())["base_model"]
        processor = None
        if is_vlm:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(weights, dtype=dtype, quantization_config=qconf, device_map={"": 0})
            processor = AutoProcessor.from_pretrained(weights)
        else:
            model = AutoModelForCausalLM.from_pretrained(weights, dtype=dtype, quantization_config=qconf, device_map={"": 0})
        tok_src = path if (Path(path) / "tokenizer_config.json").exists() else weights
        tok = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(tok_src)
        if kind == "lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, path)
        model.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if not tok.chat_template:
            tok.chat_template = CHATML_TEMPLATE
        return Loaded(model_id, name, kind, path, base_model, quantization, "transformers", model, tok, processor)

    # A GGUF only sees images if its vision projector is loaded separately; it comes in an
    # mmproj file next to the model. The handler depends on the model family.
    MMPROJ_HANDLERS = {"gemma-4": "Gemma4ChatHandler", "gemma4": "Gemma4ChatHandler",
                       "qwen2.5-vl": "Qwen25VLChatHandler", "qwen25vl": "Qwen25VLChatHandler",
                       "minicpm": "MiniCPMv26ChatHandler", "llava": "Llava16ChatHandler"}

    def _vision_handler(self, path: str):
        """Looks for an mmproj next to the model and returns the right handler, or None."""
        from pathlib import Path as _P
        folder = _P(path).parent
        mm = next((p for p in sorted(folder.glob("mmproj*.gguf")) if "F32" not in p.name), None)
        if not mm:
            return None, None
        key = next((k for k in self.MMPROJ_HANDLERS if k in _P(path).name.lower()), None)
        if not key:
            print(f"[gguf] there's {mm.name} but I don't know which handler to use for {_P(path).name}")
            return None, None
        try:
            from llama_cpp import llama_chat_format
            cls = getattr(llama_chat_format, self.MMPROJ_HANDLERS[key])
            return cls(clip_model_path=str(mm), verbose=False), mm.name
        except Exception as e:  # noqa: BLE001
            print(f"[gguf] couldn't enable vision with {mm.name}: {e}")
            return None, None

    def _load_gguf(self, model_id, name, path, n_ctx) -> Loaded:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python isn't installed. Run: uv sync --extra gguf") from e
        handler, mm = self._vision_handler(path)
        if mm:
            print(f"[gguf] vision enabled with {mm}")
        vram = _vram_free_gb()
        free_before = vram[0] if vram else None
        extra = {"chat_handler": handler} if handler else {}
        # If the weights fit but their context cache doesn't, it retries with less context instead of
        # loading nothing: a model with 4k beats no model. It reports what it ended up with.
        attempts = [n_ctx] + [c for c in (8192, 4096, 2048) if c < n_ctx]
        last_err: Exception | None = None
        for i, ctx in enumerate(attempts):
            try:
                llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=ctx, verbose=False, **extra)
                if i:
                    print(f"[gguf] {Path(path).name} loaded with {ctx} of context instead of {n_ctx}: the cache didn't fit")
                ctx_used = ctx
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        else:
            raise RuntimeError(_why_it_doesnt_fit(path, n_ctx, last_err, free_before)) from last_err
        return Loaded(model_id, name, "gguf", path, None, "gguf", "llama.cpp", llm, None,
                      vision_gguf=bool(handler), n_ctx=ctx_used)

    def unload(self) -> None:
        with self.lock:
            self._unload()

    def _unload(self) -> None:
        if not self.current:
            return
        obj = self.current.obj
        if self.current.backend == "vllm" and obj is not None:
            obj.stop()
        elif self.current.backend == "llama.cpp" and obj is not None:
            # A GGUF's VRAM is reserved by the C++ library (Vulkan), not torch: empty_cache() doesn't
            # touch it, and waiting for the garbage collector leaves the previous model taking memory while the
            # next one loads. It has to be freed by hand.
            try:
                obj.close()
            except Exception as e:  # noqa: BLE001
                print("[inference] while closing the GGUF model:", e)
        self.current.obj = None
        self.current.tokenizer = None
        self.current.processor = None
        self.current = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def status(self) -> dict:
        return {"loaded": self.current.info() if self.current else None, "loading": self.loading,
                "preferred_engine": self.preferred_engine}

    # ---------------- generation ----------------
    def stream_chat(self, messages: list[dict], params: GenParams, cancel: threading.Event | None = None,
                    template_kwargs: dict | None = None) -> Iterator[str]:
        """template_kwargs is passed to the chat template (e.g. tools=[…] for agents)."""
        cancel = cancel or threading.Event()
        with self.lock:
            cur = self.current
            if not cur:
                raise RuntimeError("No model is loaded. Load one in the Models tab or in the Playground.")
            if cur.backend == "vllm":
                yield from cur.obj.stream(messages, params, cancel, template_kwargs or {})
            elif cur.backend == "llama.cpp":
                yield from self._stream_gguf(cur, messages, params, cancel)
            else:
                yield from self._stream_hf(cur, messages, params, cancel, template_kwargs or {})

    def batch_chat(self, prompts: list[list[dict]], params: GenParams) -> list[str] | None:
        """Generates several conversations at once (vLLM only). None if the engine doesn't support it."""
        with self.lock:
            cur = self.current
            if not cur or cur.backend != "vllm":
                return None
            return cur.obj.batch(prompts, params)

    def chat(self, messages: list[dict], params: GenParams) -> dict:
        t0 = time.time()
        text = "".join(self.stream_chat(messages, params))
        return {"text": text, "latency_s": round(time.time() - t0, 3)}

    def _stream_hf(self, cur: Loaded, messages, params: GenParams, cancel, template_kwargs: dict | None = None) -> Iterator[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        tok, model = cur.tokenizer, cur.obj
        # Templates ignore variables they don't use, so enable_thinking is harmless on models without reasoning
        extra = {"enable_thinking": params.enable_thinking} if params.enable_thinking is not None else {}
        extra.update(template_kwargs or {})
        if cur.processor is not None:
            # VLM: the processor resolves the images (paths, URLs or data:) declared in the messages
            enc = cur.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_tensors="pt", return_dict=True, **extra)
        else:
            enc = tok.apply_chat_template(_text_only(messages), add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, **extra)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=300)

        class Cancel(StoppingCriteria):
            def __call__(self, *a, **k):
                return cancel.is_set()

        if params.seed is not None:
            torch.manual_seed(params.seed)
        do_sample = params.temperature > 0
        kwargs = dict(**enc, streamer=streamer, max_new_tokens=params.max_new_tokens, do_sample=do_sample,
                      repetition_penalty=params.repetition_penalty, pad_token_id=tok.pad_token_id,
                      stopping_criteria=StoppingCriteriaList([Cancel()]))
        if do_sample:
            kwargs.update(temperature=params.temperature, top_p=params.top_p, top_k=params.top_k)
        err: list[Exception] = []

        def run():
            try:
                with torch.inference_mode():
                    model.generate(**kwargs)
            except Exception as e:  # noqa: BLE001
                err.append(e)
                streamer.end()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            cancel.set()
            th.join()
        if err:
            raise err[0]

    def _stream_gguf(self, cur: Loaded, messages, params: GenParams, cancel) -> Iterator[str]:
        # With an mmproj loaded the images are passed; without it they're flattened to text with a warning
        msgs = _for_llamacpp(messages) if cur.vision_gguf else _text_only(messages)
        it = cur.obj.create_chat_completion(messages=msgs, stream=True, max_tokens=params.max_new_tokens,
                                            temperature=params.temperature, top_p=params.top_p, top_k=params.top_k,
                                            repeat_penalty=params.repetition_penalty, seed=params.seed or -1)
        for chunk in it:
            if cancel.is_set():
                break
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta


manager = InferenceManager()
