"""Gestor de inferencia: mantiene UN modelo cargado en VRAM y genera texto (con streaming)."""
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
    # None = lo que decida la plantilla; False/True = desactiva/activa el razonamiento <think> (Qwen3 y similares)
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
    processor: object = field(repr=False, default=None)  # sólo modelos de visión (VLM) en transformers
    # Los GGUF no llevan processor: su visión viene de un mmproj cargado aparte
    vision_gguf: bool = False
    n_ctx: int = 0            # el que se pudo usar, que puede ser menor que el pedido
    loaded_at: float = field(default_factory=time.time)

    def info(self) -> dict:
        return {"model_id": self.model_id, "name": self.name, "kind": self.kind, "path": self.path,
                "base_model": self.base_model, "quantization": self.quantization, "backend": self.backend,
                "vision": self.processor is not None or self.vision_gguf, "n_ctx": self.n_ctx,
                "loaded_at": self.loaded_at}


def _vram_libre_gb() -> tuple[float, float] | None:
    """(libre, total) en GB según nvidia-smi, o None si no se puede consultar."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        libre, total = (float(x) / 1024 for x in out.split(","))
        return libre, total
    except Exception:  # noqa: BLE001
        return None


def _por_que_no_cabe(path: str, n_ctx: int, err: Exception, libre_antes: float | None) -> str:
    """Traduce el «Failed to create llama_context» de llama.cpp a algo accionable.

    La VRAM libre hay que medirla ANTES de intentar la carga: si se mide al fallar, los pesos ya
    están dentro y el mensaje acusa de ocuparla a «otras aplicaciones» que en realidad somos
    nosotros mismos cargando ese modelo."""
    from pathlib import Path as _P
    pesos = _P(path).stat().st_size / 1e9 if _P(path).exists() else 0
    partes = [f"No se pudo cargar {_P(path).name} ({pesos:.1f} GB) con {n_ctx} de contexto."]
    if libre_antes is not None:
        partes.append(f"Antes de intentarlo quedaban {libre_antes:.1f} GB libres.")
        if libre_antes < pesos + 1:
            partes.append("No caben ni los pesos: cierra lo que esté usando la GPU o elige un modelo más pequeño.")
        else:
            sobra = libre_antes - pesos
            partes.append(f"Los pesos caben ({sobra:.1f} GB de margen) pero no su caché de contexto: "
                          f"baja el contexto, que ahora es {n_ctx}.")
    partes.append(f"({err})")
    return " ".join(partes)


def _para_llamacpp(messages: list[dict]) -> list[dict]:
    """Traduce nuestro formato multimodal al que espera llama.cpp.

    Nosotros usamos {"type": "image", "url": …} y llama.cpp {"type": "image_url", "image_url": {"url": …}};
    sin traducir, la imagen viaja pero el modelo no la ve y contesta que no puede verla."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [{"type": "image_url", "image_url": {"url": p.get("url")}} if p.get("type") == "image" else p
                 for p in c if isinstance(p, dict)]
        out.append({**m, "content": c})
    return out


def _text_only(messages: list[dict]) -> list[dict]:
    """Aplana contenidos multimodales a texto (para modelos sin visión)."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
            imgs = sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image")
            c = " ".join(parts) + (f"\n[{imgs} imagen(es) omitida(s): el modelo cargado no admite visión]" if imgs else "")
        out.append({**m, "content": c})
    return out


class InferenceManager:
    def __init__(self):
        self.current: Loaded | None = None
        # Motor elegido por el usuario: lo reutilizan las cargas automáticas (evaluación, agentes, API)
        self.preferred_engine = "auto"
        # Lock (no RLock): el streaming puede liberar desde otro hilo del threadpool
        self.lock = threading.Lock()
        self.loading: str | None = None

    # ---------------- carga ----------------
    def load(self, *, model_id: int | None, name: str, kind: str, path: str, base_model: str | None,
             quantization: str = "none", n_ctx: int = 8192, engine: str = "auto", gpu_util: float = 0.6) -> dict:
        """engine: auto (transformers, o llama.cpp para GGUF) | vllm (servidor aparte, más rápido y por lotes)."""
        with self.lock:
            # ¿Ya está cargado ese modelo con el mismo motor? (con vLLM la cuantización no aplica: siempre 16-bit)
            cur = self.current
            if cur and cur.path == path and (engine == "vllm") == (cur.backend == "vllm") \
                    and (engine == "vllm" or cur.quantization == quantization):
                return cur.info()
            self._unload()
            # Los modelos de embeddings y el reranker del RAG se quedan cacheados en VRAM entre
            # búsquedas. Son ~3 GB que con un modelo de 17 GB marcan la diferencia entre cargar y
            # no cargar, así que se sueltan aquí: vuelven solos en la siguiente búsqueda.
            try:
                from . import rag as _rag
                _rag.unload_models()
            except Exception as e:  # noqa: BLE001
                print("[inference] al liberar los modelos del RAG:", e)
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
            raise RuntimeError("vLLM no está instalado: instálalo desde la pestaña Sistema")
        if kind == "gguf":
            raise RuntimeError("vLLM no carga modelos GGUF: usa el motor automático (llama.cpp)")
        weights = base_model if kind == "lora" else path
        if kind == "lora" and not weights:
            meta = Path(path) / "llmtrainer.json"
            if meta.exists():
                weights = json.loads(meta.read_text())["base_model"]
        server = vllm_engine.VllmServer(weights, lora_path=path if kind == "lora" else None,
                                        max_model_len=n_ctx, gpu_util=gpu_util)
        server.wait_ready()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(weights)  # sólo para contar tokens y plantillas auxiliares
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
        # ¿Modelo de visión (VLM)? Se detecta por la configuración
        try:
            conf = AutoConfig.from_pretrained(weights)
            is_vlm = hasattr(conf, "vision_config") or any("VL" in a or "Vision" in a or "Idefics" in a or "Gemma3For" in a
                                                           for a in (getattr(conf, "architectures", None) or []))
        except Exception as e:  # noqa: BLE001
            # Si la configuración no se puede leer, el modelo se carga como si fuera sólo de texto y
            # las imágenes se descartan sin avisar. Al menos que quede dicho.
            print(f"[inference] no se pudo leer la configuración de {path} ({e}); se asume sin visión")
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

    # Un GGUF sólo ve imágenes si se le carga aparte su proyector de visión, que viene en un
    # archivo mmproj junto al modelo. El manejador depende de la familia.
    MMPROJ_HANDLERS = {"gemma-4": "Gemma4ChatHandler", "gemma4": "Gemma4ChatHandler",
                       "qwen2.5-vl": "Qwen25VLChatHandler", "qwen25vl": "Qwen25VLChatHandler",
                       "minicpm": "MiniCPMv26ChatHandler", "llava": "Llava16ChatHandler"}

    def _vision_handler(self, path: str):
        """Busca un mmproj junto al modelo y devuelve el manejador que toca, o None."""
        from pathlib import Path as _P
        carpeta = _P(path).parent
        mm = next((p for p in sorted(carpeta.glob("mmproj*.gguf")) if "F32" not in p.name), None)
        if not mm:
            return None, None
        clave = next((k for k in self.MMPROJ_HANDLERS if k in _P(path).name.lower()), None)
        if not clave:
            print(f"[gguf] hay {mm.name} pero no sé qué manejador usar para {_P(path).name}")
            return None, None
        try:
            from llama_cpp import llama_chat_format
            cls = getattr(llama_chat_format, self.MMPROJ_HANDLERS[clave])
            return cls(clip_model_path=str(mm), verbose=False), mm.name
        except Exception as e:  # noqa: BLE001
            print(f"[gguf] no se pudo activar la visión con {mm.name}: {e}")
            return None, None

    def _load_gguf(self, model_id, name, path, n_ctx) -> Loaded:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python no está instalado. Ejecuta: uv sync --extra gguf") from e
        handler, mm = self._vision_handler(path)
        if mm:
            print(f"[gguf] visión activada con {mm}")
        vram = _vram_libre_gb()
        libre_antes = vram[0] if vram else None
        extra = {"chat_handler": handler} if handler else {}
        # Si los pesos entran pero su caché de contexto no, se reintenta con menos contexto en vez de
        # no cargar nada: más vale un modelo con 4k que ninguno. Se avisa de con cuánto quedó.
        intentos = [n_ctx] + [c for c in (8192, 4096, 2048) if c < n_ctx]
        ultimo: Exception | None = None
        for i, ctx in enumerate(intentos):
            try:
                llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=ctx, verbose=False, **extra)
                if i:
                    print(f"[gguf] {Path(path).name} cargado con {ctx} de contexto en vez de {n_ctx}: no cabía la caché")
                ctx_usado = ctx
                break
            except Exception as e:  # noqa: BLE001
                ultimo = e
        else:
            raise RuntimeError(_por_que_no_cabe(path, n_ctx, ultimo, libre_antes)) from ultimo
        return Loaded(model_id, name, "gguf", path, None, "gguf", "llama.cpp", llm, None,
                      vision_gguf=bool(handler), n_ctx=ctx_usado)

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
            # La VRAM de un GGUF la reserva la librería C++ (Vulkan), no torch: empty_cache() no la
            # toca y esperar al recolector deja el modelo anterior ocupando memoria mientras se carga
            # el siguiente. Hay que liberarla a mano.
            try:
                obj.close()
            except Exception as e:  # noqa: BLE001
                print("[inference] al cerrar el modelo GGUF:", e)
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

    # ---------------- generación ----------------
    def stream_chat(self, messages: list[dict], params: GenParams, cancel: threading.Event | None = None,
                    template_kwargs: dict | None = None) -> Iterator[str]:
        """template_kwargs se pasa a la plantilla de chat (p. ej. tools=[…] para agentes)."""
        cancel = cancel or threading.Event()
        with self.lock:
            cur = self.current
            if not cur:
                raise RuntimeError("No hay ningún modelo cargado. Carga uno en la pestaña Modelos o en el Playground.")
            if cur.backend == "vllm":
                yield from cur.obj.stream(messages, params, cancel, template_kwargs or {})
            elif cur.backend == "llama.cpp":
                yield from self._stream_gguf(cur, messages, params, cancel)
            else:
                yield from self._stream_hf(cur, messages, params, cancel, template_kwargs or {})

    def batch_chat(self, prompts: list[list[dict]], params: GenParams) -> list[str] | None:
        """Genera varias conversaciones a la vez (sólo vLLM). None si el motor no lo soporta."""
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
        # Las plantillas ignoran variables que no usan, así que enable_thinking es inocuo en modelos sin razonamiento
        extra = {"enable_thinking": params.enable_thinking} if params.enable_thinking is not None else {}
        extra.update(template_kwargs or {})
        if cur.processor is not None:
            # VLM: el processor resuelve las imágenes (rutas, URLs o data:) declaradas en los mensajes
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
        # Con mmproj cargado las imágenes sí se pasan; sin él se aplanan a texto con un aviso
        msgs = _para_llamacpp(messages) if cur.vision_gguf else _text_only(messages)
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
