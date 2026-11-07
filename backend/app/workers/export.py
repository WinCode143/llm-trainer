"""Export worker: merges LoRA and/or converts to GGUF. Usage: python -m app.workers.export <job_dir>/config.json"""
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from .common import Emitter, load_config


def merge_lora(base_model: str, adapter_dir: str, out_dir: str, trust_remote_code: bool = False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="auto",
                                                 trust_remote_code=trust_remote_code)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    del model
    torch.cuda.empty_cache()


def find_quantize(llama_dir: Path | None) -> str | None:
    cands = []
    if llama_dir:
        cands += [llama_dir / "build/bin/llama-quantize", llama_dir / "llama-quantize"]
    for c in cands:
        if c.exists():
            return str(c)
    return shutil.which("llama-quantize")


def run_cmd(cmd: list[str]):
    print("[export] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    cfg, job_dir = load_config()
    em = Emitter(job_dir)
    try:
        src = cfg["source_path"]
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        hf_dir = src
        if cfg.get("is_lora"):
            hf_dir = str(out_dir / "merged")
            em.status("Merging LoRA with the base model…")
            em.emit("progress", step=1, max_steps=3)
            merge_lora(cfg["base_model"], src, hf_dir, cfg.get("trust_remote_code", False))
            em.emit("merged", path=hf_dir)

        if cfg.get("gguf"):
            llama_dir = Path(cfg["llama_cpp_dir"]) if cfg.get("llama_cpp_dir") else None
            if not llama_dir or not (llama_dir / "convert_hf_to_gguf.py").exists():
                raise RuntimeError("llama.cpp isn't installed: click “Install llama.cpp” in System "
                                   "or run backend/setup-llamacpp.sh")
            qtype = cfg.get("gguf_quant", "q4_k_m").lower()
            name = cfg.get("name", "model")
            direct = qtype in ("f16", "bf16", "q8_0")
            f16_path = out_dir / f"{name}-{'f16' if not direct else qtype}.gguf"
            em.status("Converting to GGUF…")
            em.emit("progress", step=2, max_steps=3)
            run_cmd([sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), hf_dir, "--outfile", str(f16_path),
                     "--outtype", qtype if direct else "f16"])
            final = f16_path
            if not direct:
                q = find_quantize(llama_dir)
                if not q:
                    raise RuntimeError("The llama-quantize binary wasn't found (build llama.cpp with cmake).")
                final = out_dir / f"{name}-{qtype}.gguf"
                em.status(f"Cuantizando a {qtype}…")
                run_cmd([q, str(f16_path), str(final), qtype.upper()])
                if not cfg.get("keep_f16"):
                    f16_path.unlink(missing_ok=True)
            # Modelfile to import it into Ollama with: ollama create <name> -f Modelfile
            (out_dir / "Modelfile").write_text(f"FROM ./{final.name}\n", encoding="utf-8")
            em.status(f"GGUF ready: {final} ({final.stat().st_size / 1e9:.2f} GB)")
            em.emit("gguf", path=str(final), size_gb=round(final.stat().st_size / 1e9, 2))
        em.emit("progress", step=3, max_steps=3)
        em.emit("done")
    except Exception as e:
        traceback.print_exc()
        em.emit("error", message=str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
