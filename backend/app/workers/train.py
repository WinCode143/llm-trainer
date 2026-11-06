"""Worker de entrenamiento. Uso: python -m app.workers.train <job_dir>/config.json"""
import json
import math
import os
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .common import CHATML_TEMPLATE, Emitter, StopFlag, filter_kwargs, load_config  # noqa: E402


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to_prompt_completion(row):
    msgs = row["messages"]
    last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    return {"prompt": msgs[:last], "completion": [msgs[last]]}


def main():
    cfg, job_dir = load_config()
    em = Emitter(job_dir)
    stop = StopFlag()
    t0 = time.time()
    try:
        run(cfg, job_dir, em, stop)
        em.emit("done", elapsed=time.time() - t0, stopped=stop.stop)
    except Exception as e:
        traceback.print_exc()
        em.emit("error", message=str(e))
        raise SystemExit(1)


UNSLOTH_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_hf(cfg, em, use_lora, quant, dtype):
    """Ruta estándar: transformers + bitsandbytes + PEFT."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    trc = cfg.get("trust_remote_code", False)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=trc)
    qconf = None
    if quant == "4bit":
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=dtype)
    elif quant == "8bit":
        qconf = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=qconf, dtype=dtype, device_map={"": 0},
        attn_implementation=cfg.get("attn_implementation", "sdpa"), trust_remote_code=trc,
    )
    model.config.use_cache = False
    if use_lora:
        if qconf is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
        lconf = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                           lora_dropout=cfg.get("lora_dropout", 0.05), target_modules=cfg.get("target_modules") or "all-linear",
                           bias="none", task_type="CAUSAL_LM", use_rslora=cfg.get("use_rslora", False),
                           use_dora=cfg.get("use_dora", False))
        model = get_peft_model(model, lconf)
    return model, tok, qconf is not None


def load_unsloth(cfg, em, use_lora, quant):
    """Ruta Unsloth: kernels Triton propios, ~2x más rápido y menos VRAM."""
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"], max_seq_length=cfg.get("max_seq_len", 2048), dtype=None,
        load_in_4bit=quant == "4bit", load_in_8bit=quant == "8bit", full_finetuning=not use_lora,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    # Algunos modelos devuelven un processor multimodal; nos quedamos con el tokenizer de texto
    tok = getattr(tok, "tokenizer", tok)
    if use_lora:
        tm = cfg.get("target_modules")
        model = FastLanguageModel.get_peft_model(
            model, r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.0), target_modules=tm if isinstance(tm, list) else UNSLOTH_TARGETS,
            bias="none", use_gradient_checkpointing="unsloth" if cfg.get("gradient_checkpointing", True) else False,
            random_state=cfg.get("seed", 42), use_rslora=cfg.get("use_rslora", False),
        )
    return model, tok, quant in ("4bit", "8bit")


def method_of(cfg) -> str:
    return cfg.get("method", "sft")


def run(cfg, job_dir, em, stop):
    backend = cfg.get("backend", "hf")
    if backend == "vllm" and method_of(cfg) != "grpo":
        backend = "hf"  # vLLM sólo acelera la generación de GRPO
    if backend == "vllm" and cfg.get("quantization", "4bit") != "none" and cfg.get("use_lora", True):
        # vLLM no puede generar con pesos cuantizados por bitsandbytes: LoRA en 16 bits
        cfg = {**cfg, "quantization": "none"}
    if backend == "unsloth":
        import unsloth  # noqa: F401  — debe importarse antes que transformers/trl para aplicar sus parches
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback

    method = cfg.get("method", "sft")  # sft | dpo | pretrain
    out_dir = cfg["output_dir"]
    use_lora = cfg.get("use_lora", True)
    quant = cfg.get("quantization", "4bit") if use_lora else "none"
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    # ---------- Modelo ----------
    em.status(f"Cargando {cfg['base_model']} con {backend} ({quant}, {'LoRA' if use_lora else 'full fine-tune'})")
    if backend == "unsloth":
        model, tok, quantized = load_unsloth(cfg, em, use_lora, quant)
    else:
        model, tok, quantized = load_hf(cfg, em, use_lora, quant, dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not tok.chat_template and method != "pretrain":
        em.status("El tokenizer no tiene chat template; usando ChatML")
        tok.chat_template = CHATML_TEMPLATE
    if use_lora:
        trainable, total = model.get_nb_trainable_parameters()
        em.status(f"Parámetros entrenables: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        em.emit("params", trainable=trainable, total=total)
    # En unsloth el gradient checkpointing ya lo gestiona get_peft_model
    gc = cfg.get("gradient_checkpointing", True) and backend != "unsloth"

    # ---------- Dataset ----------
    rows = load_rows(cfg["dataset_path"])
    fmt = cfg["dataset_format"]
    if method in ("dpo", "orpo", "kto") and fmt != "preference":
        raise ValueError(f"{method.upper()} necesita un dataset de preferencias (prompt/chosen/rejected)")
    if method == "kto":
        # KTO usa ejemplos sueltos etiquetados como deseables/no deseables
        rows = [x for r in rows for x in ({"prompt": r["prompt"], "completion": r["chosen"], "label": True},
                                          {"prompt": r["prompt"], "completion": r["rejected"], "label": False})]
    if method == "grpo":
        if fmt == "preference":
            rows = [{"prompt": [{"role": "user", "content": r["prompt"]}], "answer": r["chosen"]} for r in rows]
        elif fmt == "chat":
            pc = [to_prompt_completion(r) for r in rows]
            rows = [{"prompt": x["prompt"], "answer": x["completion"][0]["content"]} for x in pc]
        else:
            raise ValueError("GRPO necesita prompts con respuesta de referencia (dataset de chat o preferencias)")
        if cfg.get("grpo_system_prompt"):
            for r in rows:
                if r["prompt"][0]["role"] != "system":
                    r["prompt"] = [{"role": "system", "content": cfg["grpo_system_prompt"]}] + r["prompt"]
    if method == "sft" and fmt == "preference":
        rows = [{"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["chosen"]}]} for r in rows]
        fmt = "chat"
    if method == "sft" and fmt == "chat" and cfg.get("train_on_responses_only", True):
        rows = [to_prompt_completion(r) for r in rows]
    if method == "pretrain" and fmt == "chat":
        rows = [{"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows]
    if cfg.get("max_samples"):
        rows = rows[: cfg["max_samples"]]
    ds = Dataset.from_list(rows).shuffle(seed=cfg.get("seed", 42))
    eval_ds = None
    ratio = cfg.get("eval_ratio", 0.0)
    if ratio and len(ds) > 10:
        sp = ds.train_test_split(test_size=ratio, seed=cfg.get("seed", 42))
        ds, eval_ds = sp["train"], sp["test"]
    em.status(f"Dataset: {len(ds)} ejemplos de entrenamiento" + (f", {len(eval_ds)} de evaluación" if eval_ds else ""))

    # ---------- Callbacks ----------
    class LiveCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            em.emit("progress", step=0, max_steps=state.max_steps, epoch=0)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                clean = {k: (v if not (isinstance(v, float) and math.isnan(v)) else None) for k, v in logs.items()
                         if isinstance(v, (int, float))}
                em.emit("log", step=state.global_step, **clean)
                if torch.cuda.is_available():
                    em.emit("gpu", step=state.global_step, mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))

        def on_step_end(self, args, state, control, **kw):
            em.emit("progress", step=state.global_step, max_steps=state.max_steps, epoch=state.epoch)
            if stop.stop:
                control.should_training_stop = True
                control.should_save = True
            return control

    common = dict(
        output_dir=out_dir,
        num_train_epochs=cfg.get("epochs", 1),
        max_steps=cfg.get("max_steps") or -1,
        per_device_train_batch_size=cfg.get("batch_size", 2),
        per_device_eval_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        # transformers>=5: warmup_steps admite float <1 como proporción (warmup_ratio fue retirado)
        warmup_steps=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        optim=cfg.get("optim", "paged_adamw_8bit" if quantized else "adamw_torch"),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 2),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", 100) if eval_ds is not None else None,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg.get("seed", 42),
        report_to=[],
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        neftune_noise_alpha=cfg.get("neftune_alpha") or None,
    )

    from transformers import TrainingArguments
    from dataclasses import fields as _fields
    if "warmup_ratio" in {f.name for f in _fields(TrainingArguments)}:
        common.pop("warmup_steps")

    if method == "grpo":
        from trl import GRPOConfig, GRPOTrainer

        from .rewards import build as build_rewards
        ng = cfg.get("num_generations", 4)
        # El lote efectivo debe ser múltiplo del nº de generaciones por prompt
        common["per_device_train_batch_size"] = max(common["per_device_train_batch_size"], ng)
        args = GRPOConfig(**filter_kwargs(GRPOConfig, {
            **common, "num_generations": ng, "max_completion_length": cfg.get("max_completion_length", 256),
            "max_prompt_length": cfg.get("max_seq_len", 1024), "beta": cfg.get("grpo_beta", 0.0),
            "temperature": cfg.get("grpo_temperature", 0.9), "loss_type": cfg.get("grpo_loss", "dapo"),
            "generation_batch_size": common["per_device_train_batch_size"] * common["gradient_accumulation_steps"],
            "log_completions": False,
            # vLLM en modo «colocate»: comparte GPU con el entrenamiento y genera mucho más rápido
            "use_vllm": backend == "vllm", "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": cfg.get("vllm_gpu_util", 0.25),
            # Limitamos el contexto de vLLM a lo que realmente usamos: si no, reserva caché para el
            # contexto máximo del modelo (p. ej. 40k tokens) y no cabe
            "vllm_max_model_length": cfg.get("max_seq_len", 1024) + cfg.get("max_completion_length", 256),
        }))
        if backend == "vllm":
            em.status(f"Generación con vLLM (colocate, {cfg.get('vllm_gpu_util', 0.25) * 100:.0f}% de la VRAM)")
        trainer = GRPOTrainer(model=model, reward_funcs=build_rewards(cfg), args=args, train_dataset=ds,
                              processing_class=tok, callbacks=[LiveCallback()])
    elif method == "orpo":
        try:
            from trl import ORPOConfig, ORPOTrainer
        except ImportError:
            from trl.experimental.orpo import ORPOConfig, ORPOTrainer
        args = ORPOConfig(**filter_kwargs(ORPOConfig, {
            **common, "beta": cfg.get("orpo_beta", 0.1), "max_length": cfg.get("max_seq_len", 2048),
            "max_prompt_length": cfg.get("max_seq_len", 2048) // 2}))
        trainer = ORPOTrainer(model=model, args=args, train_dataset=ds, eval_dataset=eval_ds, processing_class=tok,
                              callbacks=[LiveCallback()])
    elif method == "kto":
        from trl import KTOConfig, KTOTrainer
        args = KTOConfig(**filter_kwargs(KTOConfig, {
            **common, "beta": cfg.get("dpo_beta", 0.1), "max_length": cfg.get("max_seq_len", 2048),
            "max_prompt_length": cfg.get("max_seq_len", 2048) // 2}))
        trainer = KTOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, eval_dataset=eval_ds,
                             processing_class=tok, callbacks=[LiveCallback()])
    elif method == "dpo":
        from trl import DPOConfig, DPOTrainer

        args = DPOConfig(**filter_kwargs(DPOConfig, {
            **common, "beta": cfg.get("dpo_beta", 0.1), "max_length": cfg.get("max_seq_len", 2048),
            "max_prompt_length": cfg.get("max_seq_len", 2048) // 2, "loss_type": cfg.get("dpo_loss", "sigmoid"),
        }))
        trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, eval_dataset=eval_ds,
                             processing_class=tok, callbacks=[LiveCallback()])
    else:
        from trl import SFTConfig, SFTTrainer

        extra = {"max_length": cfg.get("max_seq_len", 2048), "max_seq_length": cfg.get("max_seq_len", 2048),
                 "packing": cfg.get("packing", False)}
        if method == "pretrain" or fmt == "text":
            extra["dataset_text_field"] = "text"
        args = SFTConfig(**filter_kwargs(SFTConfig, {**common, **extra}))
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds, eval_dataset=eval_ds, processing_class=tok,
                             callbacks=[LiveCallback()])

    em.status("Entrenando…")
    resume = cfg.get("resume_from_checkpoint") or None
    result = trainer.train(resume_from_checkpoint=resume)
    em.emit("train_result", **{k: v for k, v in result.metrics.items() if isinstance(v, (int, float))})

    final_dir = os.path.join(out_dir, "final")
    em.status(f"Guardando en {final_dir}")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "llmtrainer.json"), "w") as f:
        json.dump({"base_model": cfg["base_model"], "use_lora": use_lora, "method": method, "backend": backend}, f)
    em.emit("saved", path=final_dir, use_lora=use_lora)

    if cfg.get("merge_after") and use_lora:
        del trainer, model
        torch.cuda.empty_cache()
        from .export import merge_lora

        merged = os.path.join(out_dir, "merged")
        em.status("Fusionando LoRA con el modelo base…")
        merge_lora(cfg["base_model"], final_dir, merged, trust_remote_code=cfg.get("trust_remote_code", False))
        em.emit("merged", path=merged)


if __name__ == "__main__":
    main()
