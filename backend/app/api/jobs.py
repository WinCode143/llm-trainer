import asyncio
import json
import shutil

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..db import Dataset, Job, Model, engine, get_session, now
from ..services import jobs as svc
from ..services.datasets import slugify

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class TrainConfig(BaseModel):
    name: str
    base_model: str = Field(description="HF repo or local path, or the id of a registered model")
    base_model_id: int | None = None
    dataset_id: int
    method: str = "sft"  # sft | dpo | orpo | kto | grpo | pretrain
    backend: str = "hf"  # hf | unsloth | vllm (GRPO only)
    vllm_gpu_util: float = 0.25
    use_lora: bool = True
    quantization: str = "4bit"  # 4bit | 8bit | none
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | str | None = None
    use_rslora: bool = False
    use_dora: bool = False
    epochs: float = 1
    max_steps: int | None = None
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    optim: str | None = None
    max_seq_len: int = 2048
    packing: bool = False
    train_on_responses_only: bool = True
    gradient_checkpointing: bool = True
    eval_ratio: float = 0.0
    logging_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 100
    max_samples: int | None = None
    neftune_alpha: float | None = None
    dpo_beta: float = 0.1
    dpo_loss: str = "sigmoid"
    orpo_beta: float = 0.1
    # GRPO
    rewards: list[str] = ["correctness"]  # correctness | format | length | regex
    reward_regex: str | None = None
    reward_length_target: int = 80
    num_generations: int = 4
    max_completion_length: int = 256
    grpo_beta: float = 0.0
    grpo_temperature: float = 0.9
    grpo_system_prompt: str | None = None
    seed: int = 42
    merge_after: bool = False
    trust_remote_code: bool = False


# Only the config keys the list needs to show travel: the rest is heavy and unused
LIST_CONFIG_KEYS = ("method", "backend", "base_model", "gguf", "gguf_quant", "model_name", "dataset_name")


@router.get("")
def list_jobs(limit: int = 200, offset: int = 0, full: bool = False, s: Session = Depends(get_session)):
    """History, from newest to oldest. `full=true` returns each job's whole config."""
    q = select(Job).order_by(Job.id.desc()).offset(offset).limit(min(limit, 1000))
    out = []
    for j in s.exec(q).all():
        d = j.model_dump(mode="json")
        if not full:
            d["config"] = {k: v for k, v in j.config.items() if k in LIST_CONFIG_KEYS}
        d["progress"] = svc.last_progress(j.id) if j.status == "running" else None
        out.append(d)
    return out


@router.post("/train")
def create_train(cfg: TrainConfig, s: Session = Depends(get_session)):
    ds = s.get(Dataset, cfg.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    base = cfg.base_model
    if cfg.base_model_id:
        m = s.get(Model, cfg.base_model_id)
        if not m or m.kind not in ("hf", "merged"):
            raise HTTPException(400, "The base model must be an HF or merged model")
        base = m.path
    if cfg.backend == "unsloth" and not svc.unsloth_available():
        raise HTTPException(400, "The Unsloth engine isn't installed: install it from System")
    if cfg.backend == "vllm" and not svc.vllm_available():
        raise HTTPException(400, "The vLLM engine isn't installed: install it from System")
    if cfg.backend == "vllm" and cfg.method != "grpo":
        raise HTTPException(400, "vLLM only speeds up GRPO; use Unsloth or HF for the other methods")
    data = cfg.model_dump(exclude_none=True)
    if data.get("optim") is None:
        data.pop("optim", None)
    out = settings.outputs_dir / f"{slugify(cfg.name)}"
    i = 2
    while out.exists():
        out = settings.outputs_dir / f"{slugify(cfg.name)}-{i}"
        i += 1
    # A copy of the dataset as it is now: if you edit it later, the job stays reproducible
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / f"dataset-{slugify(ds.name)}.jsonl"
    shutil.copy2(ds.path, snapshot)
    data.update(base_model=base, dataset_path=str(snapshot), dataset_name=ds.name, dataset_rows=ds.num_rows,
                dataset_format=ds.format, output_dir=str(out))
    return svc.enqueue(cfg.name, "train", data)


class SweepBody(BaseModel):
    base: TrainConfig
    # {"learning_rate": [1e-4, 2e-4], "lora_r": [8, 16, 32]}
    grid: dict[str, list]
    strategy: str = "grid"  # grid | random
    max_runs: int = 8
    seed: int = 42


@router.post("/sweep")
def create_sweep(body: SweepBody, s: Session = Depends(get_session)):
    """Launches a sweep: one hyperparameter combination per job, queued."""
    import itertools
    import random
    import uuid

    keys = [k for k, v in body.grid.items() if v]
    if not keys:
        raise HTTPException(400, "Define at least one parameter with values")
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(body.grid[k] for k in keys))]
    if body.strategy == "random" and len(combos) > body.max_runs:
        combos = random.Random(body.seed).sample(combos, body.max_runs)
    combos = combos[: body.max_runs]
    sweep_id = uuid.uuid4().hex[:8]
    jobs = []
    for i, combo in enumerate(combos, 1):
        cfg = body.base.model_copy(update=combo)
        cfg.name = f"{body.base.name or 'sweep'}-{i}"
        job = create_train(cfg, s)
        with Session(engine) as s2:  # mark the job as part of the sweep
            j = s2.get(Job, job.id)
            j.config = {**j.config, "sweep_id": sweep_id, "sweep_params": combo}
            s2.add(j)
            s2.commit()
        jobs.append(job.id)
    return {"sweep_id": sweep_id, "runs": len(jobs), "job_ids": jobs, "combos": combos}


@router.get("/sweeps")
def list_sweeps(s: Session = Depends(get_session)):
    """Sweeps with the result of each combination (sorted by the chosen metric)."""
    out: dict[str, dict] = {}
    for j in s.exec(select(Job).order_by(Job.id)).all():
        sid = j.config.get("sweep_id")
        if not sid:
            continue
        sw = out.setdefault(sid, {"sweep_id": sid, "created_at": j.created_at, "params": list(j.config.get("sweep_params", {})), "runs": []})
        final = j.result.get("final", {})
        sw["runs"].append({"job_id": j.id, "name": j.name, "status": j.status, "params": j.config.get("sweep_params", {}),
                           "eval_loss": final.get("eval_loss"), "loss": final.get("loss"), "reward": final.get("reward"),
                           "accuracy": final.get("mean_token_accuracy"),
                           "train_loss": j.result.get("metrics", {}).get("train_loss"),
                           "runtime": j.result.get("metrics", {}).get("train_runtime")})
    return sorted(out.values(), key=lambda x: x["created_at"], reverse=True)


class ExportBody(BaseModel):
    model_id: int
    merge: bool = True
    gguf: bool = False
    gguf_quant: str = "q4_k_m"


@router.post("/export")
def create_export(body: ExportBody, s: Session = Depends(get_session)):
    m = s.get(Model, body.model_id)
    if not m:
        raise HTTPException(404, "Model not found")
    if m.kind == "gguf":
        raise HTTPException(400, "The model is already a GGUF")
    is_lora = m.kind == "lora"
    if not is_lora and not body.gguf:
        raise HTTPException(400, "Nothing to do: the model is already merged; turn on GGUF")
    out = settings.outputs_dir / f"{slugify(m.name)}-export"
    cfg = {"name": slugify(m.name), "source_path": m.path, "is_lora": is_lora, "base_model": m.base_model,
           "output_dir": str(out), "gguf": body.gguf, "gguf_quant": body.gguf_quant,
           "llama_cpp_dir": str(settings.llama_cpp) if settings.llama_cpp else None}
    return svc.enqueue(f"{m.name}-export", "export", cfg)


BENCH_TASKS = [
    {"task": "arc_easy", "label": "ARC-Easy", "desc": "Grade-school science (English)"},
    {"task": "hellaswag", "label": "HellaSwag", "desc": "Common sense (English)"},
    {"task": "mmlu", "label": "MMLU", "desc": "General knowledge, 57 subjects (English)"},
    {"task": "gsm8k", "label": "GSM8K", "desc": "Step-by-step math (English, slow)"},
    {"task": "truthfulqa_mc2", "label": "TruthfulQA", "desc": "Truthfulness (English)"},
    {"task": "arc_es", "label": "ARC (es)", "desc": "Grade-school science in Spanish"},
    {"task": "hellaswag_es", "label": "HellaSwag (es)", "desc": "Common sense in Spanish"},
    {"task": "global_mmlu_es", "label": "Global-MMLU (es)", "desc": "General knowledge in Spanish"},
    {"task": "belebele_spa_Latn", "label": "Belebele (es)", "desc": "Reading comprehension in Spanish"},
]


@router.get("/bench-tasks")
def bench_tasks():
    return BENCH_TASKS


class BenchBody(BaseModel):
    model_id: int
    tasks: list[str]
    limit: int | None = 200
    num_fewshot: int | None = None


@router.post("/bench")
def create_bench(body: BenchBody, s: Session = Depends(get_session)):
    """Standard benchmarks (lm-evaluation-harness) on an HF, merged or LoRA model."""
    m = s.get(Model, body.model_id)
    if not m or m.kind not in ("hf", "merged", "lora"):
        raise HTTPException(400, "Benchmarks need an HF, merged or LoRA model (not GGUF)")
    cfg = {"model_name": m.name, "tasks": body.tasks, "limit": body.limit, "num_fewshot": body.num_fewshot,
           "pretrained": m.base_model if m.kind == "lora" else m.path, "peft": m.path if m.kind == "lora" else None,
           "base_model": m.base_model or m.path}
    return svc.enqueue(f"bench-{m.name}", "bench", cfg)


class EmbedTrainBody(BaseModel):
    name: str
    base_model: str = "intfloat/multilingual-e5-small"
    collection_id: int | None = None  # (generated question → chunk) pairs from an enriched collection
    dataset_id: int | None = None  # or (user → assistant) pairs from a chat dataset
    # A gentle fine-tune by default: with synthetic questions, 2 epochs at 2e-5 overfit to their style and
    # got worse on real questions (measured: MRR 0.79 → 0.69); 1 epoch at 1e-5 improved slightly
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-5


@router.post("/embed")
def create_embed(body: EmbedTrainBody, s: Session = Depends(get_session)):
    """Fine-tunes the embedding model so RAG retrieves better in your domain."""
    import json as _json
    pairs: list[tuple[str, str]] = []
    if body.collection_id:
        from ..services import rag as rag_svc
        got = rag_svc._col(body.collection_id).get(include=["documents", "metadatas"])
        for d, m in zip(got["documents"], got["metadatas"]):
            for q in (m.get("questions") or "").splitlines():
                if q.strip():
                    pairs.append((q.strip(), rag_svc.contextual(d, {k: v for k, v in m.items() if k != "questions"})))
    if body.dataset_id:
        from ..services.datasets import read_all
        ds = s.get(Dataset, body.dataset_id)
        for r in read_all(ds.path) if ds and ds.format == "chat" else []:
            u = next((x["content"] for x in r["messages"] if x["role"] == "user"), None)
            a = next((x["content"] for x in r["messages"] if x["role"] == "assistant"), None)
            if u and a:
                pairs.append((u, a))
    if len(pairs) < 40:
        raise HTTPException(400, f"There are only {len(pairs)} pairs. Enrich the collection (questions per chunk) or use a bigger dataset.")
    out = settings.outputs_dir / f"{slugify(body.name)}-embed"
    out.mkdir(parents=True, exist_ok=True)
    # The pairs are written BEFORE queueing: the queue may start the job right away
    pairs_path = out / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(_json.dumps(p, ensure_ascii=False) + "\n")
    return svc.enqueue(body.name, "embed", {**body.model_dump(), "output_dir": str(out), "n_pairs": len(pairs),
                                            "pairs_path": str(pairs_path)})


def _get(s, jid) -> Job:
    j = s.get(Job, jid)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


@router.get("/{jid}")
def get_job(jid: int, s: Session = Depends(get_session)):
    return _get(s, jid)


@router.get("/{jid}/events")
def events(jid: int, limit: int = 5000):
    """The job's events. By default the last 5000: a long training run produces many more."""
    ev, _ = svc.read_events(jid)
    return ev[-limit:] if limit and len(ev) > limit else ev


@router.get("/{jid}/log")
def log(jid: int, tail: int = 20000):
    text, _ = svc.read_log(jid)
    return {"log": text[-tail:]}


@router.post("/{jid}/cancel")
def cancel(jid: int):
    svc.cancel(jid)
    return {"ok": True}


@router.post("/{jid}/kill")
def kill(jid: int):
    svc.kill(jid)
    return {"ok": True}


def _output_info(j: Job, s: Session) -> dict:
    """What's on disk for that job and whether any registered model depends on it.

    It matters because deleting the folder of a job whose LoRA is registered in Models would leave
    that model pointing at nothing."""
    from pathlib import Path

    out = j.config.get("output_dir")
    path = Path(out) if out else None
    if not path or not path.exists():
        return {"path": out, "exists": False, "size_mb": 0, "models": []}
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    # A model depends on the folder if its path is inside it
    used_by = [m.name for m in s.exec(select(Model)).all()
              if m.path and (Path(m.path) == path or path in Path(m.path).parents)]
    return {"path": str(path), "exists": True, "size_mb": round(size / 1e6, 1), "models": used_by}


@router.get("/{jid}/output")
def output_info(jid: int, s: Session = Depends(get_session)):
    """How much disk a job takes, so you can decide before deleting it."""
    return _output_info(_get(s, jid), s)


def _delete_one(j: Job, s: Session, delete_output: bool) -> dict:
    import shutil

    info = _output_info(j, s)
    freed, deleted = 0.0, False
    if delete_output and info["exists"] and not info["models"]:
        shutil.rmtree(info["path"], ignore_errors=True)
        freed, deleted = info["size_mb"], True
    s.delete(j)
    # “deleted_output” and not the size: a folder of a few KB rounds to 0.0 MB and would look untouched
    return {"id": j.id, "name": j.name, "freed_mb": freed, "deleted_output": deleted,
            "kept_output": bool(delete_output and info["models"]), "models": info["models"]}


@router.delete("/{jid}")
def delete(jid: int, delete_output: bool = False, s: Session = Depends(get_session)):
    j = _get(s, jid)
    if j.status in ("running", "queued"):
        raise HTTPException(400, "Cancel the job before deleting it")
    r = _delete_one(j, s, delete_output)
    s.commit()
    return r


class PurgeBody(BaseModel):
    ids: list[int] | None = None
    statuses: list[str] | None = None
    older_than_days: int | None = None
    dry_run: bool = True


@router.post("/purge-outputs")
def purge_outputs(body: PurgeBody, s: Session = Depends(get_session)):
    """Deletes the jobs' weights and checkpoints, but keeps their rows.

    It exists because the two things don't go together: weights take gigabytes and are almost never run
    again, while the metrics and curves are the evidence of what each training run measured
    and weigh nothing. Clearing the whole history to get disk back would throw away the latter
    because of the former.
    """
    import shutil
    from datetime import timedelta

    q = select(Job).where(Job.status.not_in(["running", "queued"]))
    if body.ids:
        q = q.where(Job.id.in_(body.ids))
    if body.statuses:
        q = q.where(Job.status.in_(body.statuses))
    if body.older_than_days:
        q = q.where(Job.created_at < now() - timedelta(days=body.older_than_days))

    details, freed_total = [], 0.0
    for j in s.exec(q).all():
        info = _output_info(j, s)
        if not info["exists"] or info["models"]:
            continue    # nothing to delete, or a registered model depends on it
        details.append({"id": j.id, "name": j.name, "size_mb": info["size_mb"]})
        freed_total += info["size_mb"]
        if not body.dry_run:
            shutil.rmtree(info["path"], ignore_errors=True)
            # The job itself records that its weights are gone
            j.config = {**j.config, "outputs_purged": True}
            s.add(j)
    if not body.dry_run:
        s.commit()
    return {"dry_run": body.dry_run, "count": len(details), "freed_mb": round(freed_total, 1), "jobs": details}


class CleanupBody(BaseModel):
    ids: list[int] | None = None
    statuses: list[str] | None = None       # p. ej. ["failed", "cancelled"]
    types: list[str] | None = None          # train | export | embed | bench
    older_than_days: int | None = None
    delete_outputs: bool = False
    dry_run: bool = True                    # by default it only reports


@router.post("/cleanup")
def cleanup(body: CleanupBody, s: Session = Depends(get_session)):
    """Deletes jobs in bulk. By default it only says what it would delete (`dry_run`).

    It never touches running jobs or folders a registered model depends on."""
    from datetime import timedelta

    q = select(Job).where(Job.status.not_in(["running", "queued"]))
    if body.ids:
        q = q.where(Job.id.in_(body.ids))
    if body.statuses:
        q = q.where(Job.status.in_(body.statuses))
    if body.types:
        q = q.where(Job.type.in_(body.types))
    if body.older_than_days:
        q = q.where(Job.created_at < now() - timedelta(days=body.older_than_days))
    jobs = s.exec(q).all()

    if body.dry_run:
        details = []
        for j in jobs:
            info = _output_info(j, s)
            details.append({"id": j.id, "name": j.name, "type": j.type, "status": j.status,
                            "size_mb": info["size_mb"], "models": info["models"]})
        freeable = sum(d["size_mb"] for d in details if not d["models"])
        return {"dry_run": True, "count": len(details), "jobs": details,
                "freeable_mb": round(freeable, 1),
                "protected": [d["name"] for d in details if d["models"]]}

    done = [_delete_one(j, s, body.delete_outputs) for j in jobs]
    s.commit()
    return {"dry_run": False, "count": len(done),
            "freed_mb": round(sum(h["freed_mb"] for h in done), 1),
            "deleted_outputs": sum(h["deleted_output"] for h in done),
            "protected": [h["name"] for h in done if h["kept_output"]]}


@router.get("/{jid}/stream")
async def stream(jid: int):
    """SSE: emits the worker's events and new log lines until the job finishes."""
    from ..db import engine

    async def gen():
        ev_off, log_off = 0, 0
        while True:
            # The status is read BEFORE the events so the last ones aren't lost when it finishes
            with Session(engine) as s:
                j = s.get(Job, jid)
                status = j.status if j else "missing"
            events, ev_off = svc.read_events(jid, ev_off)
            if events:
                yield {"event": "events", "data": json.dumps(events)}
            text, log_off = svc.read_log(jid, log_off)
            if text:
                yield {"event": "log", "data": json.dumps(text)}
            yield {"event": "status", "data": json.dumps(status)}
            if status in ("completed", "failed", "cancelled", "missing"):
                break
            await asyncio.sleep(1)

    return EventSourceResponse(gen())

