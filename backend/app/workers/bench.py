"""Standard benchmarks worker with lm-evaluation-harness. Usage: python -m app.workers.bench <job_dir>/config.json

It's for detecting whether fine-tuning made the model “forget” general skills (comparing base vs fine-tuned).
"""
import os
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .common import Emitter, load_config  # noqa: E402

# Main metric to show per task (the first one that exists)
PRIMARY = ["acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"]


def main():
    cfg, job_dir = load_config()
    em = Emitter(job_dir)
    t0 = time.time()
    try:
        import lm_eval

        margs = f"pretrained={cfg['pretrained']},dtype=bfloat16,trust_remote_code={cfg.get('trust_remote_code', False)}"
        if cfg.get("peft"):
            margs += f",peft={cfg['peft']}"
        tasks = cfg["tasks"]
        em.status(f"Evaluating {cfg['model_name']} on {', '.join(tasks)} (limit {cfg.get('limit') or 'none'} examples/task)")
        em.emit("progress", step=0, max_steps=len(tasks))
        results = {}
        for i, task in enumerate(tasks):
            em.status(f"Task {i + 1}/{len(tasks)}: {task}")
            out = lm_eval.simple_evaluate(model="hf", model_args=margs, tasks=[task], limit=cfg.get("limit") or None,
                                          batch_size=cfg.get("batch_size", "auto"), num_fewshot=cfg.get("num_fewshot"),
                                          apply_chat_template=cfg.get("apply_chat_template", False), log_samples=False)
            res = out["results"]
            # Group tasks (mmlu, spanish_bench…) return subtasks: we use the aggregate if it exists
            agg = out.get("groups", {}).get(task) or res.get(task) or next(iter(res.values()))
            key = next((k for k in PRIMARY if k in agg), None)
            results[task] = {"metric": key.split(",")[0] if key else None, "value": agg.get(key) if key else None,
                             "stderr": agg.get(key.replace(",", "_stderr,", 1)) if key else None,
                             "subtasks": len(res)}
            em.emit("bench_task", task=task, **results[task])
            em.emit("progress", step=i + 1, max_steps=len(tasks))
        em.emit("bench_result", model=cfg["model_name"], tasks=results, limit=cfg.get("limit"))
        em.emit("done", elapsed=time.time() - t0)
    except Exception as e:
        traceback.print_exc()
        em.emit("error", message=str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
