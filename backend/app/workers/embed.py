"""Embedding fine-tuning worker (sentence-transformers). Usage: python -m app.workers.embed <job_dir>/config.json

It trains on (query, passage) pairs with MultipleNegativesRankingLoss (the other passages in the batch act as
negatives). It measures recall@k and MRR on a held-out set before and after, to check the improvement.
"""
import json
import os
import random
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .common import Emitter, StopFlag, load_config  # noqa: E402


def prefix(model: str, kind: str) -> str:
    return ("query: " if kind == "query" else "passage: ") if "e5" in model.lower() else ""


def ir_metrics(model, pairs, name, k=(1, 5, 10)):
    """Recall@k and MRR@10, retrieving each query against all the passages in the evaluation set."""
    import numpy as np
    passages = list(dict.fromkeys(p for _, p in pairs))
    pid = {p: i for i, p in enumerate(passages)}
    q = model.encode([prefix(name, "query") + a for a, _ in pairs], normalize_embeddings=True, batch_size=64)
    d = model.encode([prefix(name, "passage") + p for p in passages], normalize_embeddings=True, batch_size=64)
    ranks = []
    for i, (_, p) in enumerate(pairs):
        order = np.argsort(-(d @ q[i]))
        ranks.append(int(np.where(order == pid[p])[0][0]) + 1)
    ranks = np.array(ranks)
    out = {f"recall@{x}": round(float((ranks <= x).mean()), 4) for x in k}
    out["mrr@10"] = round(float(np.mean([1 / r if r <= 10 else 0 for r in ranks])), 4)
    return out


def main():
    cfg, job_dir = load_config()
    em = Emitter(job_dir)
    stop = StopFlag()
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
        from sentence_transformers.training_args import BatchSamplers
        from datasets import Dataset
        from transformers import TrainerCallback

        base = cfg["base_model"]
        with open(cfg["pairs_path"], encoding="utf-8") as f:
            pairs = [tuple(json.loads(l)) for l in f if l.strip()]
        random.Random(cfg.get("seed", 42)).shuffle(pairs)
        n_eval = max(20, int(len(pairs) * cfg.get("eval_ratio", 0.15)))
        eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]
        em.status(f"{len(train_pairs)} training pairs, {len(eval_pairs)} for evaluation")
        if len(train_pairs) < 20:
            raise ValueError("At least ~40 pairs are needed; enrich more chunks or use a bigger dataset")

        model = SentenceTransformer(base, device="cuda")
        em.status("Measuring the base model's retrieval…")
        before = ir_metrics(model, eval_pairs, base)
        em.emit("embed_eval", stage="before", **before)

        ds = Dataset.from_dict({"anchor": [prefix(base, "query") + a for a, _ in train_pairs],
                                "positive": [prefix(base, "passage") + p for _, p in train_pairs]})
        out_dir = cfg["output_dir"]

        class Live(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs:
                    em.emit("log", step=state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))})

            def on_step_end(self, args, state, control, **kw):
                em.emit("progress", step=state.global_step, max_steps=state.max_steps, epoch=state.epoch)
                if stop.stop:
                    control.should_training_stop = True

        args = SentenceTransformerTrainingArguments(
            output_dir=out_dir, num_train_epochs=cfg.get("epochs", 2), per_device_train_batch_size=cfg.get("batch_size", 32),
            learning_rate=cfg.get("learning_rate", 2e-5), warmup_ratio=0.1, bf16=True, logging_steps=5,
            save_strategy="no", report_to=[], seed=cfg.get("seed", 42),
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # avoids the same passage twice in a batch (false negative)
        )
        trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                             loss=losses.MultipleNegativesRankingLoss(model), callbacks=[Live()])
        em.status("Training embeddings…")
        trainer.train()

        em.status("Measuring the fine-tuned model's retrieval…")
        after = ir_metrics(model, eval_pairs, base)
        em.emit("embed_eval", stage="after", **after)
        final_dir = os.path.join(out_dir, "final")
        model.save(final_dir)
        with open(os.path.join(final_dir, "llmtrainer.json"), "w") as f:
            json.dump({"base_model": base, "kind": "embedding", "before": before, "after": after}, f)
        em.emit("saved_embedding", path=final_dir, before=before, after=after)
        em.emit("done", elapsed=time.time() - t0, stopped=stop.stop)
    except Exception as e:
        traceback.print_exc()
        em.emit("error", message=str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
