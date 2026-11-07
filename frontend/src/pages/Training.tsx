import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Play } from "lucide-react";
import { api, type Dataset, type Job, type Model } from "../api";
import { Adv, Check, Field, Help, Progress, SimpleNote, useData, useDialog, useMode, useToast } from "../components/ui";
import { Bench, EmbedTrain, Sweep } from "./TrainExtras";

const DEFAULTS = {
  name: "", base_model: "Qwen/Qwen3-0.6B", base_model_id: null as number | null, dataset_id: 0, method: "sft", backend: "hf",
  use_lora: true, quantization: "4bit", lora_r: 16, lora_alpha: 32, lora_dropout: 0.05, use_rslora: false, use_dora: false,
  epochs: 1, max_steps: 0, batch_size: 2, grad_accum: 4, learning_rate: 2e-4, scheduler: "cosine", warmup_ratio: 0.03,
  weight_decay: 0.0, max_seq_len: 2048, packing: false, train_on_responses_only: true, gradient_checkpointing: true,
  eval_ratio: 0, logging_steps: 5, save_steps: 200, eval_steps: 50, neftune_alpha: 0, dpo_beta: 0.1, merge_after: false, seed: 42,
  orpo_beta: 0.1, rewards: ["correctness"] as string[], reward_regex: "", reward_length_target: 80, num_generations: 4,
  max_completion_length: 256, grpo_beta: 0.0, grpo_temperature: 0.9, grpo_system_prompt: "",
};
type Cfg = typeof DEFAULTS;

const METHOD_HELP: Record<string, string> = {
  sft: "Learns to answer by imitating your examples (chat/instruction dataset).",
  dpo: "Learns which answer is preferred over another; usually applied after SFT.",
  orpo: "Combines SFT and preferences in a single step, with no reference model.",
  kto: "Unpaired preferences: marks answers as good or bad.",
  grpo: "The model generates several answers per question and the ones that earn the most reward are reinforced.",
  pretrain: "Absorbs text from your domain without a conversation format.",
};
const METHOD_NEEDS: Record<string, string> = {
  dpo: "Needs prompt/chosen/rejected.", orpo: "Needs prompt/chosen/rejected.", kto: "Needs prompt/chosen/rejected.",
  grpo: "Needs questions with a reference answer (chat or preferences).", pretrain: "Uses a text or chat dataset.",
};

const PRESETS: Record<string, { label: string; desc: string; when: string; caution?: string; recommended?: boolean; cfg: Partial<Cfg> }> = {
  quick: {
    label: "Quick test", desc: "60 steps to validate the pipeline",
    when: "Always start here. It isn't trying to produce a good model: it checks in 1–2 minutes that your dataset loads, that it fits in VRAM and that the loss goes down. If this fails, any long run would have failed the same way, just half an hour later.",
    caution: "The resulting model is useless: 60 steps don't teach anything.",
    cfg: { max_steps: 60, lora_r: 8, lora_alpha: 16, learning_rate: 2e-4 },
  },
  qlora: {
    label: "Standard QLoRA", recommended: true, desc: "Balanced for 7–14B on 24 GB",
    when: "The everyday choice. It loads the base model in 4 bits, which is what lets you train an 8B or a 14B on a 24 GB GPU, and trains a medium-sized adapter for 2 epochs.",
    cfg: { max_steps: 0, epochs: 2, lora_r: 16, lora_alpha: 32, quantization: "4bit", learning_rate: 2e-4 },
  },
  quality: {
    label: "High quality", desc: "16-bit LoRA, high rank (≤8B)",
    when: "When the result matters more than the time and the model is 8B or smaller. With the base unquantized (16 bits) and an adapter four times bigger, it learns nuances the standard preset misses.",
    caution: "Uses quite a bit more VRAM and takes three times as long. With a 14B model you'll run out of memory.",
    cfg: { max_steps: 0, epochs: 3, lora_r: 64, lora_alpha: 64, use_rslora: true, quantization: "none", learning_rate: 1e-4 },
  },
  dpo: {
    label: "DPO (alignment)", desc: "chosen/rejected preferences",
    when: "To teach which answer is preferred over another: tone, format, refusing something. It's applied AFTER an SFT, not instead of it.",
    caution: "Needs a preference dataset (prompt / chosen / rejected). It won't work with a normal chat dataset.",
    cfg: { method: "dpo", max_steps: 0, epochs: 1, learning_rate: 5e-6, lora_r: 16, lora_alpha: 32, dpo_beta: 0.1 },
  },
  grpo: {
    label: "GRPO (reasoning)", desc: "Reinforcement with verifiable rewards",
    when: "When you can score an answer with an objective rule: whether the result is correct, whether it follows a format, whether it has the requested length. The model generates several answers per question and the highest-scoring ones are reinforced.",
    caution: "By far the slowest, because it generates several answers per example. With the vLLM engine it's about 9× faster.",
    cfg: { method: "grpo", max_steps: 100, learning_rate: 1e-5, lora_r: 16, lora_alpha: 32, batch_size: 4, grad_accum: 2, num_generations: 4, rewards: ["correctness", "format"], max_seq_len: 1024 },
  },
};

/** Rough VRAM estimate in GB. */
function estimateVram(model: string, c: Cfg): number | null {
  const m = model.match(/(\d+(?:\.\d+)?)\s*[bB](?![a-z])/);
  if (!m) return null;
  const p = parseFloat(m[1]);
  const bytesPerParam = !c.use_lora ? 16 : c.quantization === "4bit" ? 0.6 : c.quantization === "8bit" ? 1.1 : 2.05;
  const weights = p * bytesPerParam;
  const act = (c.batch_size * c.max_seq_len / 2048) * (c.gradient_checkpointing ? 0.6 : 3) * Math.sqrt(p) * (c.method === "dpo" ? 2 : 1);
  // Unsloth: offloaded checkpointing and custom kernels cut activations a lot
  return Math.round((weights + act * (c.backend === "unsloth" ? 0.4 : 1) + 1.2) * 10) / 10;
}

export default function Training() {
  const [jobs] = useData<Job[]>("/api/jobs", 4000);
  const [tab, setTab] = useState<"llm" | "sweep" | "embed" | "bench">("llm");
  // Only for the banner above: the full history lives on the Jobs screen
  const running = (jobs ?? []).filter((j) => j.status === "running" || j.status === "queued");
  return (
    <>
      <div className="page-head">
        <div><h1>Training</h1><div className="sub">SFT, DPO or continued pretraining with LoRA / QLoRA / full fine-tuning</div></div>
      </div>
      <Help title="How does training work?">
        <p><b>Fine-tuning</b> means taking an already trained model and adjusting it with your examples. You decide <b>two independent things</b> here, and it's best not to mix them up:</p>
        <p style={{ marginTop: 8 }}><b>1. What you teach it</b> — the “Method”:</p>
        <ul>
          <li><b>SFT</b>: you give it questions with their good answer and it learns to imitate them. It's 90% of cases and where you start.</li>
          <li><b>DPO / ORPO / KTO</b>: you give it two answers to the same question, one better than the other, and it learns to prefer. Useful for tone and form, not knowledge. It comes after an SFT.</li>
          <li><b>GRPO</b>: you don't give it the answer, you give it <i>a way to score it</i> (is the number right? does it follow the format?). The model generates several and the highest-scoring ones are reinforced. Only useful if you can check correctness with a rule.</li>
          <li><b>Continued pretraining</b>: you give it loose text from your domain, with no questions or answers, so it absorbs the vocabulary.</li>
        </ul>
        <p style={{ marginTop: 8 }}><b>2. How you touch the weights</b> — the “Adapter”:</p>
        <ul>
          <li><b>Full fine-tune</b>: you modify the model's billions of parameters. The best and the most expensive; an 8B model doesn't fit in 24 GB this way.</li>
          <li><b>LoRA</b>: you don't touch the model. You attach a few small new matrices and train <b>only those</b>. The result is an “adapter” of a few MB used alongside the original model. Almost as good and far cheaper.</li>
          <li><b>QLoRA</b>: LoRA, plus the base model is loaded compressed to 4 bits so it takes a quarter of the space. <b>It's what lets you train an 8B or a 14B on a 24 GB GPU.</b> The compression is only for training; the resulting adapter works the same.</li>
        </ul>
        <p style={{ marginTop: 8 }}>The presets above combine both: <b>Quick test</b>, <b>Standard QLoRA</b> and <b>High quality</b> are all <b>SFT</b> and only change the adapter and the length; <b>DPO</b> and <b>GRPO</b> do change what it learns.</p>
        <p style={{ marginTop: 8 }}><b>If you don't know which to pick</b>: click <b>Quick test</b>, check in two minutes that everything works, then <b>Standard QLoRA</b>, which is the right one for almost everything.</p>
        <p style={{ marginTop: 8 }}>Tabs: <b>LLM</b> trains the language model · <b>Sweep</b> tries several configurations and ranks them · <b>Embeddings</b> improves RAG search · <b>Benchmarks</b> checks that fine-tuning didn't make it forget general skills.</p>
        <p style={{ marginTop: 8 }}>When it finishes, the adapter shows up in <b>Models</b>, ready to try in the Playground or export to GGUF. The history of what you've launched, with its curves and the comparison between runs, is in <b>Jobs</b>.</p>
      </Help>
      {!!running.length && (
        <div className="card" style={{ borderColor: "var(--accent)" }}>
          <div className="row between">
            <div>
              <b>{running.length === 1 ? "A job is running" : `${running.length} jobs are running`}</b>
              <div className="sub">{running.map((j) => j.name).join(", ")}</div>
            </div>
            <Link to="/jobs">See it in Jobs →</Link>
          </div>
          {running.map((j) => j.progress?.max_steps ? (
            <div key={j.id} style={{ marginTop: 6 }}>
              <Progress value={j.progress.step / j.progress.max_steps} />
              <div className="sub" style={{ fontSize: 11 }}>{j.name}: {j.progress.step}/{j.progress.max_steps} steps</div>
            </div>
          ) : null)}
        </div>
      )}
      <div className="stack" style={{ gap: 0 }}>
        <div className="tabs">
          <button className={`tab ${tab === "llm" ? "active" : ""}`} onClick={() => setTab("llm")}>LLM</button>
          <button className={`tab ${tab === "sweep" ? "active" : ""}`} onClick={() => setTab("sweep")}>Sweep</button>
          <button className={`tab ${tab === "embed" ? "active" : ""}`} onClick={() => setTab("embed")}>Embeddings</button>
          <button className={`tab ${tab === "bench" ? "active" : ""}`} onClick={() => setTab("bench")}>Benchmarks</button>
        </div>
        {tab === "llm" ? <NewJob /> : tab === "sweep" ? <Sweep /> : tab === "embed" ? <EmbedTrain /> : <Bench />}
      </div>
    </>
  );
}

function NewJob() {
  const [datasets] = useData<Dataset[]>("/api/datasets");
  const [models] = useData<Model[]>("/api/models");
  const [suggested] = useData<any[]>("/api/models/suggested");
  const [sys] = useData<any>("/api/system");
  const [c, setC] = useState<Cfg>(DEFAULTS);
  const unslothOk = !!sys?.unsloth;
  const vllmOk = !!sys?.vllm;
  // Unsloth by default when it's installed
  useEffect(() => { if (unslothOk) setC((p) => ({ ...p, backend: "unsloth", lora_dropout: 0 })); }, [unslothOk]);
  // vLLM only applies to GRPO: if the method changes, go back to the default engine
  useEffect(() => { setC((p) => (p.backend === "vllm" && p.method !== "grpo" ? { ...p, backend: unslothOk ? "unsloth" : "hf" } : p)); }, [c.method, unslothOk]);
  // vLLM can't generate with quantized weights (bitsandbytes): LoRA is trained in 16 bits
  useEffect(() => { setC((p) => (p.backend === "vllm" && p.use_lora && p.quantization !== "none" ? { ...p, quantization: "none" } : p)); }, [c.backend]);
  const { simple, setSimple } = useMode();
  const [preset, setPreset] = useState<string | null>(null);
  const { notify } = useToast();
  const { prompt } = useDialog();
  const nav = useNavigate();
  const set = <K extends keyof Cfg>(k: K, v: Cfg[K]) => setC((p) => ({ ...p, [k]: v }));
  const num = (k: keyof Cfg) => (e: React.ChangeEvent<HTMLInputElement>) => set(k, (e.target.value === "" ? 0 : +e.target.value) as never);

  const ds = datasets?.find((d) => d.id === c.dataset_id);
  const baseModels = models?.filter((m) => m.kind === "hf" || m.kind === "merged") ?? [];
  const baseName = c.base_model_id ? baseModels.find((m) => m.id === c.base_model_id)?.name ?? "" : c.base_model;
  const vram = useMemo(() => estimateVram(baseName + " " + c.base_model, c), [baseName, c]);
  const steps = ds ? Math.ceil((ds.num_rows * (1 - c.eval_ratio)) / (c.batch_size * c.grad_accum)) * c.epochs : null;

  const submit = async () => {
    if (!c.dataset_id) return notify("Choose a dataset", "err");
    try {
      const body: any = { ...c, name: c.name || `${baseName.split("/").pop()}-${ds?.name}`, max_steps: c.max_steps || null, neftune_alpha: c.neftune_alpha || null,
        reward_regex: c.reward_regex || null, grpo_system_prompt: c.grpo_system_prompt || null };
      if (c.method === "grpo" && !c.rewards.length) return notify("Choose at least one reward", "err");
      if (!body.base_model_id) delete body.base_model_id;
      const job = await api.post<Job>("/api/jobs/train", body);
      nav(`/training/${job.id}`);
    } catch (e: any) { notify(e.message, "err"); }
  };

  const methodMismatch = ds && ((["dpo", "orpo", "kto"].includes(c.method) && ds.format !== "preference") || (c.method === "pretrain" && ds.format === "preference") || (c.method === "grpo" && ds.format === "text"));
  const toggleReward = (r: string) => set("rewards", c.rewards.includes(r) ? c.rewards.filter((x) => x !== r) : [...c.rewards, r]);

  return (
    <div className="card stack" style={{ gap: 14 }}>
      <h2 style={{ margin: 0 }}>New training run</h2>
      <div className="row">
        {Object.entries(PRESETS).map(([k, p]) => (
          <button key={k} className={`sm ${preset === k ? "primary" : ""}`} title={p.desc} onClick={() => { setPreset(k); setC((prev) => ({ ...prev, ...DEFAULTS, name: prev.name, base_model: prev.base_model, base_model_id: prev.base_model_id, dataset_id: prev.dataset_id, backend: prev.backend, lora_dropout: prev.backend === "unsloth" ? 0 : DEFAULTS.lora_dropout, ...p.cfg })); }}>{p.label}{p.recommended ? " ★" : ""}</button>
        ))}
      </div>

      {preset && (() => {
        const p = PRESETS[preset];
        const q = p.cfg.quantization ?? DEFAULTS.quantization;
        const length = p.cfg.max_steps ? `${p.cfg.max_steps} steps` : `${p.cfg.epochs ?? DEFAULTS.epochs} epoch(s)`;
        return (
          <div className="preset-info">
            <div className="row between">
              <b>{p.label}{p.recommended ? " · recommended" : ""}</b>
              <button className="sm ghost" onClick={() => setPreset(null)}>Hide</button>
            </div>
            <p>{p.when}</p>
            {p.caution && <p className="caution">⚠ {p.caution}</p>}
            <div className="chips">
              <code className="chip">{(p.cfg.method ?? DEFAULTS.method).toUpperCase()}</code>
              <code className="chip">{q === "4bit" ? "QLoRA 4-bit" : q === "8bit" ? "LoRA 8-bit" : "LoRA 16-bit"}</code>
              <code className="chip">rank {p.cfg.lora_r ?? DEFAULTS.lora_r}</code>
              <code className="chip">lr {p.cfg.learning_rate ?? DEFAULTS.learning_rate}</code>
              <code className="chip">{length}</code>
            </div>
          </div>
        );
      })()}

      <fieldset>
        <legend>Model and data</legend>
        <div className="grid g2">
          <Field label="Base model" hint="A Hugging Face repo or a registered model">
            <select value={c.base_model_id ? `id:${c.base_model_id}` : `hf:${c.base_model}`} onChange={(e) => {
              const v = e.target.value;
              if (v.startsWith("id:")) set("base_model_id", +v.slice(3));
              else if (v === "custom") prompt("Base model", { message: "Hugging Face repo (user/model) or a local path to a model folder.", placeholder: "Qwen/Qwen3-8B" }).then((r) => r && setC((p) => ({ ...p, base_model: r, base_model_id: null })));
              else setC((p) => ({ ...p, base_model: v.slice(3), base_model_id: null }));
            }}>
              {baseModels.length > 0 && <optgroup label="Registered">{baseModels.map((m) => <option key={m.id} value={`id:${m.id}`}>{m.name}</option>)}</optgroup>}
              <optgroup label="Hugging Face">
                {!suggested?.some((s) => s.repo === c.base_model) && <option value={`hf:${c.base_model}`}>{c.base_model}</option>}
                {suggested?.map((s) => <option key={s.repo} value={`hf:${s.repo}`}>{s.repo} ({s.size})</option>)}
              </optgroup>
              <option value="custom">Other…</option>
            </select>
          </Field>
          <Field label="Dataset">
            <select value={c.dataset_id} onChange={(e) => set("dataset_id", +e.target.value)}>
              <option value={0}>— choose —</option>
              {datasets?.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows} · {d.format})</option>)}
            </select>
          </Field>
          <Field label="Method">
            <select value={c.method} onChange={(e) => set("method", e.target.value)}>
              <option value="sft">SFT — supervised fine-tuning (instructions/chat)</option>
              <option value="dpo">DPO — preferences</option>
              <option value="orpo">ORPO — SFT + preferences in one step (no reference model)</option>
              <option value="kto">KTO — unpaired preferences (good/bad)</option>
              <option value="grpo">GRPO — reinforcement with rewards (reasoning, format)</option>
              <option value="pretrain">Continued pretraining (text)</option>
            </select>
          </Field>
          <Adv><Field label="Job name"><input value={c.name} onChange={(e) => set("name", e.target.value)} placeholder="auto" /></Field></Adv>
          <Field label="Training engine" hint={unslothOk ? "Unsloth: optimized kernels, less VRAM and faster" : "Install Unsloth with backend/setup-unsloth.sh"}>
            <select value={c.backend} onChange={(e) => set("backend", e.target.value)}>
              <option value="unsloth" disabled={!unslothOk}>Unsloth (recommended){unslothOk ? "" : " — not installed"}</option>
              <option value="vllm" disabled={!vllmOk || c.method !== "grpo"}>vLLM (GRPO only; 16-bit LoRA){vllmOk ? "" : " — not installed"}</option>
              <option value="hf">Hugging Face (TRL + PEFT)</option>
            </select>
          </Field>
        </div>
        <div className="sub" style={{ marginTop: 8, fontSize: 12.5 }}>{METHOD_HELP[c.method]}</div>
        {methodMismatch && <div className="error-box" style={{ marginTop: 10 }}>The chosen method doesn't match the dataset's format ({ds?.format}). {METHOD_NEEDS[c.method]}</div>}
      </fieldset>

      {c.method === "grpo" && (
        <fieldset>
          <legend>GRPO — rewards</legend>
          <div className="sub" style={{ marginBottom: 8 }}>For each prompt the model generates {c.num_generations} answers; the ones that earn the most reward are reinforced. The dataset's answer is used as the reference.</div>
          <div className="row" style={{ gap: 16 }}>
            <Check label="Correctness (vs. reference)" checked={c.rewards.includes("correctness")} onChange={() => toggleReward("correctness")} />
            <Check label="Format: reasons inside <think>" checked={c.rewards.includes("format")} onChange={() => toggleReward("format")} />
            <Check label={`Length ≤ ${c.reward_length_target} words`} checked={c.rewards.includes("length")} onChange={() => toggleReward("length")} />
            <Check label="Regular expression" checked={c.rewards.includes("regex")} onChange={() => toggleReward("regex")} />
          </div>
          <div className="grid g3" style={{ marginTop: 10 }}>
            {/* Only the settings that depend on the rewards you ticked stay visible */}
            {c.rewards.includes("length") && <Field label="Target words"><input type="number" value={c.reward_length_target} onChange={num("reward_length_target")} /></Field>}
            {c.rewards.includes("regex") && <Field label="Regex it must match"><input value={c.reward_regex} onChange={(e) => set("reward_regex", e.target.value)} placeholder="<answer>.*</answer>" /></Field>}
            <Adv>
              <Field label="Generations per prompt" hint="More answers per question = a better signal, but much slower"><input type="number" value={c.num_generations} onChange={num("num_generations")} /></Field>
              <Field label="Max tokens per answer"><input type="number" value={c.max_completion_length} onChange={num("max_completion_length")} /></Field>
              <Field label="Sampling temperature" hint="High = varied answers, which is what GRPO needs to compare"><input type="number" step="0.05" value={c.grpo_temperature} onChange={num("grpo_temperature")} /></Field>
              <Field label="β (KL)" hint="How far it may drift from the original model. 0 = no brake, and faster"><input type="number" step="0.01" value={c.grpo_beta} onChange={num("grpo_beta")} /></Field>
            </Adv>
          </div>
          <Adv>
            <Field label="System prompt for GRPO (optional)" hint="e.g. “Reason inside <think></think>, then answer.”">
              <input value={c.grpo_system_prompt} onChange={(e) => set("grpo_system_prompt", e.target.value)} />
            </Field>
          </Adv>
          <SimpleNote>
            Generates {c.num_generations} answers per question at temperature {c.grpo_temperature} with up to
            {" "}{c.max_completion_length} tokens, with no KL brake. The only thing you usually want to change is the
            rewards above.
          </SimpleNote>
        </fieldset>
      )}

      <fieldset>
        <legend>Adapter</legend>
        <div className="grid g3">
          <Field label="Type">
            <select value={c.use_lora ? c.quantization : "full"} onChange={(e) => {
              const v = e.target.value;
              if (v === "full") setC((p) => ({ ...p, use_lora: false, quantization: "none" }));
              else setC((p) => ({ ...p, use_lora: true, quantization: v }));
            }}>
              <option value="4bit">QLoRA (4-bit)</option>
              <option value="8bit">LoRA 8-bit</option>
              <option value="none">LoRA 16-bit</option>
              <option value="full">Full fine-tune</option>
            </select>
          </Field>
          <Adv>
            <Field label="Rank (r)"><input type="number" value={c.lora_r} disabled={!c.use_lora} onChange={num("lora_r")} /></Field>
            <Field label="Alpha"><input type="number" value={c.lora_alpha} disabled={!c.use_lora} onChange={num("lora_alpha")} /></Field>
          </Adv>
        </div>
      </fieldset>

      <fieldset>
        <legend>Hyperparameters</legend>
        <div className="grid g3">
          <Field label="Epochs"><input type="number" step="0.5" value={c.epochs} onChange={num("epochs")} /></Field>
          <Adv><Field label="Max steps" hint="0 = use epochs"><input type="number" value={c.max_steps} onChange={num("max_steps")} /></Field></Adv>
          <Field label="Learning rate"><input type="number" step="0.00001" value={c.learning_rate} onChange={num("learning_rate")} /></Field>
          <Adv>
            <Field label="Batch per GPU"><input type="number" value={c.batch_size} onChange={num("batch_size")} /></Field>
            <Field label="Gradient accumulation"><input type="number" value={c.grad_accum} onChange={num("grad_accum")} /></Field>
            <Field label="Max length (tokens)"><input type="number" step="256" value={c.max_seq_len} onChange={num("max_seq_len")} /></Field>
          </Adv>
        </div>
        <div className="row" style={{ marginTop: 10, gap: 18 }}>
          {c.method === "sft" && <Check label="Train on responses only" checked={c.train_on_responses_only} onChange={(v) => set("train_on_responses_only", v)} />}
          <Check label="Merge LoRA when done" checked={c.merge_after} onChange={(v) => set("merge_after", v)} />
          <button className="ghost sm" onClick={() => setSimple(!simple)}>
            {simple ? "Show all parameters" : "Back to simple mode"}
          </button>
        </div>
        <SimpleNote>
          Trains with LoRA r={c.lora_r}/α={c.lora_alpha}, batch {c.batch_size}×{c.grad_accum} and {c.max_seq_len} tokens
          of length: values that fit comfortably on a 3090.
        </SimpleNote>
      </fieldset>

      {!simple && (
        <fieldset>
          <legend>Advanced</legend>
          <div className="grid g3">
            <Field label="Scheduler">
              <select value={c.scheduler} onChange={(e) => set("scheduler", e.target.value)}>
                {["cosine", "linear", "constant", "constant_with_warmup", "cosine_with_restarts"].map((s) => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Warmup (ratio)"><input type="number" step="0.01" value={c.warmup_ratio} onChange={num("warmup_ratio")} /></Field>
            <Field label="Weight decay"><input type="number" step="0.01" value={c.weight_decay} onChange={num("weight_decay")} /></Field>
            <Field label="LoRA dropout"><input type="number" step="0.01" value={c.lora_dropout} onChange={num("lora_dropout")} /></Field>
            <Field label="Eval split" hint="Holds out a share for eval_loss"><input type="number" step="0.05" value={c.eval_ratio} onChange={num("eval_ratio")} /></Field>
            <Field label="Eval every N steps"><input type="number" value={c.eval_steps} onChange={num("eval_steps")} /></Field>
            <Field label="Log every N steps"><input type="number" value={c.logging_steps} onChange={num("logging_steps")} /></Field>
            <Field label="Checkpoint every N steps"><input type="number" value={c.save_steps} onChange={num("save_steps")} /></Field>
            <Field label="NEFTune α" hint="0 = off; 5 is typical"><input type="number" value={c.neftune_alpha} onChange={num("neftune_alpha")} /></Field>
            {c.method === "orpo" && <Field label="ORPO β"><input type="number" step="0.05" value={c.orpo_beta} onChange={num("orpo_beta")} /></Field>}
            {(c.method === "dpo" || c.method === "kto") && <Field label="DPO β"><input type="number" step="0.05" value={c.dpo_beta} onChange={num("dpo_beta")} /></Field>}
            <Field label="Seed"><input type="number" value={c.seed} onChange={num("seed")} /></Field>
          </div>
          <div className="row" style={{ marginTop: 10, gap: 18 }}>
            <Check label="Gradient checkpointing" checked={c.gradient_checkpointing} onChange={(v) => set("gradient_checkpointing", v)} />
            <Check label="Packing" checked={c.packing} onChange={(v) => set("packing", v)} />
            <Check label="rsLoRA" checked={c.use_rslora} onChange={(v) => set("use_rslora", v)} />
            {c.backend === "hf" && <Check label="DoRA" checked={c.use_dora} onChange={(v) => set("use_dora", v)} />}
          </div>
        </fieldset>
      )}

      <div className="row between">
        <div className="sub">
          {vram !== null && <>Estimated VRAM: <b style={{ color: vram > 23 ? "var(--err)" : "var(--text)" }}>~{vram} GB</b> · </>}
          Effective batch: <b>{c.batch_size * c.grad_accum}</b>
          {steps !== null && !c.max_steps && <> · ≈ <b>{steps}</b> steps</>}
        </div>
        <button className="primary" onClick={submit} disabled={!c.dataset_id || !!methodMismatch}><Play size={14} /> Start training</button>
      </div>
    </div>
  );
}
