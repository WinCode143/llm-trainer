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
  sft: "Aprende a responder imitando tus ejemplos (dataset de chat/instrucciones).",
  dpo: "Aprende qué respuesta se prefiere frente a otra; suele aplicarse después de un SFT.",
  orpo: "Combina SFT y preferencias en un solo paso, sin modelo de referencia.",
  kto: "Preferencias sin emparejar: marca respuestas como buenas o malas.",
  grpo: "El modelo genera varias respuestas por pregunta y se refuerzan las que más recompensa obtienen.",
  pretrain: "Absorbe texto de tu dominio sin formato de conversación.",
};
const METHOD_NEEDS: Record<string, string> = {
  dpo: "Necesita prompt/chosen/rejected.", orpo: "Necesita prompt/chosen/rejected.", kto: "Necesita prompt/chosen/rejected.",
  grpo: "Necesita preguntas con respuesta de referencia (chat o preferencias).", pretrain: "Usa un dataset de texto o chat.",
};

const PRESETS: Record<string, { label: string; desc: string; cuando: string; ojo?: string; recomendado?: boolean; cfg: Partial<Cfg> }> = {
  quick: {
    label: "Prueba rápida", desc: "60 pasos para validar el pipeline",
    cuando: "Empieza siempre por aquí. No busca un buen modelo: busca comprobar en 1–2 minutos que tu dataset se lee bien, que cabe en la VRAM y que la pérdida baja. Si esto falla, cualquier entrenamiento largo habría fallado igual, pero tras media hora.",
    ojo: "El modelo que salga no sirve para nada: 60 pasos no aprenden.",
    cfg: { max_steps: 60, lora_r: 8, lora_alpha: 16, learning_rate: 2e-4 },
  },
  qlora: {
    label: "QLoRA estándar", recomendado: true, desc: "Equilibrado para 7–14B en 24 GB",
    cuando: "El de todos los días. Carga el modelo base en 4 bits, que es lo que permite entrenar un 8B o un 14B en una GPU de 24 GB, y entrena un adaptador de tamaño medio durante 2 épocas.",
    cfg: { max_steps: 0, epochs: 2, lora_r: 16, lora_alpha: 32, quantization: "4bit", learning_rate: 2e-4 },
  },
  quality: {
    label: "Alta calidad", desc: "LoRA 16-bit, rango alto (≤8B)",
    cuando: "Cuando el resultado importa más que el tiempo y el modelo es de 8B o menos. Sin cuantizar el base (16 bits) y con un adaptador cuatro veces mayor, aprende matices que el estándar se deja.",
    ojo: "Gasta bastante más VRAM y tarda el triple. Con un modelo de 14B te quedarás sin memoria.",
    cfg: { max_steps: 0, epochs: 3, lora_r: 64, lora_alpha: 64, use_rslora: true, quantization: "none", learning_rate: 1e-4 },
  },
  dpo: {
    label: "DPO (alineamiento)", desc: "Preferencias chosen/rejected",
    cuando: "Para enseñar qué respuesta se prefiere frente a otra: tono, formato, negarse a algo. Se aplica DESPUÉS de un SFT, no en lugar de él.",
    ojo: "Necesita un dataset de preferencias (prompt / chosen / rejected). Con uno de chat normal no funciona.",
    cfg: { method: "dpo", max_steps: 0, epochs: 1, learning_rate: 5e-6, lora_r: 16, lora_alpha: 32, dpo_beta: 0.1 },
  },
  grpo: {
    label: "GRPO (razonamiento)", desc: "Refuerzo con recompensas verificables",
    cuando: "Cuando puedes puntuar una respuesta con una regla objetiva: si el resultado es correcto, si cumple un formato, si tiene la longitud pedida. El modelo genera varias respuestas por pregunta y se refuerzan las que más puntúan.",
    ojo: "Es el más lento con diferencia porque genera varias respuestas por cada ejemplo. Con el motor vLLM va unas 9× más rápido.",
    cfg: { method: "grpo", max_steps: 100, learning_rate: 1e-5, lora_r: 16, lora_alpha: 32, batch_size: 4, grad_accum: 2, num_generations: 4, rewards: ["correctness", "format"], max_seq_len: 1024 },
  },
};

/** Estimación aproximada de VRAM en GB. */
function estimateVram(model: string, c: Cfg): number | null {
  const m = model.match(/(\d+(?:\.\d+)?)\s*[bB](?![a-z])/);
  if (!m) return null;
  const p = parseFloat(m[1]);
  const bytesPerParam = !c.use_lora ? 16 : c.quantization === "4bit" ? 0.6 : c.quantization === "8bit" ? 1.1 : 2.05;
  const weights = p * bytesPerParam;
  const act = (c.batch_size * c.max_seq_len / 2048) * (c.gradient_checkpointing ? 0.6 : 3) * Math.sqrt(p) * (c.method === "dpo" ? 2 : 1);
  // Unsloth: checkpointing con offload y kernels propios reducen mucho las activaciones
  return Math.round((weights + act * (c.backend === "unsloth" ? 0.4 : 1) + 1.2) * 10) / 10;
}

export default function Training() {
  const [jobs] = useData<Job[]>("/api/jobs", 4000);
  const [tab, setTab] = useState<"llm" | "sweep" | "embed" | "bench">("llm");
  // Sólo para el aviso de arriba: el historial completo vive en la pantalla Trabajos
  const running = (jobs ?? []).filter((j) => j.status === "running" || j.status === "queued");
  return (
    <>
      <div className="page-head">
        <div><h1>Entrenamiento</h1><div className="sub">Fine-tuning SFT, DPO o preentrenamiento continuo con LoRA / QLoRA / full</div></div>
      </div>
      <Help title="¿Cómo funciona el entrenamiento?">
        <p><b>Fine-tuning</b> es coger un modelo ya entrenado y ajustarlo con tus ejemplos. Aquí se deciden <b>dos cosas independientes</b>, y conviene no mezclarlas:</p>
        <p style={{ marginTop: 8 }}><b>1. Qué le enseñas</b> — es el «Método»:</p>
        <ul>
          <li><b>SFT</b>: le das preguntas con su respuesta buena y aprende a imitarlas. Es el 90 % de los casos y por donde se empieza.</li>
          <li><b>DPO / ORPO / KTO</b>: le das dos respuestas a la misma pregunta, una mejor que otra, y aprende a preferir. Sirve para tono y forma, no para conocimiento. Va después de un SFT.</li>
          <li><b>GRPO</b>: no le das la respuesta, le das una <i>forma de puntuarla</i> (¿el número es correcto?, ¿cumple el formato?). El modelo genera varias, se refuerzan las que más puntúan. Sólo sirve si puedes verificar el acierto con una regla.</li>
          <li><b>Preentrenamiento continuo</b>: le das texto suelto de tu dominio, sin preguntas ni respuestas, para que absorba vocabulario.</li>
        </ul>
        <p style={{ marginTop: 8 }}><b>2. Cómo tocas los pesos</b> — es el «Adaptador»:</p>
        <ul>
          <li><b>Full fine-tune</b>: modificas los miles de millones de parámetros del modelo. Lo mejor y lo más caro; un modelo de 8B no cabe en 24 GB así.</li>
          <li><b>LoRA</b>: no tocas el modelo. Le enganchas unas matrices pequeñas nuevas y entrenas <b>sólo esas</b>. Sale un «adaptador» de pocos MB que se usa junto al modelo original. Casi tan bueno y muchísimo más barato.</li>
          <li><b>QLoRA</b>: LoRA, y además el modelo base se carga comprimido a 4 bits para ocupar la cuarta parte. <b>Es lo que permite entrenar un 8B o un 14B en una GPU de 24 GB.</b> La compresión es sólo para entrenar; el adaptador que sale sirve igual.</li>
        </ul>
        <p style={{ marginTop: 8 }}>Los preajustes de arriba combinan las dos cosas: <b>Prueba rápida</b>, <b>QLoRA estándar</b> y <b>Alta calidad</b> son los tres <b>SFT</b> y sólo cambian el adaptador y la duración; <b>DPO</b> y <b>GRPO</b> sí cambian lo que aprende.</p>
        <p style={{ marginTop: 8 }}><b>Si no sabes cuál elegir</b>: pulsa <b>Prueba rápida</b>, comprueba en dos minutos que todo funciona, y luego <b>QLoRA estándar</b>, que es el bueno para casi todo.</p>
        <p style={{ marginTop: 8 }}>Pestañas: <b>LLM</b> entrena el modelo de lenguaje · <b>Barrido</b> prueba varias configuraciones y las ordena · <b>Embeddings</b> mejora la búsqueda del RAG · <b>Benchmarks</b> comprueba que el fine-tuning no le hizo olvidar capacidades generales.</p>
        <p style={{ marginTop: 8 }}>Al terminar, el adaptador aparece en <b>Modelos</b> listo para probar en el Playground o exportar a GGUF. El historial de lo ya lanzado, con sus curvas y la comparación entre entrenamientos, está en <b>Trabajos</b>.</p>
      </Help>
      {!!running.length && (
        <div className="card" style={{ borderColor: "var(--accent)" }}>
          <div className="row between">
            <div>
              <b>{running.length === 1 ? "Hay un trabajo en curso" : `Hay ${running.length} trabajos en curso`}</b>
              <div className="sub">{running.map((j) => j.name).join(", ")}</div>
            </div>
            <Link to="/jobs">Ver en Trabajos →</Link>
          </div>
          {running.map((j) => j.progress?.max_steps ? (
            <div key={j.id} style={{ marginTop: 6 }}>
              <Progress value={j.progress.step / j.progress.max_steps} />
              <div className="sub" style={{ fontSize: 11 }}>{j.name}: {j.progress.step}/{j.progress.max_steps} pasos</div>
            </div>
          ) : null)}
        </div>
      )}
      <div className="stack" style={{ gap: 0 }}>
        <div className="tabs">
          <button className={`tab ${tab === "llm" ? "active" : ""}`} onClick={() => setTab("llm")}>LLM</button>
          <button className={`tab ${tab === "sweep" ? "active" : ""}`} onClick={() => setTab("sweep")}>Barrido</button>
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
  // Unsloth por defecto cuando está instalado
  useEffect(() => { if (unslothOk) setC((p) => ({ ...p, backend: "unsloth", lora_dropout: 0 })); }, [unslothOk]);
  // vLLM sólo aplica a GRPO: si se cambia de método, se vuelve al motor por defecto
  useEffect(() => { setC((p) => (p.backend === "vllm" && p.method !== "grpo" ? { ...p, backend: unslothOk ? "unsloth" : "hf" } : p)); }, [c.method, unslothOk]);
  // vLLM no puede generar con pesos cuantizados (bitsandbytes): se entrena LoRA en 16 bits
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
    if (!c.dataset_id) return notify("Elige un dataset", "err");
    try {
      const body: any = { ...c, name: c.name || `${baseName.split("/").pop()}-${ds?.name}`, max_steps: c.max_steps || null, neftune_alpha: c.neftune_alpha || null,
        reward_regex: c.reward_regex || null, grpo_system_prompt: c.grpo_system_prompt || null };
      if (c.method === "grpo" && !c.rewards.length) return notify("Elige al menos una recompensa", "err");
      if (!body.base_model_id) delete body.base_model_id;
      const job = await api.post<Job>("/api/jobs/train", body);
      nav(`/training/${job.id}`);
    } catch (e: any) { notify(e.message, "err"); }
  };

  const methodMismatch = ds && ((["dpo", "orpo", "kto"].includes(c.method) && ds.format !== "preference") || (c.method === "pretrain" && ds.format === "preference") || (c.method === "grpo" && ds.format === "text"));
  const toggleReward = (r: string) => set("rewards", c.rewards.includes(r) ? c.rewards.filter((x) => x !== r) : [...c.rewards, r]);

  return (
    <div className="card stack" style={{ gap: 14 }}>
      <h2 style={{ margin: 0 }}>Nuevo entrenamiento</h2>
      <div className="row">
        {Object.entries(PRESETS).map(([k, p]) => (
          <button key={k} className={`sm ${preset === k ? "primary" : ""}`} title={p.desc} onClick={() => { setPreset(k); setC((prev) => ({ ...prev, ...DEFAULTS, name: prev.name, base_model: prev.base_model, base_model_id: prev.base_model_id, dataset_id: prev.dataset_id, backend: prev.backend, lora_dropout: prev.backend === "unsloth" ? 0 : DEFAULTS.lora_dropout, ...p.cfg })); }}>{p.label}{p.recomendado ? " ★" : ""}</button>
        ))}
      </div>

      {preset && (() => {
        const p = PRESETS[preset];
        const q = p.cfg.quantization ?? DEFAULTS.quantization;
        const duracion = p.cfg.max_steps ? `${p.cfg.max_steps} pasos` : `${p.cfg.epochs ?? DEFAULTS.epochs} época(s)`;
        return (
          <div className="preset-info">
            <div className="row between">
              <b>{p.label}{p.recomendado ? " · el recomendado" : ""}</b>
              <button className="sm ghost" onClick={() => setPreset(null)}>Ocultar</button>
            </div>
            <p>{p.cuando}</p>
            {p.ojo && <p className="aviso">⚠ {p.ojo}</p>}
            <div className="chips">
              <code className="chip">{(p.cfg.method ?? DEFAULTS.method).toUpperCase()}</code>
              <code className="chip">{q === "4bit" ? "QLoRA 4-bit" : q === "8bit" ? "LoRA 8-bit" : "LoRA 16-bit"}</code>
              <code className="chip">rango {p.cfg.lora_r ?? DEFAULTS.lora_r}</code>
              <code className="chip">lr {p.cfg.learning_rate ?? DEFAULTS.learning_rate}</code>
              <code className="chip">{duracion}</code>
            </div>
          </div>
        );
      })()}

      <fieldset>
        <legend>Modelo y datos</legend>
        <div className="grid g2">
          <Field label="Modelo base" hint="Repo de Hugging Face o un modelo registrado">
            <select value={c.base_model_id ? `id:${c.base_model_id}` : `hf:${c.base_model}`} onChange={(e) => {
              const v = e.target.value;
              if (v.startsWith("id:")) set("base_model_id", +v.slice(3));
              else if (v === "custom") prompt("Modelo base", { message: "Repo de Hugging Face (usuario/modelo) o ruta local a una carpeta de modelo.", placeholder: "Qwen/Qwen3-8B" }).then((r) => r && setC((p) => ({ ...p, base_model: r, base_model_id: null })));
              else setC((p) => ({ ...p, base_model: v.slice(3), base_model_id: null }));
            }}>
              {baseModels.length > 0 && <optgroup label="Registrados">{baseModels.map((m) => <option key={m.id} value={`id:${m.id}`}>{m.name}</option>)}</optgroup>}
              <optgroup label="Hugging Face">
                {!suggested?.some((s) => s.repo === c.base_model) && <option value={`hf:${c.base_model}`}>{c.base_model}</option>}
                {suggested?.map((s) => <option key={s.repo} value={`hf:${s.repo}`}>{s.repo} ({s.size})</option>)}
              </optgroup>
              <option value="custom">Otro…</option>
            </select>
          </Field>
          <Field label="Dataset">
            <select value={c.dataset_id} onChange={(e) => set("dataset_id", +e.target.value)}>
              <option value={0}>— elige —</option>
              {datasets?.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_rows} · {d.format})</option>)}
            </select>
          </Field>
          <Field label="Método">
            <select value={c.method} onChange={(e) => set("method", e.target.value)}>
              <option value="sft">SFT — ajuste supervisado (instrucciones/chat)</option>
              <option value="dpo">DPO — preferencias</option>
              <option value="orpo">ORPO — SFT + preferencias en un paso (sin modelo de referencia)</option>
              <option value="kto">KTO — preferencias sin emparejar (bueno/malo)</option>
              <option value="grpo">GRPO — refuerzo con recompensas (razonamiento, formato)</option>
              <option value="pretrain">Preentrenamiento continuo (texto)</option>
            </select>
          </Field>
          <Adv><Field label="Nombre del trabajo"><input value={c.name} onChange={(e) => set("name", e.target.value)} placeholder="auto" /></Field></Adv>
          <Field label="Motor de entrenamiento" hint={unslothOk ? "Unsloth: kernels optimizados, menos VRAM y más rápido" : "Instala Unsloth con backend/setup-unsloth.sh"}>
            <select value={c.backend} onChange={(e) => set("backend", e.target.value)}>
              <option value="unsloth" disabled={!unslothOk}>Unsloth (recomendado){unslothOk ? "" : " — no instalado"}</option>
              <option value="vllm" disabled={!vllmOk || c.method !== "grpo"}>vLLM (sólo GRPO; LoRA 16-bit){vllmOk ? "" : " — no instalado"}</option>
              <option value="hf">Hugging Face (TRL + PEFT)</option>
            </select>
          </Field>
        </div>
        <div className="sub" style={{ marginTop: 8, fontSize: 12.5 }}>{METHOD_HELP[c.method]}</div>
        {methodMismatch && <div className="error-box" style={{ marginTop: 10 }}>El método elegido no es compatible con el formato del dataset ({ds?.format}). {METHOD_NEEDS[c.method]}</div>}
      </fieldset>

      {c.method === "grpo" && (
        <fieldset>
          <legend>GRPO — recompensas</legend>
          <div className="sub" style={{ marginBottom: 8 }}>Por cada prompt el modelo genera {c.num_generations} respuestas; las que más recompensa obtienen se refuerzan. La respuesta del dataset se usa como referencia.</div>
          <div className="row" style={{ gap: 16 }}>
            <Check label="Corrección (vs. referencia)" checked={c.rewards.includes("correctness")} onChange={() => toggleReward("correctness")} />
            <Check label="Formato: razona en <think>" checked={c.rewards.includes("format")} onChange={() => toggleReward("format")} />
            <Check label={`Longitud ≤ ${c.reward_length_target} palabras`} checked={c.rewards.includes("length")} onChange={() => toggleReward("length")} />
            <Check label="Expresión regular" checked={c.rewards.includes("regex")} onChange={() => toggleReward("regex")} />
          </div>
          <div className="grid g3" style={{ marginTop: 10 }}>
            {/* Sólo se quedan a la vista los que dependen de la recompensa que hayas marcado */}
            {c.rewards.includes("length") && <Field label="Palabras objetivo"><input type="number" value={c.reward_length_target} onChange={num("reward_length_target")} /></Field>}
            {c.rewards.includes("regex") && <Field label="Regex que debe cumplir"><input value={c.reward_regex} onChange={(e) => set("reward_regex", e.target.value)} placeholder="<respuesta>.*</respuesta>" /></Field>}
            <Adv>
              <Field label="Generaciones por prompt" hint="Más respuestas por pregunta = mejor señal, pero mucho más lento"><input type="number" value={c.num_generations} onChange={num("num_generations")} /></Field>
              <Field label="Máx. tokens por respuesta"><input type="number" value={c.max_completion_length} onChange={num("max_completion_length")} /></Field>
              <Field label="Temperatura de muestreo" hint="Alta = respuestas variadas, que es lo que GRPO necesita para comparar"><input type="number" step="0.05" value={c.grpo_temperature} onChange={num("grpo_temperature")} /></Field>
              <Field label="β (KL)" hint="Cuánto se le deja alejarse del modelo original. 0 = sin freno y más rápido"><input type="number" step="0.01" value={c.grpo_beta} onChange={num("grpo_beta")} /></Field>
            </Adv>
          </div>
          <Adv>
            <Field label="System prompt para GRPO (opcional)" hint="p. ej. «Razona dentro de <think></think> y luego responde.»">
              <input value={c.grpo_system_prompt} onChange={(e) => set("grpo_system_prompt", e.target.value)} />
            </Field>
          </Adv>
          <SimpleNote>
            Genera {c.num_generations} respuestas por pregunta con temperatura {c.grpo_temperature} y hasta
            {" "}{c.max_completion_length} tokens, sin freno KL. Lo único que sueles querer cambiar son las
            recompensas de arriba.
          </SimpleNote>
        </fieldset>
      )}

      <fieldset>
        <legend>Adaptador</legend>
        <div className="grid g3">
          <Field label="Tipo">
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
            <Field label="Rango (r)"><input type="number" value={c.lora_r} disabled={!c.use_lora} onChange={num("lora_r")} /></Field>
            <Field label="Alpha"><input type="number" value={c.lora_alpha} disabled={!c.use_lora} onChange={num("lora_alpha")} /></Field>
          </Adv>
        </div>
      </fieldset>

      <fieldset>
        <legend>Hiperparámetros</legend>
        <div className="grid g3">
          <Field label="Épocas"><input type="number" step="0.5" value={c.epochs} onChange={num("epochs")} /></Field>
          <Adv><Field label="Pasos máx." hint="0 = usar épocas"><input type="number" value={c.max_steps} onChange={num("max_steps")} /></Field></Adv>
          <Field label="Learning rate"><input type="number" step="0.00001" value={c.learning_rate} onChange={num("learning_rate")} /></Field>
          <Adv>
            <Field label="Batch por GPU"><input type="number" value={c.batch_size} onChange={num("batch_size")} /></Field>
            <Field label="Acumulación de gradiente"><input type="number" value={c.grad_accum} onChange={num("grad_accum")} /></Field>
            <Field label="Longitud máx. (tokens)"><input type="number" step="256" value={c.max_seq_len} onChange={num("max_seq_len")} /></Field>
          </Adv>
        </div>
        <div className="row" style={{ marginTop: 10, gap: 18 }}>
          {c.method === "sft" && <Check label="Entrenar sólo en respuestas" checked={c.train_on_responses_only} onChange={(v) => set("train_on_responses_only", v)} />}
          <Check label="Fusionar LoRA al terminar" checked={c.merge_after} onChange={(v) => set("merge_after", v)} />
          <button className="ghost sm" onClick={() => setSimple(!simple)}>
            {simple ? "Mostrar todos los parámetros" : "Volver al modo simple"}
          </button>
        </div>
        <SimpleNote>
          Se entrena con LoRA r={c.lora_r}/α={c.lora_alpha}, batch {c.batch_size}×{c.grad_accum} y {c.max_seq_len} tokens
          de longitud: valores que caben de sobra en una 3090.
        </SimpleNote>
      </fieldset>

      {!simple && (
        <fieldset>
          <legend>Avanzado</legend>
          <div className="grid g3">
            <Field label="Scheduler">
              <select value={c.scheduler} onChange={(e) => set("scheduler", e.target.value)}>
                {["cosine", "linear", "constant", "constant_with_warmup", "cosine_with_restarts"].map((s) => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Warmup (proporción)"><input type="number" step="0.01" value={c.warmup_ratio} onChange={num("warmup_ratio")} /></Field>
            <Field label="Weight decay"><input type="number" step="0.01" value={c.weight_decay} onChange={num("weight_decay")} /></Field>
            <Field label="LoRA dropout"><input type="number" step="0.01" value={c.lora_dropout} onChange={num("lora_dropout")} /></Field>
            <Field label="Proporción de evaluación" hint="Separa un % para eval_loss"><input type="number" step="0.05" value={c.eval_ratio} onChange={num("eval_ratio")} /></Field>
            <Field label="Eval cada N pasos"><input type="number" value={c.eval_steps} onChange={num("eval_steps")} /></Field>
            <Field label="Log cada N pasos"><input type="number" value={c.logging_steps} onChange={num("logging_steps")} /></Field>
            <Field label="Checkpoint cada N pasos"><input type="number" value={c.save_steps} onChange={num("save_steps")} /></Field>
            <Field label="NEFTune α" hint="0 = desactivado; 5 es típico"><input type="number" value={c.neftune_alpha} onChange={num("neftune_alpha")} /></Field>
            {c.method === "orpo" && <Field label="ORPO β"><input type="number" step="0.05" value={c.orpo_beta} onChange={num("orpo_beta")} /></Field>}
            {(c.method === "dpo" || c.method === "kto") && <Field label="DPO β"><input type="number" step="0.05" value={c.dpo_beta} onChange={num("dpo_beta")} /></Field>}
            <Field label="Semilla"><input type="number" value={c.seed} onChange={num("seed")} /></Field>
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
          {vram !== null && <>VRAM estimada: <b style={{ color: vram > 23 ? "var(--err)" : "var(--text)" }}>~{vram} GB</b> · </>}
          Batch efectivo: <b>{c.batch_size * c.grad_accum}</b>
          {steps !== null && !c.max_steps && <> · ≈ <b>{steps}</b> pasos</>}
        </div>
        <button className="primary" onClick={submit} disabled={!c.dataset_id || !!methodMismatch}><Play size={14} /> Lanzar entrenamiento</button>
      </div>
    </div>
  );
}
