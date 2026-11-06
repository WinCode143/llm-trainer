import { AlertTriangle, CheckCircle2, Info, Trophy } from "lucide-react";

type Sev = "win" | "warn" | "info";
export type Finding = { sev: Sev; title: string; body: string };

const ICON = { win: Trophy, warn: AlertTriangle, info: Info };

/** Lee dos o más entrenamientos y dice qué mirar, en vez de dejar una tabla de números sin interpretar.
 *  Todo se calcula con reglas explícitas: no hay ningún modelo opinando. */
export function verdictForTraining(rows: {
  id: number; label: string; status: string;
  eval_loss?: number; last_loss?: number; train_loss?: number;
  runtime?: number; vram?: number;
  dataset?: string; dataset_rows?: number; steps?: number;
  evalCurve: { step: number; value: number }[];
  diffs: Record<string, any>;   // parámetros en los que este run se diferencia del resto
}[]): Finding[] {
  const out: Finding[] = [];
  const ok = rows.filter((r) => r.status === "completed");
  if (ok.length < 2) {
    out.push({ sev: "warn", title: "No hay dos entrenamientos completos que comparar",
      body: "Alguno falló o se canceló, así que sus números están a medias y no sirven para decidir." });
    if (ok.length < 1) return out;
  }

  // ---- 1. Quién gana, y con qué métrica ----
  const conEval = ok.filter((r) => r.eval_loss != null);
  const usandoEval = conEval.length === ok.length && ok.length >= 2;
  const pool = usandoEval ? conEval : ok.filter((r) => r.last_loss != null);
  const valor = (r: typeof rows[0]) => (usandoEval ? r.eval_loss! : r.last_loss!);

  if (pool.length >= 2) {
    const orden = [...pool].sort((a, b) => valor(a) - valor(b));
    const mejor = orden[0];
    const segundo = orden[1];
    const brecha = (valor(segundo) - valor(mejor)) / Math.abs(valor(segundo));
    const metrica = usandoEval ? "eval loss" : "loss de entrenamiento";

    if (brecha < 0.02) {
      out.push({ sev: "info", title: "Empate técnico",
        body: `«${mejor.label}» y «${segundo.label}» quedan a menos del 2 % en ${metrica} (${valor(mejor).toFixed(4)} frente a ${valor(segundo).toFixed(4)}). Esa diferencia es ruido: quédate con el que sea más barato de entrenar o más simple de mantener, no con el número más bajo.` });
    } else {
      out.push({ sev: "win", title: `Gana «${mejor.label}»`,
        body: `${metrica} de ${valor(mejor).toFixed(4)} frente a ${valor(segundo).toFixed(4)} del siguiente: un ${(brecha * 100).toFixed(1)} % mejor.` });
    }
  }

  // ---- 2. La advertencia más importante: sin eval, la loss engaña ----
  if (!usandoEval) {
    out.push({ sev: "warn", title: "Estás comparando loss de entrenamiento",
      body: conEval.length
        ? "Sólo algunos entrenamientos reservaron datos de evaluación, así que no se pueden enfrentar en igualdad. Relanza con la misma «proporción de evaluación» en todos."
        : "Ninguno reservó datos para evaluar. La loss de entrenamiento baja también cuando el modelo se limita a memorizar, así que no distingue aprender de sobreajustar. Pon «Proporción de evaluación» en 0,1 y vuelve a lanzarlos." });
  }

  // ---- 3. Sobreajuste: la eval sube mientras la de entrenamiento baja ----
  for (const r of ok) {
    const c = r.evalCurve;
    if (c.length < 3) continue;
    const min = Math.min(...c.map((p) => p.value));
    const fin = c[c.length - 1].value;
    if (fin > min * 1.05) {
      const paso = c.find((p) => p.value === min)?.step;
      out.push({ sev: "warn", title: `«${r.label}» se sobreajusta`,
        body: `Su eval loss tocó fondo en ${min.toFixed(4)} (paso ${paso}) y terminó en ${fin.toFixed(4)}: a partir de ahí dejó de aprender y empezó a memorizar. Entrena menos épocas o baja el learning rate.` });
    }
  }

  // ---- 4. Datasets distintos: la comparación no mide lo que crees ----
  const datasets = new Set(ok.map((r) => r.dataset).filter(Boolean));
  if (datasets.size > 1) {
    out.push({ sev: "warn", title: "Usaron datasets distintos",
      body: `Hay ${datasets.size} datasets en juego (${[...datasets].join(", ")}). Las loss de datasets distintos no son comparables entre sí: estás midiendo qué datos son más fáciles, no qué configuración es mejor.` });
  }

  // ---- 5. Duraciones muy dispares en pasos: tampoco es igualdad de condiciones ----
  const pasos = ok.map((r) => r.steps).filter((s): s is number => !!s);
  if (pasos.length === ok.length && Math.max(...pasos) > Math.min(...pasos) * 1.5) {
    out.push({ sev: "warn", title: "No entrenaron lo mismo",
      body: `Van de ${Math.min(...pasos)} a ${Math.max(...pasos)} pasos. El que más entrena parte con ventaja, así que la diferencia de loss puede ser sólo eso.` });
  }

  // ---- 6. Coste: si el ganador cuesta mucho más, decirlo ----
  if (pool.length >= 2) {
    const orden = [...pool].sort((a, b) => valor(a) - valor(b));
    const [mejor, segundo] = orden;
    if (mejor.runtime && segundo.runtime && mejor.runtime > segundo.runtime * 1.5) {
      const veces = (mejor.runtime / segundo.runtime).toFixed(1);
      out.push({ sev: "info", title: "El ganador sale caro",
        body: `«${mejor.label}» tardó ${veces}× más que «${segundo.label}». Si la mejora es pequeña, puede no compensar.` });
    }
    const vrams = ok.map((r) => r.vram).filter((v): v is number => !!v);
    if (vrams.length >= 2 && Math.max(...vrams) > 22) {
      out.push({ sev: "info", title: "Rozando el límite de VRAM",
        body: `El pico más alto fue ${Math.max(...vrams)} GB. Con otra aplicación usando la GPU eso se va a OOM: baja el batch o la longitud máxima.` });
    }
  }

  // ---- 7. Qué parámetro fue el responsable ----
  const claves = [...new Set(ok.flatMap((r) => Object.keys(r.diffs)))];
  if (claves.length === 0) {
    out.push({ sev: "info", title: "Configuraciones idénticas",
      body: "No se diferencian en ningún hiperparámetro. Lo que veas entre ellos es la variación normal entre dos ejecuciones, útil justo para saber cuánto ruido hay." });
  } else if (claves.length === 1) {
    out.push({ sev: "info", title: `La única diferencia es «${claves[0]}»`,
      body: "Al cambiar sólo un parámetro, la diferencia de resultados se le puede atribuir a él con bastante confianza." });
  } else {
    out.push({ sev: "warn", title: `Cambian ${claves.length} parámetros a la vez`,
      body: `Se diferencian en ${claves.join(", ")}. Con varios cambios juntos no se sabe cuál causó la mejora. Para atribuirla, cambia uno cada vez o usa Entrenamiento → Barrido.` });
  }

  // ---- 8. Qué hacer ahora ----
  out.push({ sev: "info", title: "El siguiente paso",
    body: "La loss mide lo bien que predice el texto de tu dataset, no lo útil que es el modelo. Antes de dar por bueno al ganador, pásalo por Evaluación (con tus preguntas reales) y por Benchmarks, que es donde se detecta si el fine-tuning le hizo olvidar capacidades generales." });

  return out;
}

export function VerdictPanel({ findings }: { findings: Finding[] }) {
  if (!findings.length) return null;
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Qué dice esta comparación</h3>
      <div className="stack" style={{ gap: 8 }}>
        {findings.map((f, i) => {
          const Icon = ICON[f.sev];
          return (
            <div key={i} className={`finding ${f.sev}`}>
              <Icon size={15} />
              <div><b>{f.title}</b><div className="sub">{f.body}</div></div>
            </div>
          );
        })}
      </div>
      <div className="sub" style={{ marginTop: 10, fontSize: 11 }}>
        <CheckCircle2 size={11} /> Este resumen se calcula con reglas fijas sobre tus números; no lo redacta ningún modelo.
      </div>
    </div>
  );
}
