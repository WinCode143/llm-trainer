import { AlertTriangle, CheckCircle2, Info, Trophy } from "lucide-react";

type Sev = "win" | "warn" | "info";
export type Finding = { sev: Sev; title: string; body: string };

const ICON = { win: Trophy, warn: AlertTriangle, info: Info };

/** Reads two or more training runs and says what to look at, instead of leaving a table of raw numbers.
  *  Everything comes from explicit rules: no model is giving its opinion. */
export function verdictForTraining(rows: {
  id: number; label: string; status: string;
  eval_loss?: number; last_loss?: number; train_loss?: number;
  runtime?: number; vram?: number;
  dataset?: string; dataset_rows?: number; steps?: number;
  evalCurve: { step: number; value: number }[];
  diffs: Record<string, any>;   // the parameters where this run differs from the rest
}[]): Finding[] {
  const out: Finding[] = [];
  const ok = rows.filter((r) => r.status === "completed");
  if (ok.length < 2) {
    out.push({ sev: "warn", title: "There aren't two completed runs to compare",
      body: "Some failed or were cancelled, so their numbers are incomplete and can't support a decision." });
    if (ok.length < 1) return out;
  }

  // ---- 1. Who wins, and on which metric ----
  const conEval = ok.filter((r) => r.eval_loss != null);
  const usingEval = conEval.length === ok.length && ok.length >= 2;
  const pool = usingEval ? conEval : ok.filter((r) => r.last_loss != null);
  const score = (r: typeof rows[0]) => (usingEval ? r.eval_loss! : r.last_loss!);

  if (pool.length >= 2) {
    const ranked = [...pool].sort((a, b) => score(a) - score(b));
    const best = ranked[0];
    const second = ranked[1];
    const gap = (score(second) - score(best)) / Math.abs(score(second));
    const metric = usingEval ? "eval loss" : "training loss";

    if (gap < 0.02) {
      out.push({ sev: "info", title: "Statistical tie",
        body: `“${best.label}” and “${second.label}” are within 2% on ${metric} (${score(best).toFixed(4)} vs ${score(second).toFixed(4)}). That gap is noise: keep whichever is cheaper to train or simpler to maintain, not the lowest number.` });
    } else {
      out.push({ sev: "win", title: `“${best.label}” wins`,
        body: `${metric} of ${score(best).toFixed(4)} vs ${score(second).toFixed(4)} for the runner-up: ${(gap * 100).toFixed(1)}% better.` });
    }
  }

  // ---- 2. The most important warning: without eval data, the loss misleads ----
  if (!usingEval) {
    out.push({ sev: "warn", title: "You're comparing training loss",
      body: conEval.length
        ? "Only some runs held out evaluation data, so they can't be compared on equal terms. Re-run them all with the same “eval split”."
        : "None of them held out data for evaluation. Training loss also goes down when the model just memorizes, so it can't tell learning from overfitting. Set “Eval split” to 0.1 and run them again." });
  }

  // ---- 3. Overfitting: eval loss goes up while training loss goes down ----
  for (const r of ok) {
    const c = r.evalCurve;
    if (c.length < 3) continue;
    const min = Math.min(...c.map((p) => p.value));
    const final = c[c.length - 1].value;
    if (final > min * 1.05) {
      const bestStep = c.find((p) => p.value === min)?.step;
      out.push({ sev: "warn", title: `“${r.label}” is overfitting`,
        body: `Its eval loss bottomed out at ${min.toFixed(4)} (step ${bestStep}) and ended at ${final.toFixed(4)}: from there on it stopped learning and started memorizing. Train for fewer epochs or lower the learning rate.` });
    }
  }

  // ---- 4. Different datasets: the comparison doesn't measure what you think ----
  const datasets = new Set(ok.map((r) => r.dataset).filter(Boolean));
  if (datasets.size > 1) {
    out.push({ sev: "warn", title: "They used different datasets",
      body: `There are ${datasets.size} datasets involved (${[...datasets].join(", ")}). Losses on different datasets aren't comparable: you're measuring which data is easier, not which configuration is better.` });
  }

  // ---- 5. Very different lengths in steps: not a level playing field either ----
  const stepCounts = ok.map((r) => r.steps).filter((s): s is number => !!s);
  if (stepCounts.length === ok.length && Math.max(...stepCounts) > Math.min(...stepCounts) * 1.5) {
    out.push({ sev: "warn", title: "They didn't train for the same length",
      body: `They range from ${Math.min(...stepCounts)} to ${Math.max(...stepCounts)} steps. The one that trains longer starts ahead, so the loss gap may be just that.` });
  }

  // ---- 6. Cost: if the winner costs much more, say so ----
  if (pool.length >= 2) {
    const ranked = [...pool].sort((a, b) => score(a) - score(b));
    const [best, second] = ranked;
    if (best.runtime && second.runtime && best.runtime > second.runtime * 1.5) {
      const times = (best.runtime / second.runtime).toFixed(1);
      out.push({ sev: "info", title: "The winner is expensive",
        body: `“${best.label}” took ${times}× longer than “${second.label}”. If the improvement is small, it may not be worth it.` });
    }
    const vrams = ok.map((r) => r.vram).filter((v): v is number => !!v);
    if (vrams.length >= 2 && Math.max(...vrams) > 22) {
      out.push({ sev: "info", title: "Close to the VRAM limit",
        body: `The highest peak was ${Math.max(...vrams)} GB. With another application using the GPU that becomes an OOM: lower the batch size or the maximum length.` });
    }
  }

  // ---- 7. Which parameter was responsible ----
  const changed = [...new Set(ok.flatMap((r) => Object.keys(r.diffs)))];
  if (changed.length === 0) {
    out.push({ sev: "info", title: "Identical configurations",
      body: "They don't differ in any hyperparameter. Whatever you see between them is the normal run-to-run variation, which is exactly what tells you how much noise there is." });
  } else if (changed.length === 1) {
    out.push({ sev: "info", title: `The only difference is “${changed[0]}”`,
      body: "Since only one parameter changed, the difference in results can be attributed to it with reasonable confidence." });
  } else {
    out.push({ sev: "warn", title: `${changed.length} parameters change at once`,
      body: `They differ in ${changed.join(", ")}. With several changes together you can't tell which one caused the improvement. To attribute it, change one at a time or use Training → Sweep.` });
  }

  // ---- 8. What to do next ----
  out.push({ sev: "info", title: "Next step",
    body: "The loss measures how well the model predicts your dataset's text, not how useful it is. Before you trust the winner, run it through Evaluation (with your real questions) and Benchmarks, which is where you find out whether fine-tuning made it forget general skills." });

  return out;
}

export function VerdictPanel({ findings }: { findings: Finding[] }) {
  if (!findings.length) return null;
  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>What this comparison says</h3>
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
        <CheckCircle2 size={11} /> This summary comes from fixed rules over your numbers; no model writes it.
      </div>
    </div>
  );
}
