import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCw } from "lucide-react";

type Props = { children: ReactNode; where?: string };
type State = { error: Error | null; info: string };

/** Sin esto, cualquier error de render deja la pantalla en blanco y sin pista de qué pasó.
 *  Aquí al menos se ve el mensaje, la pila y un botón para reintentar sin recargar toda la app. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: "" };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[render]", this.props.where ?? "", error, info.componentStack);
    this.setState({ info: info.componentStack ?? "" });
  }

  render() {
    const { error, info } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="card" style={{ borderColor: "var(--err)" }}>
        <div className="row" style={{ gap: 8, alignItems: "center" }}>
          <AlertTriangle size={18} style={{ color: "var(--err)" }} />
          <h2 style={{ margin: 0 }}>Esta pantalla ha fallado</h2>
        </div>
        <p className="sub" style={{ marginTop: 8 }}>
          El error es de la interfaz, no de tus datos: nada se ha perdido. Suele pasar cuando el servidor
          devuelve algo con una forma distinta a la esperada.
        </p>
        <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 12, marginTop: 8 }}>{String(error.message || error)}</pre>
        {info && (
          <details style={{ marginTop: 8 }}>
            <summary className="sub">Detalle técnico</summary>
            <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>{info.trim()}</pre>
          </details>
        )}
        <div className="row" style={{ marginTop: 10, gap: 8 }}>
          <button className="primary" onClick={() => this.setState({ error: null, info: "" })}>
            <RotateCw size={14} /> Reintentar
          </button>
          <button onClick={() => location.reload()}>Recargar la página</button>
        </div>
      </div>
    );
  }
}
