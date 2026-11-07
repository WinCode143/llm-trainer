import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCw } from "lucide-react";

type Props = { children: ReactNode; where?: string };
type State = { error: Error | null; info: string };

/** Without this, any render error leaves a blank screen with no clue about what happened.
  *  Here you at least see the message, the stack and a button to retry without reloading the whole app. */
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
          <h2 style={{ margin: 0 }}>This screen crashed</h2>
        </div>
        <p className="sub" style={{ marginTop: 8 }}>
          The error is in the interface, not in your data: nothing was lost. It usually happens when the server
          returns something shaped differently than expected.
        </p>
        <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 12, marginTop: 8 }}>{String(error.message || error)}</pre>
        {info && (
          <details style={{ marginTop: 8 }}>
            <summary className="sub">Technical details</summary>
            <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>{info.trim()}</pre>
          </details>
        )}
        <div className="row" style={{ marginTop: 10, gap: 8 }}>
          <button className="primary" onClick={() => this.setState({ error: null, info: "" })}>
            <RotateCw size={14} /> Retry
          </button>
          <button onClick={() => location.reload()}>Reload the page</button>
        </div>
      </div>
    );
  }
}
