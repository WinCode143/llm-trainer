import { useCallback, useEffect, useMemo } from "react";
import {
  addEdge, Background, Controls, type Edge, Handle, MarkerType, MiniMap, type Node, Position,
  ReactFlow, ReactFlowProvider, useEdgesState, useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

export type FlowNode = { id: string; type: string; label?: string; config?: any; x?: number; y?: number };
export type FlowEdge = { from: string; to: string; when?: string | null };
export type Spec = { nodes: FlowNode[]; edges: FlowEdge[] };

const COLORS: Record<string, string> = {
  input: "#3fb950", llm: "#7c8cff", tool: "#d29922", rag: "#39c5cf", agent: "#a371f7",
  router: "#f85149", approval: "#db61a2", output: "#3fb950",
  foreach: "#e3b341", subflow: "#8b949e", code: "#79c0ff",
};

/** Nodo del lienzo: cabecera con color por tipo y un resumen de su configuración. */
function NodeBox({ data, selected }: any) {
  const color = COLORS[data.ntype] ?? "#7c8cff";
  return (
    <div className={`flownode ${selected ? "sel" : ""} ${data.running ? "running" : ""} ${data.done ? "done" : ""}`}
      style={{ borderColor: selected ? color : undefined }}>
      <Handle type="target" position={Position.Left} />
      <div className="flownode-head" style={{ background: `color-mix(in srgb, ${color} 22%, transparent)` }}>
        <span className="dot" style={{ background: color }} />
        <b>{data.label}</b>
        <span className="sub" style={{ marginLeft: "auto", fontSize: 10 }}>{data.typeLabel}</span>
      </div>
      {data.live
        ? <div className="flownode-body live">{data.live.slice(-180)}<span className="caret" /></div>
        : data.summary && <div className="flownode-body">{data.summary}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { box: NodeBox };

function summarize(n: FlowNode): string {
  const c = n.config || {};
  switch (n.type) {
    case "llm": return (c.prompt || "").slice(0, 70) || "sin prompt";
    case "tool": return `${c.tool_id ?? "—"}`;
    case "rag": return `colección ${c.collection_id ?? "—"} · k=${c.k ?? 4}`;
    case "agent": return `agente ${c.agent_id ?? "—"}`;
    case "router": return c.mode === "llm" ? "decide el modelo" : (c.branches ?? []).map((b: any) => b.when).join(" · ").slice(0, 60);
    case "approval": return (c.message || "").slice(0, 60);
    case "output": return (c.template || "").slice(0, 60);
    case "input": return (c.fields ?? []).map((f: any) => f.key).join(", ") || "entrada";
    case "foreach": return `cada ${c.items_key ?? "?"} · ${c.mode ?? "llm"}${(c.parallel ?? 1) > 1 ? ` ×${c.parallel}` : ""}`;
    case "subflow": return `flujo ${c.flow_id ?? "—"}`;
    case "code": return (c.code || "").split("\n")[0].slice(0, 60) || "sin código";
    default: return "";
  }
}

export function FlowCanvas({ spec, onChange, selected, onSelect, running, done, types, tokens }: {
  spec: Spec; onChange: (s: Spec) => void; selected: string | null; onSelect: (id: string | null) => void;
  running?: string | null; done?: string[]; types: Record<string, any>;
  tokens?: Record<string, string>;   // texto que va llegando en vivo, por nodo
}) {
  const toNodes = useCallback((): Node[] => spec.nodes.map((n) => ({
    id: n.id, type: "box", position: { x: n.x ?? 100, y: n.y ?? 100 },
    data: { label: n.label || n.id, ntype: n.type, typeLabel: types[n.type]?.label ?? n.type,
            summary: summarize(n), running: running === n.id, done: (done ?? []).includes(n.id),
            live: tokens?.[n.id] },
  })), [spec.nodes, running, done, types, tokens]);

  const toEdges = useCallback((): Edge[] => spec.edges.map((e, i) => ({
    id: `e${i}-${e.from}-${e.to}`, source: e.from, target: e.to, label: e.when || undefined,
    animated: running != null, markerEnd: { type: MarkerType.ArrowClosed },
    style: { stroke: "var(--muted)" }, labelStyle: { fill: "var(--text)", fontSize: 11 },
    labelBgStyle: { fill: "var(--panel)" },
  })), [spec.edges, running]);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(toNodes());
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(toEdges());
  useEffect(() => { setNodes(toNodes()); }, [toNodes, setNodes]);
  useEffect(() => { setEdges(toEdges()); }, [toEdges, setEdges]);

  const commitPositions = useCallback((ns: Node[]) => {
    onChange({ ...spec, nodes: spec.nodes.map((n) => {
      const f = ns.find((x) => x.id === n.id);
      return f ? { ...n, x: Math.round(f.position.x), y: Math.round(f.position.y) } : n;
    }) });
  }, [spec, onChange]);

  const onConnect = useCallback((c: any) => {
    if (c.source === c.target) return;
    setEdges((eds) => addEdge(c, eds));
    onChange({ ...spec, edges: [...spec.edges, { from: c.source, to: c.target, when: null }] });
  }, [spec, onChange, setEdges]);

  const onEdgesDelete = useCallback((eds: Edge[]) => {
    const dead = new Set(eds.map((e) => `${e.source}->${e.target}`));
    onChange({ ...spec, edges: spec.edges.filter((e) => !dead.has(`${e.from}->${e.to}`)) });
  }, [spec, onChange]);

  const onNodesDelete = useCallback((ns: Node[]) => {
    const dead = new Set(ns.map((n) => n.id));
    onChange({ nodes: spec.nodes.filter((n) => !dead.has(n.id)),
               edges: spec.edges.filter((e) => !dead.has(e.from) && !dead.has(e.to)) });
  }, [spec, onChange]);

  const mini = useMemo(() => ({ nodeColor: (n: Node) => COLORS[(n.data as any).ntype] ?? "#7c8cff" }), []);

  return (
    <div className="flowcanvas">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes} edges={edges} nodeTypes={nodeTypes}
          onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
          onNodeDragStop={(_, __, all) => commitPositions(all ?? nodes)}
          onEdgesDelete={onEdgesDelete} onNodesDelete={onNodesDelete}
          onNodeClick={(_, n) => onSelect(n.id)} onPaneClick={() => onSelect(null)}
          selectionOnDrag panOnDrag={[1, 2]} fitView proOptions={{ hideAttribution: true }}
          defaultEdgeOptions={{ markerEnd: { type: MarkerType.ArrowClosed } }}
        >
          <Background color="var(--border)" gap={18} />
          <Controls showInteractive={false} />
          <MiniMap {...mini} pannable style={{ background: "var(--panel)" }} />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}

/** Coloca los nodos por capas según a qué distancia están de la entrada.
 *  Evita el amasijo que queda cuando añades nodos sueltos y los conectas después. */
export function autoLayout(spec: Spec): Spec {
  const entrantes = new Map<string, number>();
  spec.nodes.forEach((n) => entrantes.set(n.id, 0));
  spec.edges.forEach((e) => entrantes.set(e.to, (entrantes.get(e.to) ?? 0) + 1));

  const nivel = new Map<string, number>();
  let cola = spec.nodes.filter((n) => (entrantes.get(n.id) ?? 0) === 0).map((n) => n.id);
  if (!cola.length && spec.nodes.length) cola = [spec.nodes[0].id];   // todo en ciclo: empezamos por el primero
  cola.forEach((id) => nivel.set(id, 0));

  const pendientes = new Map(entrantes);
  while (cola.length) {
    const id = cola.shift()!;
    for (const e of spec.edges.filter((x) => x.from === id)) {
      nivel.set(e.to, Math.max(nivel.get(e.to) ?? 0, (nivel.get(id) ?? 0) + 1));
      pendientes.set(e.to, (pendientes.get(e.to) ?? 1) - 1);
      if ((pendientes.get(e.to) ?? 0) <= 0) cola.push(e.to);
    }
  }
  // Los nodos de un ciclo no reciben nivel: se dejan al final
  const maxNivel = Math.max(0, ...[...nivel.values()]);
  spec.nodes.forEach((n) => { if (!nivel.has(n.id)) nivel.set(n.id, maxNivel + 1); });

  const porNivel = new Map<number, string[]>();
  spec.nodes.forEach((n) => {
    const l = nivel.get(n.id) ?? 0;
    porNivel.set(l, [...(porNivel.get(l) ?? []), n.id]);
  });
  return {
    ...spec,
    nodes: spec.nodes.map((n) => {
      const l = nivel.get(n.id) ?? 0;
      const fila = porNivel.get(l)!;
      const i = fila.indexOf(n.id);
      return { ...n, x: 60 + l * 260, y: 120 + (i - (fila.length - 1) / 2) * 150 };
    }),
  };
}
