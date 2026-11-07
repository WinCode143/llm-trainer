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

/** Canvas node: a header coloured by type and a summary of its settings. */
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
    case "rag": return `collection ${c.collection_id ?? "—"} · k=${c.k ?? 4}`;
    case "agent": return `agent ${c.agent_id ?? "—"}`;
    case "router": return c.mode === "llm" ? "the model decides" : (c.branches ?? []).map((b: any) => b.when).join(" · ").slice(0, 60);
    case "approval": return (c.message || "").slice(0, 60);
    case "output": return (c.template || "").slice(0, 60);
    case "input": return (c.fields ?? []).map((f: any) => f.key).join(", ") || "input";
    case "foreach": return `each ${c.items_key ?? "?"} · ${c.mode ?? "llm"}${(c.parallel ?? 1) > 1 ? ` ×${c.parallel}` : ""}`;
    case "subflow": return `flow ${c.flow_id ?? "—"}`;
    case "code": return (c.code || "").split("\n")[0].slice(0, 60) || "no code";
    default: return "";
  }
}

export function FlowCanvas({ spec, onChange, selected, onSelect, running, done, types, tokens }: {
  spec: Spec; onChange: (s: Spec) => void; selected: string | null; onSelect: (id: string | null) => void;
  running?: string | null; done?: string[]; types: Record<string, any>;
  tokens?: Record<string, string>;   // text streaming in live, per node
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

/** Lays nodes out in layers by their distance from the input.
  *  Avoids the tangle you get when you add loose nodes and connect them afterwards. */
export function autoLayout(spec: Spec): Spec {
  const indegree = new Map<string, number>();
  spec.nodes.forEach((n) => indegree.set(n.id, 0));
  spec.edges.forEach((e) => indegree.set(e.to, (indegree.get(e.to) ?? 0) + 1));

  const level = new Map<string, number>();
  let queue = spec.nodes.filter((n) => (indegree.get(n.id) ?? 0) === 0).map((n) => n.id);
  if (!queue.length && spec.nodes.length) queue = [spec.nodes[0].id];   // everything is in a cycle: start with the first
  queue.forEach((id) => level.set(id, 0));

  const pending = new Map(indegree);
  while (queue.length) {
    const id = queue.shift()!;
    for (const e of spec.edges.filter((x) => x.from === id)) {
      level.set(e.to, Math.max(level.get(e.to) ?? 0, (level.get(id) ?? 0) + 1));
      pending.set(e.to, (pending.get(e.to) ?? 1) - 1);
      if ((pending.get(e.to) ?? 0) <= 0) queue.push(e.to);
    }
  }
  // Nodes in a cycle get no level: they go at the end
  const maxLevel = Math.max(0, ...[...level.values()]);
  spec.nodes.forEach((n) => { if (!level.has(n.id)) level.set(n.id, maxLevel + 1); });

  const byLevel = new Map<number, string[]>();
  spec.nodes.forEach((n) => {
    const l = level.get(n.id) ?? 0;
    byLevel.set(l, [...(byLevel.get(l) ?? []), n.id]);
  });
  return {
    ...spec,
    nodes: spec.nodes.map((n) => {
      const l = level.get(n.id) ?? 0;
      const row = byLevel.get(l)!;
      const i = row.indexOf(n.id);
      return { ...n, x: 60 + l * 260, y: 120 + (i - (row.length - 1) / 2) * 150 };
    }),
  };
}
