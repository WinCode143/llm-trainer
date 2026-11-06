import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const PALETTE = ["#7c8cff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#39c5cf", "#db61a2", "#e3b341", "#6cb6ff", "#8ddb8c"];

type Node = { id: string; name: string; type: string; degree: number; community: number | null; desc: string; sources: string[] };
type Link = { source: any; target: any; relation: string; weight: number };

/** Grafo de entidades interactivo (canvas). Resalta los vecinos al pasar el ratón. */
export function GraphView({ nodes, links, highlight, height = 460 }: { nodes: Node[]; links: Link[]; highlight?: string[]; height?: number }) {
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  const [hover, setHover] = useState<Node | null>(null);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  const data = useMemo(() => ({ nodes: nodes.map((n) => ({ ...n })), links: links.map((l) => ({ ...l })) }), [nodes, links]);
  const hi = useMemo(() => new Set((highlight ?? []).map((h) => h.toLowerCase())), [highlight]);
  const near = useMemo(() => {
    if (!hover) return new Set<string>();
    const s = new Set<string>([hover.id]);
    links.forEach((l) => {
      const a = typeof l.source === "object" ? l.source.id : l.source, b = typeof l.target === "object" ? l.target.id : l.target;
      if (a === hover.id) s.add(b); else if (b === hover.id) s.add(a);
    });
    return s;
  }, [hover, links]);

  return (
    <div ref={box} style={{ position: "relative" }}>
      <ForceGraph2D
        graphData={data} width={w} height={height} backgroundColor="transparent"
        nodeRelSize={4} cooldownTicks={120} onNodeHover={(n: any) => setHover(n ?? null)}
        linkColor={(l: any) => (hover && (near.has(l.source.id) && near.has(l.target.id)) ? "#8b93a7" : "rgba(139,147,167,.25)")}
        linkDirectionalArrowLength={3} linkDirectionalArrowRelPos={1}
        linkLabel={(l: any) => l.relation}
        nodeCanvasObject={(n: any, ctx, scale) => {
          const dim = hover && !near.has(n.id);
          const color = hi.has(n.name.toLowerCase()) ? "#f85149" : PALETTE[(n.community ?? 0) % PALETTE.length];
          const r = 2 + Math.min(6, Math.sqrt(n.degree || 1));
          ctx.globalAlpha = dim ? 0.15 : 1;
          ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
          if (scale > 1.2 || n.degree > 4 || hi.has(n.name.toLowerCase())) {
            ctx.font = `${Math.max(3, 10 / scale + 2)}px Inter, sans-serif`;
            ctx.fillStyle = "#e4e7ee"; ctx.textAlign = "center";
            ctx.fillText(n.name.slice(0, 28), n.x, n.y - r - 2);
          }
          ctx.globalAlpha = 1;
        }}
      />
      {hover && (
        <div className="source" style={{ position: "absolute", bottom: 8, left: 8, right: 8, fontSize: 12, pointerEvents: "none" }}>
          <b>{hover.name}</b> <span className="badge">{hover.type}</span> <span className="sub">· {hover.degree} conexiones · {hover.sources.join(", ").slice(0, 60)}</span>
          {hover.desc && <div className="sub">{hover.desc}</div>}
        </div>
      )}
    </div>
  );
}
