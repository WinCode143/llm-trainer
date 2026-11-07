import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy } from "lucide-react";

/** Safe Markdown (no raw HTML) with tables, lists and copyable code blocks. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
        pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
        a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
      }}>{text}</ReactMarkdown>
    </div>
  );
}

function CodeBlock({ children }: { children: React.ReactNode }) {
  const [done, setDone] = useState(false);
  const copy = (e: React.MouseEvent) => {
    const code = (e.currentTarget.parentElement?.querySelector("code")?.textContent) ?? "";
    navigator.clipboard?.writeText(code).then(() => { setDone(true); setTimeout(() => setDone(false), 1500); });
  };
  return (
    <div className="codeblock">
      <button className="ghost sm copy" onClick={copy} title="Copy">{done ? <Check size={12} /> : <Copy size={12} />}</button>
      <pre>{children}</pre>
    </div>
  );
}

/** Assistant reply: splits out <think>…</think> blocks (reasoning models) and renders Markdown. */
export function AssistantText({ text, streaming }: { text: string; streaming?: boolean }) {
  const m = text.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
  if (!m) return <Markdown text={text} />;
  const thinking = m[1].trim();
  const closed = m[2] === "</think>";
  const rest = m[3].replace(/^\s+/, "");
  return (
    <>
      {thinking && (
        <details className="think" open={streaming && !closed}>
          <summary>{closed ? "Reasoning" : "Thinking…"}</summary>
          <div style={{ whiteSpace: "pre-wrap" }}>{thinking}</div>
        </details>
      )}
      <Markdown text={rest} />
    </>
  );
}
