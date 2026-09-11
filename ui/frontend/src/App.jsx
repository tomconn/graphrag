import { useEffect, useRef, useState } from "react";

const STEP_ORDER = ["route", "rewrite", "retrieve", "traverse", "synthesize"];

function summarizeDetail(detail) {
  if (detail == null) return "";
  if (typeof detail === "string") return detail;
  const s = JSON.stringify(detail);
  return s.length > 180 ? s.slice(0, 180) + "…" : s;
}

function citationLabel(c) {
  // Regulatory: "¶27"; code: "path#Symbol[:lines]"; else the section path.
  const where = c.clause
    ? `¶${c.clause}`
    : c.code_ref || c.section || "";
  return where ? `${c.title} — ${where}` : c.title;
}

export default function App() {
  const [messages, setMessages] = useState([]); // {role, text, citations?, error?}
  const [steps, setSteps] = useState([]); // {step, detail}
  const [stepsOpen, setStepsOpen] = useState(false);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("hybrid"); // retrieval_mode sent to /api/chat
  const [busy, setBusy] = useState(false);
  const [drawer, setDrawer] = useState(null); // {title, content, loading}
  const listRef = useRef(null);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  function patchLastAssistant(patch) {
    setMessages((m) => {
      const copy = [...m];
      copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch };
      return copy;
    });
  }

  function handleEvent(payload) {
    let ev;
    try {
      ev = JSON.parse(payload);
    } catch {
      return;
    }
    if (ev.type === "step") {
      setSteps((s) => [...s, { step: ev.step, detail: ev.detail }]);
    } else if (ev.type === "token") {
      patchLastAssistant({});
      setMessages((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = { ...last, text: last.text + (ev.text || "") };
        return copy;
      });
    } else if (ev.type === "done") {
      patchLastAssistant({ text: ev.answer, citations: ev.citations || [] });
    } else if (ev.type === "error") {
      patchLastAssistant({ error: ev.message || "unknown error" });
    }
  }

  async function send(e) {
    e.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);
    setSteps([]);
    setMessages((m) => [
      ...m,
      { role: "user", text: message },
      { role: "assistant", text: "", citations: null },
    ]);
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, retrieval_mode: mode }),
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      // SSE over a POST body: parse the ReadableStream manually.
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const event = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of event.split("\n")) {
            if (line.startsWith("data:")) handleEvent(line.slice(5).trim());
          }
        }
      }
    } catch (err) {
      patchLastAssistant({ error: String(err.message || err) });
    } finally {
      setBusy(false);
    }
  }

  async function openCitation(c) {
    const params = new URLSearchParams({ path: c.source_path });
    if (c.section) params.set("section", c.section);
    setDrawer({ title: citationLabel(c), content: "", loading: true });
    try {
      const resp = await fetch(`/api/source?${params}`);
      const body = await resp.json();
      if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
      setDrawer({ title: citationLabel(c), content: body.content, loading: false });
    } catch (err) {
      setDrawer({ title: citationLabel(c), content: "Failed to load source: " + err, loading: false });
    }
  }

  return (
    <div className="app">
      <header>
        <h1>GraphRAG</h1>
        <label>
          Mode{" "}
          <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={busy}>
            <option value="hybrid">hybrid</option>
            <option value="vector">vector</option>
          </select>
        </label>
      </header>

      <div className="steps-bar">
        <button className="link" onClick={() => setStepsOpen((o) => !o)}>
          {stepsOpen ? "▾" : "▸"} Retrieval steps ({steps.length})
        </button>
        {stepsOpen && (
          <ul className="steps">
            {steps.map((s, i) => (
              <li key={i}>
                <code>{s.step}</code> {summarizeDetail(s.detail)}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="messages" ref={listRef}>
        {messages.length === 0 && <p className="hint">Ask a question across the corpus…</p>}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              {m.role === "assistant" ? m.text || "…" : m.text}
              {m.error && <div className="error">⚠ {m.error}</div>}
              {m.role === "assistant" && m.citations && m.citations.length > 0 && (
                <div className="citations">
                  {m.citations.map((c, j) => (
                    <button key={j} className="chip" onClick={() => openCitation(c)}>
                      {citationLabel(c)}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <form className="composer" onSubmit={send}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={busy ? "Waiting for the agent…" : "Ask about CPS 234, OWASP, our architecture…"}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>

      {drawer && (
        <aside className="drawer">
          <div className="drawer-head">
            <span>{drawer.title}</span>
            <button className="link" onClick={() => setDrawer(null)}>
              Close ✕
            </button>
          </div>
          <pre>{drawer.loading ? "Loading…" : drawer.content}</pre>
        </aside>
      )}
    </div>
  );
}