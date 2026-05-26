import React, { useState, useEffect, useRef, useCallback } from "react";

// ============================================================================
// GRAPH.AI — Graph-based Risk Analysis & Pattern Hunting
// Self-contained preview build. Mock streaming data. Four views.
// Port target: Next.js App Router (one folder per view) wired to the
// System 2 backend (S4 SSE stream + P2 alerts API + S6 subgraph endpoint).
// ============================================================================

const COLORS = {
  bg: "#0a0e17",
  panel: "#111725",
  panelHi: "#161d2e",
  border: "#1f2940",
  text: "#e6ebf5",
  textDim: "#8a96b0",
  textFaint: "#5a6680",
  accent: "#7c3aed",
  high: "#e23d6e",
  highBg: "rgba(226,61,110,0.12)",
  medium: "#d99a2b",
  mediumBg: "rgba(217,154,43,0.12)",
  low: "#2bb673",
  teal: "#34d399",
  cyan: "#22d3ee",
};

const MONO = "'JetBrains Mono', 'Fira Code', ui-monospace, monospace";
const DISPLAY = "'Space Grotesk', 'Segoe UI', sans-serif";

// --- Seed data ----------------------------------------------------------------
const SEED_TX = [
  { id: "TXN1004", route: "D -> E", from: "D", to: "E", amount: 66000, severity: "High",
    time: "2026-03-22 10:06", desc: "Final hop to newly linked beneficiary in suspicious network.",
    risk: 0.82, patterns: ["Layering"], accounts: ["C", "D", "E"] },
  { id: "TXN1003", route: "C -> D", from: "C", to: "D", amount: 39000, severity: "Medium",
    time: "2026-03-22 10:04", desc: "Funds fragmented below reporting comfort threshold.",
    risk: 0.80, patterns: ["Structuring", "Layering"], accounts: ["B", "C", "D"] },
  { id: "TXN1002", route: "B -> C", from: "B", to: "C", amount: 41000, severity: "High",
    time: "2026-03-22 10:02", desc: "Rapid relay transfer consistent with layering behavior.",
    risk: 0.78, patterns: ["Layering"], accounts: ["A", "B", "C"] },
  { id: "TXN1001", route: "A -> B", from: "A", to: "B", amount: 48000, severity: "High",
    time: "2026-03-22 10:00", desc: "Dormant account reactivated and moved funds immediately.",
    risk: 0.85, patterns: ["Dormant Activation"], accounts: ["A", "B"] },
];

const GEO = {
  A: { city: "New Delhi", lat: 28.61, lng: 77.21, note: "Origin account; dormant before reactivation." },
  B: { city: "Delhi", lat: 28.61, lng: 77.21, note: "Relay node passing funds within two minutes, indicating potential layering." },
  C: { city: "Mumbai", lat: 19.08, lng: 72.88, note: "Mid-chain account used for split movement and fund obfuscation." },
  D: { city: "Bengaluru", lat: 12.97, lng: 77.59, note: "Acts as a pass-through node before final beneficiary settlement." },
  E: { city: "Chennai", lat: 13.08, lng: 80.27, note: "Newly linked terminal beneficiary." },
  F: { city: "Kolkata", lat: 22.57, lng: 88.36, note: "Peripheral linked account." },
  G: { city: "Hyderabad", lat: 17.38, lng: 78.48, note: "Peripheral linked account." },
  H: { city: "Pune", lat: 18.52, lng: 73.85, note: "Peripheral linked account." },
};

const NAV = [
  { id: "overview", label: "Overview" },
  { id: "case", label: "Case Analysis" },
  { id: "geo", label: "Geographical View" },
  { id: "report", label: "Fraud Report" },
];

// --- Small UI atoms -----------------------------------------------------------
function Badge({ level }) {
  const map = {
    High: { c: COLORS.high, bg: COLORS.highBg },
    Medium: { c: COLORS.medium, bg: COLORS.mediumBg },
    Low: { c: COLORS.low, bg: "rgba(43,182,115,0.12)" },
  };
  const s = map[level] || map.Low;
  return (
    <span style={{
      color: s.c, background: s.bg, border: `1px solid ${s.c}55`,
      padding: "4px 14px", borderRadius: 999, fontSize: 12, fontWeight: 600,
      fontFamily: DISPLAY, letterSpacing: 0.3,
    }}>{level}</span>
  );
}

function Label({ children }) {
  return <div style={{
    fontSize: 10, letterSpacing: 2.5, textTransform: "uppercase",
    color: COLORS.textDim, fontFamily: MONO, marginBottom: 8,
  }}>{children}</div>;
}

function Panel({ children, style }) {
  return <div style={{
    background: COLORS.panel, border: `1px solid ${COLORS.border}`,
    borderRadius: 16, padding: 22, ...style,
  }}>{children}</div>;
}

function Stat({ label, value, valueColor }) {
  return (
    <div>
      <Label>{label}</Label>
      <div style={{ fontSize: 30, fontWeight: 700, fontFamily: DISPLAY,
        color: valueColor || COLORS.text, lineHeight: 1 }}>{value}</div>
    </div>
  );
}

function Btn({ children, onClick, variant = "ghost" }) {
  const styles = {
    primary: { bg: COLORS.high, color: "#fff", border: COLORS.high },
    ghost: { bg: "transparent", color: COLORS.text, border: COLORS.border },
    accent: { bg: "rgba(124,58,237,0.15)", color: "#c4b5fd", border: "#7c3aed66" },
  };
  const s = styles[variant];
  return (
    <button onClick={onClick} style={{
      background: s.bg, color: s.color, border: `1px solid ${s.border}`,
      borderRadius: 10, padding: "9px 16px", fontSize: 13, fontWeight: 600,
      fontFamily: DISPLAY, cursor: "pointer", transition: "all .15s",
    }}
    onMouseEnter={e => e.currentTarget.style.filter = "brightness(1.2)"}
    onMouseLeave={e => e.currentTarget.style.filter = "brightness(1)"}>
      {children}
    </button>
  );
}

// --- Node chain graph (Case Analysis) ----------------------------------------
function NodeChain({ tx }) {
  const chain = ["A", "B", "C", "D", "E"];
  const side = ["F", "G", "H"];
  const w = 560, h = 360, cx0 = 50, gap = 120, y = 60;
  const inPath = new Set(tx.accounts);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto" }}>
      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={COLORS.high} />
        </marker>
      </defs>
      {chain.slice(0, -1).map((n, i) => {
        const active = inPath.has(chain[i]) && inPath.has(chain[i + 1]);
        return (
          <g key={i}>
            <line x1={cx0 + i * gap + 14} y1={y} x2={cx0 + (i + 1) * gap - 14} y2={y}
              stroke={active ? COLORS.high : COLORS.border} strokeWidth={active ? 2.5 : 1.5}
              markerEnd="url(#arr)" />
            <text x={cx0 + i * gap + gap / 2} y={y - 14} fill={COLORS.textDim}
              fontSize={9} fontFamily={MONO} textAnchor="middle">
              {["1M 48s", "2M 41s", "2M 36s", "3M 44s"][i]}
            </text>
          </g>
        );
      })}
      {chain.map((n, i) => {
        const active = inPath.has(n);
        return (
          <g key={n}>
            <circle cx={cx0 + i * gap} cy={y} r={13}
              fill={active ? COLORS.high : COLORS.teal}
              stroke={active ? "#fff2" : "transparent"} strokeWidth={3} />
            <text x={cx0 + i * gap} y={y + 34} fill={COLORS.text} fontSize={12}
              fontFamily={DISPLAY} fontWeight={600} textAnchor="middle">{n}</text>
          </g>
        );
      })}
      {side.map((n, i) => (
        <g key={n}>
          <circle cx={cx0 + 4 * gap} cy={y + 80 + i * 70} r={11} fill={COLORS.teal} />
          <text x={cx0 + 4 * gap + 26} y={y + 84 + i * 70} fill={COLORS.textDim}
            fontSize={11} fontFamily={DISPLAY}>{n}</text>
        </g>
      ))}
    </svg>
  );
}

// --- India map (Geographical View) -------------------------------------------
// Lightweight equirectangular projection over an India bounding box.
function IndiaMap({ tx }) {
  const W = 560, H = 460;
  const BB = { latMin: 6, latMax: 36, lngMin: 68, lngMax: 98 };
  const proj = (lat, lng) => ({
    x: ((lng - BB.lngMin) / (BB.lngMax - BB.lngMin)) * W,
    y: H - ((lat - BB.latMin) / (BB.latMax - BB.latMin)) * H,
  });
  const fraudSet = new Set(tx.accounts);
  const pts = Object.entries(GEO);
  return (
    <div style={{ position: "relative", borderRadius: 12, overflow: "hidden",
      border: `1px solid ${COLORS.border}`, background: "#0d1320" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }}>
        {/* faint grid to suggest a map */}
        {Array.from({ length: 7 }).map((_, i) => (
          <line key={"v" + i} x1={(i * W) / 6} y1={0} x2={(i * W) / 6} y2={H}
            stroke={COLORS.border} strokeWidth={0.5} opacity={0.4} />
        ))}
        {Array.from({ length: 6 }).map((_, i) => (
          <line key={"h" + i} x1={0} y1={(i * H) / 5} x2={W} y2={(i * H) / 5}
            stroke={COLORS.border} strokeWidth={0.5} opacity={0.4} />
        ))}
        {/* fraud edges */}
        {tx.accounts.slice(0, -1).map((a, i) => {
          const b = tx.accounts[i + 1];
          if (!GEO[a] || !GEO[b]) return null;
          const p1 = proj(GEO[a].lat, GEO[a].lng), p2 = proj(GEO[b].lat, GEO[b].lng);
          return <line key={i} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
            stroke={COLORS.medium} strokeWidth={2} strokeDasharray="4 3" opacity={0.8} />;
        })}
        {pts.map(([id, g]) => {
          const p = proj(g.lat, g.lng);
          const isFraud = fraudSet.has(id);
          return (
            <g key={id}>
              <circle cx={p.x} cy={p.y} r={isFraud ? 9 : 6}
                fill={isFraud ? COLORS.high : COLORS.teal}
                stroke={isFraud ? "#fff3" : "transparent"} strokeWidth={3}>
                {isFraud && <animate attributeName="r" values="9;12;9" dur="2s" repeatCount="indefinite" />}
              </circle>
              <text x={p.x + 11} y={p.y + 4} fill={COLORS.textDim} fontSize={10}
                fontFamily={MONO}>{id} · {g.city}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ============================================================================
// MAIN
// ============================================================================
export default function App() {
  const [tab, setTab] = useState("overview");
  const [transactions, setTransactions] = useState(SEED_TX);
  const [selected, setSelected] = useState(SEED_TX[1]); // TXN1003 selected, like the mockup
  const [streaming, setStreaming] = useState(false);
  const [minAmt, setMinAmt] = useState(5000);
  const [maxAmt, setMaxAmt] = useState(90000);
  const [accounts, setAccounts] = useState(8);
  const counter = useRef(1005);
  const timer = useRef(null);

  // Mock SSE generator — in the real app this is an EventSource on /stream
  const tick = useCallback(() => {
    const letters = ["A", "B", "C", "D", "E", "F", "G", "H"];
    const from = letters[Math.floor(Math.random() * 5)];
    const to = letters[Math.floor(Math.random() * letters.length)];
    const sev = ["High", "Medium", "High", "Low"][Math.floor(Math.random() * 4)];
    const amt = Math.floor(minAmt + Math.random() * (maxAmt - minAmt));
    const id = "TXN" + counter.current++;
    const now = new Date();
    const t = `2026-03-22 ${String(10 + Math.floor(counter.current / 60) % 14).padStart(2, "0")}:${String(counter.current % 60).padStart(2, "0")}`;
    const patternPool = ["Structuring", "Layering", "Dormant Activation", "Fan-out", "Round-tripping"];
    const tx = {
      id, route: `${from} -> ${to}`, from, to, amount: amt, severity: sev, time: t,
      desc: ["Funds fragmented below reporting comfort threshold.",
        "Rapid relay transfer consistent with layering behavior.",
        "Final hop to newly linked beneficiary in suspicious network.",
        "Dormant account reactivated and moved funds immediately."][Math.floor(Math.random() * 4)],
      risk: +(0.6 + Math.random() * 0.35).toFixed(2),
      patterns: [patternPool[Math.floor(Math.random() * patternPool.length)]],
      accounts: [from, to],
    };
    setTransactions(prev => [tx, ...prev].slice(0, 20));
    if (Math.random() > 0.7) setAccounts(a => a + 1);
  }, [minAmt, maxAmt]);

  useEffect(() => {
    if (streaming) {
      timer.current = setInterval(tick, 1500 + Math.random() * 2000);
      return () => clearInterval(timer.current);
    }
  }, [streaming, tick]);

  const latest = transactions[0];

  // ---- View renderers -------------------------------------------------------
  const Overview = () => (
    <div>
      <ViewHeader title="Fraud Overview"
        sub="Review recently flagged transactions and choose one case to investigate across the rest of the workspace." />

      <Panel style={{ marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <Label>Live Generator</Label>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: DISPLAY }}>Random Transaction Stream</div>
          </div>
          <span style={{ padding: "5px 14px", borderRadius: 999, fontSize: 12, fontFamily: DISPLAY,
            fontWeight: 600, border: `1px solid ${COLORS.border}`,
            color: streaming ? COLORS.teal : COLORS.textDim,
            background: streaming ? "rgba(52,211,153,0.1)" : "transparent" }}>
            {streaming ? "Streaming" : "Stopped"}
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, margin: "18px 0" }}>
          <div>
            <Label>Minimum Amount</Label>
            <input type="number" value={minAmt} onChange={e => setMinAmt(+e.target.value)}
              style={inputStyle} />
          </div>
          <div>
            <Label>Maximum Amount</Label>
            <input type="number" value={maxAmt} onChange={e => setMaxAmt(+e.target.value)}
              style={inputStyle} />
          </div>
        </div>

        <p style={{ color: COLORS.textDim, fontSize: 13, lineHeight: 1.6, marginBottom: 16 }}>
          When started, the backend creates a new random transaction every 1 to 10 seconds and may open a new account as part of the transfer path.
        </p>

        <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
          <Btn variant="primary" onClick={() => setStreaming(true)}>Start Generator</Btn>
          <Btn onClick={() => setStreaming(false)}>Stop Generator</Btn>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
          <Panel style={{ padding: 16 }}><Stat label="Transactions" value={transactions.length} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Accounts" value={accounts} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Connection"
            value={<span style={{ fontSize: 18, color: COLORS.teal }}>Connected</span>} /></Panel>
        </div>

        <Panel style={{ padding: 16, marginTop: 14, background: COLORS.panelHi }}>
          <Label>Latest Transaction</Label>
          <div style={{ fontFamily: MONO, fontSize: 14 }}>
            {latest.id}: {latest.route} for ₹{latest.amount.toLocaleString("en-IN")} at {latest.time}
          </div>
        </Panel>
      </Panel>

      <Panel>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 18 }}>
          <div>
            <Label>Recently Flagged</Label>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: DISPLAY }}>Flagged Transactions</div>
          </div>
          <span style={{ ...pillNote }}>Select one transaction to investigate in the other tabs</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {transactions.map(tx => {
            const isSel = selected.id === tx.id;
            return (
              <div key={tx.id} onClick={() => setSelected(tx)}
                style={{
                  border: `1px solid ${isSel ? COLORS.high + "66" : COLORS.border}`,
                  background: isSel ? "linear-gradient(135deg, rgba(226,61,110,0.10), rgba(124,58,237,0.06))" : COLORS.panelHi,
                  borderRadius: 14, padding: 18, cursor: "pointer", transition: "all .15s",
                }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontFamily: MONO, fontSize: 11, color: COLORS.textDim }}>{tx.id}</span>
                  <Badge level={tx.severity} />
                </div>
                <div style={{ fontSize: 19, fontWeight: 700, fontFamily: DISPLAY, margin: "8px 0 6px" }}>{tx.route}</div>
                <p style={{ color: COLORS.textDim, fontSize: 13, marginBottom: 12 }}>{tx.desc}</p>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                  <span style={{ fontFamily: MONO, fontSize: 14 }}>₹{tx.amount.toLocaleString("en-IN")}</span>
                  <span style={{ fontFamily: MONO, fontSize: 12, color: COLORS.textDim }}>{tx.time}</span>
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <Btn onClick={(e) => { e.stopPropagation(); setSelected(tx); setTab("case"); }}>Investigate</Btn>
                  <Btn onClick={(e) => { e.stopPropagation(); setSelected(tx); setTab("geo"); }}>Open Map</Btn>
                  <Btn onClick={(e) => { e.stopPropagation(); setSelected(tx); setTab("report"); }}>Generate Report</Btn>
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );

  const CaseAnalysis = () => (
    <div>
      <ViewHeader title="Case Analysis"
        sub="See the fraud alert, laundering patterns, amount, accounts involved, and the network connected to the selected transaction." />
      <SelectedCard tx={selected} />

      <Panel style={{ marginBottom: 18, background: COLORS.highBg, borderColor: COLORS.high + "55" }}>
        <Label>Fraud Alert</Label>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontSize: 17, fontWeight: 700, fontFamily: DISPLAY }}>
            {selected.id} flagged: {selected.desc}
          </div>
          <span style={{ width: 10, height: 10, borderRadius: 999, background: COLORS.high, boxShadow: `0 0 12px ${COLORS.high}` }} />
        </div>
      </Panel>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 14, marginBottom: 18 }}>
        <Panel style={{ padding: 16 }}><Stat label="Amount Transferred" value={`₹${selected.amount.toLocaleString("en-IN")}`} /></Panel>
        <Panel style={{ padding: 16 }}><Stat label="Accounts Involved" value={selected.accounts.length} /></Panel>
        <Panel style={{ padding: 16 }}><Stat label="Detected Patterns" value={selected.patterns.length} /></Panel>
        <Panel style={{ padding: 16 }}><Stat label="Risk Score" value={selected.risk.toFixed(2)} valueColor={COLORS.high} /></Panel>
      </div>

      <Panel style={{ marginBottom: 18 }}>
        <Label>Laundering Patterns</Label>
        <div style={{ fontSize: 18, fontWeight: 700, fontFamily: DISPLAY, marginBottom: 14 }}>Detected Indicators</div>
        {[
          { t: "Structuring", d: "Value is broken into smaller transfers to reduce reporting visibility." },
          { t: "Layering", d: "Funds move rapidly across accounts to hide the original source and final beneficiary." },
        ].map(p => (
          <div key={p.t} style={{ border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 16, marginBottom: 10, background: COLORS.panelHi }}>
            <div style={{ fontWeight: 700, fontFamily: DISPLAY, marginBottom: 4 }}>{p.t}</div>
            <p style={{ color: COLORS.textDim, fontSize: 13 }}>{p.d}</p>
          </div>
        ))}
      </Panel>

      <Panel style={{ marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div>
            <Label>Fraud Network</Label>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: DISPLAY }}>Case Network Graph</div>
          </div>
          <span style={{ ...pillNote }}>Whole selected network</span>
        </div>
        <div style={{ background: COLORS.panelHi, border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 16 }}>
          <NodeChain tx={selected} />
        </div>
      </Panel>

      <Panel>
        <Label>Investigation Notes</Label>
        <div style={{ fontSize: 18, fontWeight: 700, fontFamily: DISPLAY, marginBottom: 14 }}>Why It Was Flagged</div>
        {["Funds fragmented below comfort threshold",
          "Mid-chain account used for obfuscation",
          "Geography changed while staying in same network"].map((n, i) => (
          <div key={i} style={{ border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "14px 16px", marginBottom: 10, fontWeight: 600, fontFamily: DISPLAY, fontSize: 14 }}>
            {n}
          </div>
        ))}
      </Panel>
    </div>
  );

  const GeoView = () => (
    <div>
      <ViewHeader title="Geographical View"
        sub="Show all active accounts geographically and highlight only the accounts and edges involved in the selected fraud case." />
      <SelectedCard tx={selected} />

      <Panel style={{ marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div>
            <Label>Geographical View</Label>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: DISPLAY }}>Fraud Geography Map</div>
          </div>
          <span style={{ ...pillNote }}>All active accounts shown, fraud network highlighted</span>
        </div>
        <IndiaMap tx={selected} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginTop: 16 }}>
          <Panel style={{ padding: 16 }}><Stat label="Active Accounts" value={accounts} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Fraud Accounts" value={selected.accounts.length} valueColor={COLORS.high} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Fraud Edges" value={Math.max(1, selected.accounts.length - 1)} valueColor={COLORS.medium} /></Panel>
        </div>
      </Panel>

      <Panel>
        <Label>Highlighted Geography</Label>
        <div style={{ fontSize: 18, fontWeight: 700, fontFamily: DISPLAY, marginBottom: 14 }}>Fraud Network Locations</div>
        {selected.accounts.map(a => GEO[a] && (
          <div key={a} style={{ border: `1px solid ${COLORS.border}`, borderRadius: 12, padding: 16, marginBottom: 10, background: COLORS.panelHi }}>
            <div style={{ fontWeight: 700, fontFamily: DISPLAY, marginBottom: 4 }}>Account {a} | {GEO[a].city}</div>
            <div style={{ fontFamily: MONO, fontSize: 12, color: COLORS.textDim, marginBottom: 6 }}>Location: {GEO[a].lat}, {GEO[a].lng}</div>
            <p style={{ color: COLORS.textDim, fontSize: 13 }}>{GEO[a].note}</p>
          </div>
        ))}
      </Panel>
    </div>
  );

  const Report = () => (
    <div>
      <ViewHeader title="FIU Reporting"
        sub="Build a regulator-ready report for the currently selected transaction and its related suspicious network." />
      <SelectedCard tx={selected} />

      <Panel>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
          <div>
            <Label>FIU Filing Preview</Label>
            <div style={{ fontSize: 20, fontWeight: 700, fontFamily: DISPLAY }}>Fraud Report &amp; Evidence Package</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <Btn variant="accent">Download Evidence Package</Btn>
            <Btn variant="primary">Download Report</Btn>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
          <Panel style={{ padding: 16 }}><Stat label="Case ID" value={<span style={{ fontFamily: MONO, fontSize: 20 }}>FRAUD_001{selected.to}</span>} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Transaction" value={<span style={{ fontFamily: MONO, fontSize: 20 }}>{selected.id}</span>} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Total Amount" value={`₹${(selected.amount * 2).toLocaleString("en-IN")}`} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Risk Score" value={selected.risk.toFixed(2)} valueColor={COLORS.high} /></Panel>
        </div>

        <ReportRow label="Impacted Accounts">
          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            {selected.accounts.map(a => (
              <span key={a} style={{ width: 30, height: 30, borderRadius: 999, border: `1px solid ${COLORS.border}`,
                display: "grid", placeItems: "center", fontFamily: MONO, fontSize: 13, background: COLORS.panelHi }}>{a}</span>
            ))}
          </div>
          <p style={{ color: COLORS.textDim, fontSize: 13 }}>{selected.desc}</p>
        </ReportRow>

        <ReportRow label="Recommendation">
          <div style={{ fontSize: 16, fontWeight: 700, fontFamily: DISPLAY, marginBottom: 6 }}>Continue tracing downstream beneficiaries</div>
          <p style={{ color: COLORS.textDim, fontSize: 13 }}>The latest hop fragments value and preserves velocity, reinforcing a structuring narrative.</p>
        </ReportRow>

        <ReportRow label="Fraud Patterns">
          <div style={{ display: "flex", gap: 8 }}>
            {selected.patterns.map(p => (
              <span key={p} style={{ padding: "6px 16px", borderRadius: 999, background: "rgba(124,58,237,0.15)",
                color: "#c4b5fd", fontFamily: DISPLAY, fontSize: 13, fontWeight: 600, border: "1px solid #7c3aed44" }}>{p}</span>
            ))}
          </div>
        </ReportRow>

        <ReportRow label="Suggested Filing Notes">
          <p style={{ color: COLORS.textDim, fontSize: 13, lineHeight: 1.7 }}>
            Selected transaction {selected.id} moved ₹{selected.amount.toLocaleString("en-IN")} along {selected.route} on {selected.time}.
            The case was escalated because funds fragmented below comfort threshold, a mid-chain account was used for obfuscation,
            and geography changed while staying in the same network. This produced a case risk score of {selected.risk.toFixed(2)} and
            supports the recommendation to continue tracing downstream beneficiaries.
          </p>
        </ReportRow>

        <ReportRow label="Evidence Package">
          <p style={{ color: COLORS.textDim, fontSize: 13, marginBottom: 12 }}>
            This package includes the traced fund journey, the involved accounts, the suspicious timeline, and the supporting indicators required for FIU escalation.
          </p>
          {transactions.slice(0, 2).map(t => (
            <div key={t.id} style={{ border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: 14, marginBottom: 10, background: COLORS.panelHi }}>
              <div style={{ fontFamily: MONO, fontSize: 13, marginBottom: 4 }}>{t.id} | {t.route}</div>
              <div style={{ fontFamily: MONO, fontSize: 12, color: COLORS.textDim }}>₹{t.amount.toLocaleString("en-IN")} at {t.time}</div>
            </div>
          ))}
        </ReportRow>
      </Panel>
    </div>
  );

  // ---- Shared sub-renderers -------------------------------------------------
  function ViewHeader({ title, sub }) {
    return (
      <div style={{ marginBottom: 22 }}>
        <div style={{ fontSize: 11, letterSpacing: 3, fontFamily: MONO, color: COLORS.textDim, marginBottom: 8 }}>G.R.A.P.H. AI</div>
        <h1 style={{ fontSize: 34, fontWeight: 700, fontFamily: DISPLAY, margin: 0 }}>{title}</h1>
        <p style={{ color: COLORS.textDim, fontSize: 14, marginTop: 8, maxWidth: 640, lineHeight: 1.6 }}>{sub}</p>
      </div>
    );
  }

  function SelectedCard({ tx }) {
    return (
      <Panel style={{ marginBottom: 18 }}>
        <Label>Selected Transaction</Label>
        <span style={{ fontFamily: MONO, fontSize: 13, padding: "5px 14px", borderRadius: 8,
          background: COLORS.panelHi, border: `1px solid ${COLORS.border}`, display: "inline-block", marginBottom: 12 }}>{tx.id}</span>
        <div style={{ fontSize: 18, fontWeight: 700, fontFamily: DISPLAY, marginBottom: 6 }}>{tx.route} | Mid-chain fragmentation suggests structuring</div>
        <p style={{ color: COLORS.textDim, fontSize: 13, marginBottom: 16, lineHeight: 1.6 }}>
          The transaction through C into D reduced value and maintained speed, strengthening the structuring hypothesis.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Panel style={{ padding: 16 }}><Stat label="Severity" value={<span style={{ fontSize: 18 }}>{tx.severity}</span>} /></Panel>
          <Panel style={{ padding: 16 }}><Stat label="Risk Score" value={tx.risk.toFixed(2)} valueColor={COLORS.high} /></Panel>
        </div>
      </Panel>
    );
  }

  function ReportRow({ label, children }) {
    return (
      <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 16, marginTop: 16 }}>
        <Label>{label}</Label>
        {children}
      </div>
    );
  }

  const views = { overview: Overview, case: CaseAnalysis, geo: GeoView, report: Report };
  const View = views[tab];

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: DISPLAY }}>
      <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      {/* Sidebar */}
      <aside style={{ width: 240, borderRight: `1px solid ${COLORS.border}`, padding: 24, position: "sticky", top: 0, height: "100vh" }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 32 }}>
          <div style={{ width: 40, height: 40, borderRadius: 10, background: "linear-gradient(135deg, #1e3a5f, #2bb673)",
            display: "grid", placeItems: "center", fontWeight: 700, fontFamily: DISPLAY, fontSize: 14 }}>GA</div>
          <div>
            <div style={{ fontSize: 10, letterSpacing: 2, fontFamily: MONO, color: COLORS.textDim }}>G.R.A.P.H. AI</div>
            <div style={{ fontSize: 13, fontWeight: 700, lineHeight: 1.3 }}>Graph-based Risk Analysis &amp; Pattern Hunting</div>
          </div>
        </div>
        <nav style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {NAV.map(n => (
            <button key={n.id} onClick={() => setTab(n.id)} style={{
              textAlign: "left", padding: "12px 16px", borderRadius: 10, fontSize: 14, fontWeight: 600,
              fontFamily: DISPLAY, cursor: "pointer", border: "none", transition: "all .15s",
              background: tab === n.id ? "rgba(52,211,153,0.10)" : "transparent",
              color: tab === n.id ? COLORS.text : COLORS.textDim,
              borderLeft: tab === n.id ? `2px solid ${COLORS.teal}` : "2px solid transparent",
            }}>{n.label}</button>
          ))}
        </nav>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, padding: "32px 40px", maxWidth: 760, margin: "0 auto" }}>
        <View />
      </main>
    </div>
  );
}

const inputStyle = {
  width: "100%", background: "#0a0e17", border: `1px solid ${COLORS.border}`,
  borderRadius: 10, padding: "12px 14px", color: COLORS.text, fontFamily: MONO,
  fontSize: 15, outline: "none", boxSizing: "border-box",
};

const pillNote = {
  padding: "8px 14px", borderRadius: 10, fontSize: 12, fontWeight: 600, fontFamily: DISPLAY,
  background: "rgba(52,211,153,0.08)", color: COLORS.teal, border: `1px solid rgba(52,211,153,0.2)`,
  maxWidth: 200, lineHeight: 1.4,
};
