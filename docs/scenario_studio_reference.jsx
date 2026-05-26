import React, { useState, useMemo } from "react";

// ============================================================================
// GRAPH.AI — Scenario Studio (control panel preview)
// Pattern-level controllable transaction generator.
// One engine; two entry points: LIVE inject into stream, or EXPORT to CSV.
// This preview computes a deterministic plan locally so you can see the
// interaction model. In the real build, "Build Plan" calls the backend
// ScenarioEngine and the plan/preview come from it.
// ============================================================================

const C = {
  bg: "#0a0e17", panel: "#111725", panelHi: "#161d2e", border: "#1f2940",
  text: "#e6ebf5", dim: "#8a96b0", high: "#e23d6e", highBg: "rgba(226,61,110,0.12)",
  med: "#d99a2b", teal: "#34d399", cyan: "#22d3ee", accent: "#7c3aed",
};
const DISPLAY = "'Space Grotesk',sans-serif";
const MONO = "'JetBrains Mono',ui-monospace,monospace";

// --- Typology catalog. Each defines its tunable params + which rules it targets.
const TYPOLOGIES = {
  structuring: {
    label: "Structuring", color: C.med,
    blurb: "Repeated sub-threshold transfers to stay under reporting limits.",
    pattern: "Linear A→B, repeated",
    rules: ["R02", "R11", "R13"],
    params: {
      count: { label: "Sub-threshold transfers", min: 5, max: 15, def: 8, step: 1 },
      amountLo: { label: "Amount floor (₹)", min: 700000, max: 950000, def: 850000, step: 10000 },
      amountHi: { label: "Amount ceiling (₹)", min: 950000, max: 999999, def: 999000, step: 1000 },
      windowH: { label: "Window (hours)", min: 1, max: 48, def: 24, step: 1 },
    },
  },
  circular_laundering: {
    label: "Circular Laundering", color: C.high,
    blurb: "Round-trip cycle A→B→C→A that returns funds to origin.",
    pattern: "Directed cycle",
    rules: ["R10", "R03"],
    params: {
      ringSize: { label: "Ring size (accounts)", min: 3, max: 6, def: 4, step: 1 },
      amount: { label: "Cycle amount (₹)", min: 100000, max: 2000000, def: 500000, step: 50000 },
      windowH: { label: "Cycle window (hours)", min: 6, max: 48, def: 48, step: 1 },
    },
  },
  layering_chain: {
    label: "Layering Chain", color: C.high,
    blurb: "Multi-hop chain A→B→C→…→Z, each hop changing bank/country.",
    pattern: "Linear chain",
    rules: ["R03", "R06"],
    params: {
      hops: { label: "Hops", min: 3, max: 8, def: 5, step: 1 },
      amount: { label: "Initial amount (₹)", min: 100000, max: 5000000, def: 1000000, step: 100000 },
      decay: { label: "Value retained per hop (%)", min: 60, max: 98, def: 90, step: 1 },
      gapH: { label: "Avg gap between hops (h)", min: 1, max: 12, def: 6, step: 1 },
    },
  },
  fan_in: {
    label: "Fan-In", color: C.med,
    blurb: "Many sources funnel into one aggregator account.",
    pattern: "Star, target at center",
    rules: ["R09"],
    params: {
      sources: { label: "Source accounts", min: 5, max: 15, def: 8, step: 1 },
      amount: { label: "Per-source amount (₹)", min: 10000, max: 500000, def: 80000, step: 10000 },
      windowH: { label: "Collection window (h)", min: 1, max: 24, def: 12, step: 1 },
    },
  },
  fan_out: {
    label: "Fan-Out", color: C.med,
    blurb: "One distributor splits funds across many targets.",
    pattern: "Star, source at center",
    rules: ["R09", "R02"],
    params: {
      targets: { label: "Target accounts", min: 5, max: 15, def: 8, step: 1 },
      amount: { label: "Per-target amount (₹)", min: 10000, max: 500000, def: 70000, step: 10000 },
      windowH: { label: "Distribution window (h)", min: 1, max: 24, def: 6, step: 1 },
    },
  },
  fraud_ring: {
    label: "Fraud Ring", color: C.high,
    blurb: "Dense clique of mutually transacting accounts.",
    pattern: "Dense clique (density > 0.6)",
    rules: ["R08", "R10"],
    params: {
      members: { label: "Ring members", min: 4, max: 10, def: 6, step: 1 },
      density: { label: "Edge density", min: 0.6, max: 1.0, def: 0.7, step: 0.05 },
      amount: { label: "Avg edge amount (₹)", min: 50000, max: 1000000, def: 200000, step: 50000 },
    },
  },
  dormant_activation: {
    label: "Dormant Activation", color: C.high,
    blurb: "Long-silent account reactivates and bursts funds out.",
    pattern: "Single node burst",
    rules: ["R04", "R07", "R03"],
    params: {
      silentDays: { label: "Silent period (days)", min: 180, max: 720, def: 200, step: 10 },
      burstCount: { label: "Burst transactions", min: 10, max: 30, def: 12, step: 1 },
      amount: { label: "Inbound amount (₹)", min: 100000, max: 3000000, def: 800000, step: 100000 },
      disburse: { label: "Disbursement targets", min: 3, max: 10, def: 4, step: 1 },
    },
  },
  velocity_burst: {
    label: "Velocity Burst", color: C.med,
    blurb: "Intense cluster of transactions in a short window.",
    pattern: "Temporal cluster",
    rules: ["R03"],
    params: {
      count: { label: "Transactions", min: 15, max: 60, def: 20, step: 1 },
      windowMin: { label: "Window (minutes)", min: 10, max: 60, def: 60, step: 5 },
      amount: { label: "Avg amount (₹)", min: 10000, max: 500000, def: 60000, step: 10000 },
    },
  },
  cross_border_layering: {
    label: "Cross-Border Layering", color: C.high,
    blurb: "Chain routed through high-risk jurisdictions.",
    pattern: "Chain via high-risk countries",
    rules: ["R06", "R03"],
    params: {
      hops: { label: "International hops", min: 2, max: 6, def: 3, step: 1 },
      amount: { label: "Amount (₹)", min: 500000, max: 5000000, def: 1500000, step: 100000 },
      gapH: { label: "Settlement gap (h)", min: 6, max: 24, def: 12, step: 1 },
    },
  },
  round_tripping: {
    label: "Round-Tripping", color: C.high,
    blurb: "International round-trip A(IN)→B(AE)→C(SG)→A(IN).",
    pattern: "Directed cycle via international",
    rules: ["R10", "R06"],
    params: {
      hops: { label: "International hops", min: 3, max: 5, def: 3, step: 1 },
      amount: { label: "Amount (₹)", min: 500000, max: 5000000, def: 2000000, step: 100000 },
      gapH: { label: "Settlement gap (h)", min: 12, max: 48, def: 24, step: 1 },
    },
  },
};

const RULE_NAMES = {
  R02: "Structuring", R03: "Velocity spike", R04: "Dormant activation",
  R06: "High-risk jurisdiction", R07: "Device anomaly", R08: "Shared device",
  R09: "Excessive beneficiaries", R10: "Cycle closure", R11: "Round-amount structuring",
  R13: "Benford anomaly",
};

// --- Deterministic local planner (preview only) -------------------------------
function buildPlan(key, p, seed) {
  const t0 = new Date("2026-03-22T10:00:00");
  const edges = [];
  const letter = (i) => String.fromCharCode(65 + (i % 26));
  const fmt = (d) => d.toISOString().slice(0, 16).replace("T", " ");
  const addMin = (mins) => new Date(t0.getTime() + mins * 60000);
  let acc = 0;

  const push = (from, to, amt, mins) =>
    edges.push({ from, to, amount: Math.round(amt), time: fmt(addMin(mins)), idx: edges.length });

  switch (key) {
    case "structuring": {
      const span = p.windowH * 60;
      for (let i = 0; i < p.count; i++) {
        const amt = p.amountLo + ((p.amountHi - p.amountLo) * ((i * 37 + seed) % 100)) / 100;
        push("A", "B", amt, (span / (p.count + 1)) * (i + 1));
      }
      break;
    }
    case "circular_laundering": {
      const per = (p.windowH * 60) / p.ringSize;
      for (let i = 0; i < p.ringSize; i++)
        push(letter(i), letter((i + 1) % p.ringSize), p.amount * 0.97 ** i, per * (i + 1));
      break;
    }
    case "layering_chain": {
      let amt = p.amount;
      for (let i = 0; i < p.hops; i++) {
        push(letter(i), letter(i + 1), amt, p.gapH * 60 * (i + 1));
        amt *= p.decay / 100;
      }
      break;
    }
    case "fan_in":
      for (let i = 0; i < p.sources; i++)
        push(letter(i + 1), "A", p.amount, (p.windowH * 60 / p.sources) * (i + 1));
      break;
    case "fan_out":
      for (let i = 0; i < p.targets; i++)
        push("A", letter(i + 1), p.amount, (p.windowH * 60 / p.targets) * (i + 1) + 5);
      break;
    case "fraud_ring": {
      const n = p.members, maxE = Math.round(n * (n - 1) * p.density);
      let c = 0;
      for (let i = 0; i < n && c < maxE; i++)
        for (let j = 0; j < n && c < maxE; j++)
          if (i !== j && (i + j + seed) % 3 === 0) { push(letter(i), letter(j), p.amount, c * 60); c++; }
      break;
    }
    case "dormant_activation":
      push("EXT", "A", p.amount, 0);
      for (let i = 0; i < p.disburse; i++)
        push("A", letter(i + 1), p.amount / p.disburse, 30 + i * 12);
      break;
    case "velocity_burst":
      for (let i = 0; i < p.count; i++)
        push("A", letter((i % 5) + 1), p.amount, (p.windowMin / p.count) * i);
      break;
    case "cross_border_layering": {
      const ctry = ["IN", "AE", "SG", "MU", "CN", "NG"];
      let amt = p.amount;
      for (let i = 0; i < p.hops; i++) {
        push(`${letter(i)}·${ctry[i % ctry.length]}`, `${letter(i + 1)}·${ctry[(i + 1) % ctry.length]}`, amt, p.gapH * 60 * (i + 1));
        amt *= 0.95;
      }
      break;
    }
    case "round_tripping": {
      const ctry = ["IN", "AE", "SG"];
      for (let i = 0; i < p.hops; i++)
        push(`${letter(i)}·${ctry[i % 3]}`, `${letter((i + 1) % p.hops)}·${ctry[(i + 1) % 3]}`, p.amount * 0.96 ** i, p.gapH * 60 * (i + 1));
      break;
    }
    default: break;
  }
  return edges;
}

// --- UI atoms -----------------------------------------------------------------
const Label = ({ children }) => (
  <div style={{ fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: C.dim, fontFamily: MONO, marginBottom: 6 }}>{children}</div>
);

function Slider({ cfg, value, onChange }) {
  const isFloat = cfg.step < 1;
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: C.text, fontFamily: DISPLAY }}>{cfg.label}</span>
        <span style={{ fontSize: 13, color: C.teal, fontFamily: MONO }}>
          {isFloat ? value.toFixed(2) : value.toLocaleString("en-IN")}
        </span>
      </div>
      <input type="range" min={cfg.min} max={cfg.max} step={cfg.step} value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%", accentColor: C.accent }} />
    </div>
  );
}

// ============================================================================
export default function ScenarioStudio() {
  const [active, setActive] = useState("layering_chain");
  const [params, setParams] = useState(() =>
    Object.fromEntries(Object.entries(TYPOLOGIES.layering_chain.params).map(([k, v]) => [k, v.def])));
  const [mode, setMode] = useState("live"); // "live" | "export"
  const [seed, setSeed] = useState(42);
  const [count, setCount] = useState(1);
  const [log, setLog] = useState([]);

  const T = TYPOLOGIES[active];

  const selectTypology = (key) => {
    setActive(key);
    setParams(Object.fromEntries(Object.entries(TYPOLOGIES[key].params).map(([k, v]) => [k, v.def])));
  };

  const plan = useMemo(() => buildPlan(active, params, seed), [active, params, seed]);
  const totalValue = plan.reduce((s, e) => s + e.amount, 0);
  const accounts = new Set(plan.flatMap((e) => [e.from, e.to])).size;

  const fire = () => {
    const stamp = new Date().toLocaleTimeString();
    const verb = mode === "live" ? "Injected into live stream" : "Appended to export buffer";
    setLog((l) => [{ stamp, msg: `${verb}: ${count}× ${T.label} (${plan.length} edges, ${accounts} accts, ₹${totalValue.toLocaleString("en-IN")})` }, ...l].slice(0, 8));
  };

  return (
    <div style={{ background: C.bg, color: C.text, fontFamily: DISPLAY, minHeight: "100vh", padding: "28px 32px" }}>
      <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />

      <div style={{ fontSize: 11, letterSpacing: 3, fontFamily: MONO, color: C.dim, marginBottom: 8 }}>G.R.A.P.H. AI · SCENARIO STUDIO</div>
      <h1 style={{ fontSize: 32, fontWeight: 700, margin: "0 0 6px" }}>Controllable Transaction Generator</h1>
      <p style={{ color: C.dim, fontSize: 14, maxWidth: 720, marginBottom: 24, lineHeight: 1.6 }}>
        Pick a laundering typology, tune its parameters, and either inject it into the live stream or append it to an exportable labeled dataset. Same engine, both entry points.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 320px", gap: 18, alignItems: "start" }}>

        {/* --- Column 1: typology picker --- */}
        <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16, padding: 16 }}>
          <Label>Typology</Label>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
            {Object.entries(TYPOLOGIES).map(([key, t]) => {
              const on = key === active;
              return (
                <button key={key} onClick={() => selectTypology(key)} style={{
                  textAlign: "left", padding: "10px 12px", borderRadius: 10, cursor: "pointer",
                  fontFamily: DISPLAY, fontSize: 13, fontWeight: 600, transition: "all .15s",
                  background: on ? `${t.color}1a` : "transparent",
                  border: `1px solid ${on ? t.color + "66" : "transparent"}`,
                  color: on ? C.text : C.dim,
                }}>
                  <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 999, background: t.color, marginRight: 8 }} />
                  {t.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* --- Column 2: parameters + preview --- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16, padding: 22 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <span style={{ width: 10, height: 10, borderRadius: 999, background: T.color }} />
              <h2 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>{T.label}</h2>
            </div>
            <p style={{ color: C.dim, fontSize: 13, marginBottom: 4 }}>{T.blurb}</p>
            <div style={{ fontFamily: MONO, fontSize: 11, color: C.dim, marginBottom: 18 }}>Graph pattern: {T.pattern}</div>

            {Object.entries(T.params).map(([k, cfg]) => (
              <Slider key={k} cfg={cfg} value={params[k]} onChange={(v) => setParams((p) => ({ ...p, [k]: v }))} />
            ))}

            <Label>Targets detection rules</Label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
              {T.rules.map((r) => (
                <span key={r} style={{ padding: "5px 11px", borderRadius: 999, background: "rgba(124,58,237,0.15)", border: "1px solid #7c3aed44", color: "#c4b5fd", fontFamily: MONO, fontSize: 11 }}>
                  {r} · {RULE_NAMES[r]}
                </span>
              ))}
            </div>
          </div>

          {/* preview timeline */}
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16, padding: 22 }}>
            <Label>Planned sequence preview ({plan.length} edges)</Label>
            <div style={{ maxHeight: 260, overflowY: "auto", marginTop: 8 }}>
              {plan.map((e) => (
                <div key={e.idx} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderBottom: `1px solid ${C.border}` }}>
                  <span style={{ fontFamily: MONO, fontSize: 11, color: C.dim, width: 28 }}>#{e.idx + 1}</span>
                  <span style={{ fontFamily: DISPLAY, fontWeight: 700, fontSize: 14, width: 130 }}>
                    {e.from} <span style={{ color: T.color }}>→</span> {e.to}
                  </span>
                  <span style={{ fontFamily: MONO, fontSize: 13, flex: 1 }}>₹{e.amount.toLocaleString("en-IN")}</span>
                  <span style={{ fontFamily: MONO, fontSize: 11, color: C.dim }}>{e.time}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* --- Column 3: action panel --- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16, padding: 22 }}>
            <Label>Output mode</Label>
            <div style={{ display: "flex", gap: 8, marginTop: 6, marginBottom: 18 }}>
              {[["live", "Live inject"], ["export", "Export CSV"]].map(([m, lbl]) => (
                <button key={m} onClick={() => setMode(m)} style={{
                  flex: 1, padding: "10px", borderRadius: 10, cursor: "pointer", fontFamily: DISPLAY,
                  fontSize: 13, fontWeight: 600, transition: "all .15s",
                  background: mode === m ? (m === "live" ? C.highBg : "rgba(52,211,153,0.12)") : "transparent",
                  border: `1px solid ${mode === m ? (m === "live" ? C.high + "66" : C.teal + "66") : C.border}`,
                  color: mode === m ? C.text : C.dim,
                }}>{lbl}</button>
              ))}
            </div>

            <Slider cfg={{ label: "How many instances", min: 1, max: 25, def: 1, step: 1 }} value={count} onChange={setCount} />
            <Slider cfg={{ label: "Random seed", min: 1, max: 999, def: 42, step: 1 }} value={seed} onChange={setSeed} />

            <button onClick={fire} style={{
              width: "100%", marginTop: 8, padding: "13px", borderRadius: 12, cursor: "pointer",
              fontFamily: DISPLAY, fontSize: 15, fontWeight: 700, border: "none",
              background: mode === "live" ? C.high : C.teal, color: mode === "live" ? "#fff" : "#06281c",
            }}>
              {mode === "live" ? `⚡ Inject ${count}× into stream` : `↓ Add ${count}× to export`}
            </button>
            {mode === "export" && (
              <button style={{
                width: "100%", marginTop: 10, padding: "11px", borderRadius: 12, cursor: "pointer",
                fontFamily: DISPLAY, fontSize: 13, fontWeight: 600,
                background: "transparent", border: `1px solid ${C.border}`, color: C.text,
              }}>Download stream_transactions.csv + ground_truth.csv</button>
            )}
          </div>

          {/* summary */}
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16, padding: 22 }}>
            <Label>This instance</Label>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 6 }}>
              <div><div style={{ fontSize: 24, fontWeight: 700 }}>{plan.length}</div><div style={{ fontSize: 11, color: C.dim }}>edges</div></div>
              <div><div style={{ fontSize: 24, fontWeight: 700 }}>{accounts}</div><div style={{ fontSize: 11, color: C.dim }}>accounts</div></div>
              <div style={{ gridColumn: "1/3" }}><div style={{ fontSize: 22, fontWeight: 700, fontFamily: MONO }}>₹{totalValue.toLocaleString("en-IN")}</div><div style={{ fontSize: 11, color: C.dim }}>total value, labeled is_suspicious=true</div></div>
            </div>
          </div>

          {/* activity log */}
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 16, padding: 22 }}>
            <Label>Activity</Label>
            {log.length === 0 ? (
              <p style={{ color: C.dim, fontSize: 12, marginTop: 8 }}>No scenarios fired yet. Configure a typology and inject or export.</p>
            ) : (
              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
                {log.map((l, i) => (
                  <div key={i} style={{ fontSize: 12, color: C.text, borderLeft: `2px solid ${C.teal}`, paddingLeft: 10 }}>
                    <span style={{ fontFamily: MONO, color: C.dim, fontSize: 10 }}>{l.stamp}</span>
                    <div style={{ lineHeight: 1.4 }}>{l.msg}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
