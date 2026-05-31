"use client";

/**
 * /transactions — All Transactions tab.
 *
 * Shows every event currently in the backend's recent-events buffer (capped
 * at MAX_BUFFER=200 by the generator loop), merged with the live SSE
 * stream so new events appear at the top in real time.
 *
 * Layout: filter chips + stat row + dense table. Each row expands to show
 * the full per-layer score breakdown, triggered rules, triggered patterns,
 * and the fusion explanation. Clicking the case-id chip selects that
 * transaction across the rest of the workspace (same Zustand store
 * Overview uses), so the analyst can jump straight to /case or /report.
 */

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useEventStream, LiveEvent } from "@/hooks/useEventStream";
import { useSelectionStore } from "@/stores/selectionStore";
import { Badge, Btn, Label, Panel, PillNote, SectionTitle, Stat, ViewHeader } from "@/components/ui";

type Filter = "all" | "alerted" | "Low" | "Medium" | "High" | "Critical";

const LAYER_COLORS = {
  rule:       "#22d3ee",
  graph:      "#a78bfa",
  supervised: "#34d399",
  anomaly:    "#fbbf24",
  tgn:        "#f472b6",
} as const;

function fmtTime(iso?: string): string {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 19);
}

function fmtAmount(amt?: number): string {
  if (amt == null || !Number.isFinite(amt)) return "—";
  if (amt >= 1e7) return `₹${(amt / 1e7).toFixed(2)} Cr`;
  if (amt >= 1e5) return `₹${(amt / 1e5).toFixed(2)} L`;
  return `₹${amt.toLocaleString("en-IN")}`;
}

export default function TransactionsPage() {
  const router = useRouter();
  const setSelected = useSelectionStore(s => s.setSelected);
  const { events: liveEvents, connected } = useEventStream(200);
  const [snapshot, setSnapshot] = useState<LiveEvent[]>([]);
  const [bufferSize, setBufferSize] = useState(0);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  // Pull the backend snapshot on mount + every 5 s so we have everything
  // the generator has produced, not just what arrived via SSE.
  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const r = await api.recentEvents({ limit: 200 });
        if (!cancel) {
          setSnapshot(r.items as LiveEvent[]);
          setBufferSize(r.buffer_size);
        }
      } catch { /* ignore */ }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { cancel = true; clearInterval(id); };
  }, []);

  // Merge snapshot + live SSE stream, keyed by transaction_id; live wins
  // (it carries the most recent fields). Sorted by timestamp DESC.
  const merged = useMemo<LiveEvent[]>(() => {
    const map = new Map<string, LiveEvent>();
    for (const ev of snapshot) map.set(ev.transaction_id, ev);
    for (const ev of liveEvents) map.set(ev.transaction_id, ev);
    return Array.from(map.values()).sort(
      (a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || "")),
    );
  }, [snapshot, liveEvents]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return merged.filter(ev => {
      if (filter === "alerted" && !ev.alerted) return false;
      if (["Low", "Medium", "High", "Critical"].includes(filter) && ev.risk_level !== filter) return false;
      if (q) {
        const hay = `${ev.transaction_id} ${ev.sender_account} ${ev.receiver_account} ${(ev.triggered_rules || []).join(",")}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [merged, filter, query]);

  const stats = useMemo(() => {
    const byLevel: Record<string, number> = { Low: 0, Medium: 0, High: 0, Critical: 0 };
    let alerted = 0, sumRisk = 0;
    for (const ev of merged) {
      byLevel[ev.risk_level] = (byLevel[ev.risk_level] || 0) + 1;
      if (ev.alerted) alerted++;
      sumRisk += ev.transaction_risk_score || 0;
    }
    return {
      total: merged.length,
      alerted,
      meanRisk: merged.length ? sumRisk / merged.length : 0,
      byLevel,
    };
  }, [merged]);

  const toggleRow = (txId: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(txId)) next.delete(txId); else next.add(txId);
      return next;
    });
  };

  const investigate = (tab: "case" | "geo" | "report", ev: LiveEvent) => {
    setSelected(ev.transaction_id, ev);
    router.push(`/${tab}`);
  };

  return (
    <div className="max-w-7xl mx-auto pb-12">
      <ViewHeader
        title="All Transactions"
        sub="Every transaction in the live detector's recency buffer, with per-layer scores and fusion verdict. Click any row to see the full evidence."
      />

      {/* Stat row + filters */}
      <Panel className="mb-4">
        <div className="flex justify-between items-start mb-4">
          <div>
            <Label>Live transaction buffer</Label>
            <SectionTitle>
              {merged.length} events shown · buffer size {bufferSize}
            </SectionTitle>
          </div>
          <PillNote>
            {connected ? "stream live · merging with backend snapshot" : "stream offline · snapshot only"}
          </PillNote>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-6 gap-3.5 mb-4">
          <Panel className="p-3"><Stat label="Total" value={stats.total} /></Panel>
          <Panel className="p-3"><Stat label="Alerts" value={stats.alerted} valueColor="#e23d6e" /></Panel>
          <Panel className="p-3"><Stat label="Mean risk" value={stats.meanRisk.toFixed(1)} /></Panel>
          <Panel className="p-3"><Stat label="Low" value={stats.byLevel.Low || 0} valueColor="#34d399" /></Panel>
          <Panel className="p-3"><Stat label="Medium" value={stats.byLevel.Medium || 0} valueColor="#fbbf24" /></Panel>
          <Panel className="p-3"><Stat label="High+Crit" value={(stats.byLevel.High || 0) + (stats.byLevel.Critical || 0)} valueColor="#e23d6e" /></Panel>
        </div>

        <div className="flex flex-wrap gap-2 items-center">
          {(["all", "alerted", "Critical", "High", "Medium", "Low"] as Filter[]).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={
                "px-3 py-1.5 rounded-lg text-xs font-mono uppercase tracking-wider transition " +
                (filter === f
                  ? "bg-teal/20 text-teal border border-teal/40"
                  : "bg-panelHi text-textDim border border-border hover:text-text")
              }
            >
              {f === "all" ? "ALL" : f === "alerted" ? "ALERTS" : f.toUpperCase()}
            </button>
          ))}
          <input
            placeholder="search by tx-id / account / rule …"
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="ml-auto bg-bg border border-border rounded-lg px-3 py-1.5 text-xs font-mono text-text outline-none focus:border-teal/60 w-72"
          />
        </div>
      </Panel>

      {/* Table */}
      <Panel className="p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-panelHi text-textDim font-mono">
              <tr className="text-left">
                <th className="px-3 py-2 w-6"></th>
                <th className="px-3 py-2">Time</th>
                <th className="px-3 py-2">Tx-ID</th>
                <th className="px-3 py-2">Sender → Receiver</th>
                <th className="px-3 py-2 text-right">Amount</th>
                <th className="px-3 py-2">Type</th>
                <th className="px-3 py-2 text-right">L1</th>
                <th className="px-3 py-2 text-right">L2</th>
                <th className="px-3 py-2 text-right">L3</th>
                <th className="px-3 py-2 text-right">L4</th>
                <th className="px-3 py-2 text-right">L5</th>
                <th className="px-3 py-2 text-right">FUSED</th>
                <th className="px-3 py-2">Level</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(ev => {
                const open = expanded.has(ev.transaction_id);
                const bd = ev.score_breakdown ?? {};
                const scores = bd.scores ?? {};
                const contribs = bd.weighted_contributions ?? {};
                const weights  = bd.weights ?? {};
                return (
                  <FragmentRow
                    key={ev.transaction_id}
                    ev={ev}
                    open={open}
                    onToggle={() => toggleRow(ev.transaction_id)}
                    onInvestigate={investigate}
                    scores={scores}
                    contribs={contribs}
                    weights={weights}
                  />
                );
              })}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={14} className="text-center text-textDim text-sm py-8">
                    {merged.length === 0
                      ? "No events yet. Start the generator on Overview to begin streaming."
                      : "No events match the current filter."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function FragmentRow({
  ev, open, onToggle, onInvestigate, scores, contribs, weights,
}: {
  ev: LiveEvent;
  open: boolean;
  onToggle: () => void;
  onInvestigate: (tab: "case" | "geo" | "report", ev: LiveEvent) => void;
  scores: any;
  contribs: any;
  weights: any;
}) {
  const layerNumber = (level: number | string | null | undefined) => {
    const v = Number(level ?? 0);
    return Number.isFinite(v) && v > 0 ? v.toFixed(0) : "—";
  };
  return (
    <>
      <tr
        className="border-t border-border hover:bg-panelHi cursor-pointer"
        onClick={onToggle}
      >
        <td className="px-3 py-2 text-textFaint text-center">{open ? "▾" : "▸"}</td>
        <td className="px-3 py-2 font-mono text-textDim whitespace-nowrap">{fmtTime(ev.timestamp)}</td>
        <td className="px-3 py-2 font-mono text-text">{ev.transaction_id.slice(0, 10)}</td>
        <td className="px-3 py-2 font-mono">
          <span className="text-text">{ev.sender_account.slice(0, 8)}</span>
          <span className="text-textFaint mx-1">→</span>
          <span className="text-text">{ev.receiver_account.slice(0, 8)}</span>
          <span className="text-textFaint ml-2 text-[10px]">
            {ev.sender_country}→{ev.receiver_country}
          </span>
        </td>
        <td className="px-3 py-2 text-right whitespace-nowrap">{fmtAmount(ev.amount)}</td>
        <td className="px-3 py-2 text-textDim">{ev.transaction_type}</td>
        <td className="px-3 py-2 text-right font-mono" style={{ color: LAYER_COLORS.rule }}>{layerNumber(ev.rule_score)}</td>
        <td className="px-3 py-2 text-right font-mono" style={{ color: LAYER_COLORS.graph }}>{layerNumber(scores.graph_score)}</td>
        <td className="px-3 py-2 text-right font-mono" style={{ color: LAYER_COLORS.supervised }}>{layerNumber(ev.supervised_score)}</td>
        <td className="px-3 py-2 text-right font-mono" style={{ color: LAYER_COLORS.anomaly }}>{layerNumber(ev.anomaly_score)}</td>
        <td className="px-3 py-2 text-right font-mono" style={{ color: LAYER_COLORS.tgn }}>{layerNumber(ev.tgn_score)}</td>
        <td className="px-3 py-2 text-right font-mono font-bold text-high">
          {Number(ev.transaction_risk_score ?? 0).toFixed(0)}
        </td>
        <td className="px-3 py-2">
          <Badge level={ev.risk_level} />
        </td>
        <td className="px-3 py-2">
          {ev.alerted && (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-high/15 text-high border border-high/30">
              ALERT
            </span>
          )}
        </td>
      </tr>
      {open && (
        <tr className="bg-bg/40 border-t border-border">
          <td colSpan={14} className="px-6 py-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs">
              {/* Layer breakdown */}
              <div>
                <Label>Per-layer breakdown</Label>
                <div className="space-y-1.5 font-mono">
                  {(["rule_score","graph_score","supervised_score","anomaly_score","tgn_score"] as const).map(k => {
                    const v = Number(scores[k] ?? 0);
                    const w = Number(weights[k] ?? 0);
                    const c = Number(contribs[k] ?? 0);
                    const color = {
                      rule_score: LAYER_COLORS.rule, graph_score: LAYER_COLORS.graph,
                      supervised_score: LAYER_COLORS.supervised, anomaly_score: LAYER_COLORS.anomaly,
                      tgn_score: LAYER_COLORS.tgn,
                    }[k];
                    return (
                      <div key={k} className="flex items-center gap-2">
                        <span className="w-24 text-textDim">{k.replace("_score", "")}</span>
                        <div className="flex-1 h-1.5 bg-panelHi rounded-full overflow-hidden">
                          <div className="h-full" style={{ width: `${Math.min(100, v)}%`, background: color }} />
                        </div>
                        <span style={{ color }} className="w-10 text-right">{v.toFixed(1)}</span>
                        <span className="text-textFaint w-12 text-right">×{w.toFixed(2)}</span>
                        <span className="text-text w-10 text-right">+{c.toFixed(1)}</span>
                      </div>
                    );
                  })}
                  <div className="flex items-center gap-2 pt-1 border-t border-border mt-1.5">
                    <span className="w-24 text-textDim">FUSED</span>
                    <span className="flex-1"></span>
                    <span className="text-high font-bold w-10 text-right">{Number(ev.transaction_risk_score ?? 0).toFixed(1)}</span>
                  </div>
                </div>
              </div>

              {/* Triggered rules + patterns */}
              <div>
                <Label>Triggered rules + patterns</Label>
                {(ev.triggered_rules || []).length === 0 && (ev.triggered_patterns || []).length === 0 ? (
                  <div className="text-textDim">No triggers — fusion driven by raw layer scores.</div>
                ) : (
                  <>
                    {(ev.triggered_rules || []).length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mb-2">
                        {ev.triggered_rules.map(r => (
                          <span key={r} className="px-2 py-0.5 rounded font-mono bg-layer1/15 text-layer1 border border-layer1/30 text-[10px]">
                            {r}
                          </span>
                        ))}
                      </div>
                    )}
                    {(ev.triggered_patterns || []).length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {ev.triggered_patterns.slice(0, 8).map(p => (
                          <span key={p} className="px-2 py-0.5 rounded-full font-mono bg-accent/15 text-violet-300 border border-accent/30 text-[10px]">
                            {p}
                          </span>
                        ))}
                      </div>
                    )}
                  </>
                )}
                {(ev.top_shap_features || []).length > 0 && (
                  <div className="mt-3">
                    <div className="text-textFaint text-[10px] mb-1 uppercase tracking-wider">Top SHAP features (L3)</div>
                    <div className="flex flex-wrap gap-1.5">
                      {ev.top_shap_features.slice(0, 5).map(f => (
                        <span key={f} className="px-2 py-0.5 rounded font-mono bg-teal/10 text-teal border border-teal/30 text-[10px]">
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Explanation + actions */}
              <div>
                <Label>Explanation</Label>
                <p className="text-textDim leading-relaxed mb-3 whitespace-pre-wrap font-mono text-[10px]">
                  {(ev.explanation || "—").slice(0, 380)}
                  {ev.explanation && ev.explanation.length > 380 && "…"}
                </p>
                <div className="flex flex-wrap gap-2">
                  <Btn onClick={(e) => { e?.stopPropagation(); onInvestigate("case", ev); }}>Investigate</Btn>
                  <Btn onClick={(e) => { e?.stopPropagation(); onInvestigate("geo", ev); }}>Map</Btn>
                  <Btn onClick={(e) => { e?.stopPropagation(); onInvestigate("report", ev); }}>Report</Btn>
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
