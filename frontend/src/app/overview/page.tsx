"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { useSelectionStore } from "@/stores/selectionStore";
import { Badge, Btn, ConnectionDot, Label, Panel, PillNote, SectionTitle, Stat, ViewHeader } from "@/components/ui";

export default function OverviewPage() {
  const router = useRouter();
  const { events, alerts, connected } = useEventStream(40);
  const [streaming, setStreaming] = useState(false);
  const [minAmt, setMinAmt] = useState(100);
  const [maxAmt, setMaxAmt] = useState(5_000_000);
  const [status, setStatus] = useState<any>(null);
  const setSelected = useSelectionStore(s => s.setSelected);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const s = await api.generatorStatus();
        if (!cancel) {
          setStatus(s);
          setStreaming(Boolean(s?.running));
        }
      } catch (e) { /* ignore */ }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => { cancel = true; clearInterval(id); };
  }, []);

  const start = async () => {
    await api.generatorStart({ min_amount: minAmt, max_amount: maxAmt, interval_seconds: 1.5 });
    setStreaming(true);
  };
  const stop = async () => {
    await api.generatorStop();
    setStreaming(false);
  };

  const investigate = (tab: "case" | "geo" | "report", ev: any) => {
    setSelected(ev.transaction_id, ev);
    router.push(`/${tab}`);
  };

  const latest = events[0];

  return (
    <div className="max-w-4xl mx-auto">
      <ViewHeader
        title="Fraud Overview"
        sub="Review recently flagged transactions and choose one case to investigate across the rest of the workspace."
      />

      <Panel className="mb-4">
        <div className="flex justify-between items-start">
          <div>
            <Label>Live Generator</Label>
            <div className="text-xl font-bold font-display">Live 5-Layer Detection Pipeline</div>
          </div>
          <div className="flex items-center gap-2">
            <ConnectionDot connected={connected} />
            <span className="text-xs font-mono text-textDim">
              {connected ? "stream connected" : "stream offline"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3.5 my-4">
          <div>
            <Label>Minimum Amount</Label>
            <input type="number" value={minAmt} onChange={e => setMinAmt(+e.target.value)}
              className="w-full bg-bg border border-border rounded-lg p-2.5 text-text font-mono outline-none" />
          </div>
          <div>
            <Label>Maximum Amount</Label>
            <input type="number" value={maxAmt} onChange={e => setMaxAmt(+e.target.value)}
              className="w-full bg-bg border border-border rounded-lg p-2.5 text-text font-mono outline-none" />
          </div>
        </div>

        <p className="text-textDim text-sm leading-relaxed mb-4">
          When started, the backend pulls events from <span className="font-mono text-text">stream_transactions.csv</span> at
          a 1–10s cadence and runs the full pipeline (D1 → D2 → D3 → D4 → D5 → D6 → D7 → D8 → P2) for every transaction.
          Heavy layers run in a thread pool to keep the async loop responsive.
        </p>

        <div className="flex gap-3 mb-5">
          <Btn variant="primary" onClick={start} disabled={streaming}>Start Generator</Btn>
          <Btn onClick={stop} disabled={!streaming}>Stop Generator</Btn>
          <span className={`px-3 py-2 rounded-lg text-xs font-semibold font-display ${streaming ? "bg-teal/10 text-teal border border-teal/30" : "bg-panelHi text-textDim border border-border"}`}>
            {streaming ? "Streaming" : "Stopped"}
          </span>
        </div>

        <div className="grid grid-cols-4 gap-3.5">
          <Panel className="p-4"><Stat label="Events" value={status?.events_emitted ?? 0} /></Panel>
          <Panel className="p-4"><Stat label="Alerts" value={status?.alerts_emitted ?? 0} valueColor="#e23d6e" /></Panel>
          <Panel className="p-4"><Stat label="Stream Cadence" value={`${status?.config?.interval_seconds ?? 1.5}s`} /></Panel>
          <Panel className="p-4"><Stat label="Driver" value={`${(status?.driver_event_count ?? 0).toLocaleString()}`} /></Panel>
        </div>

        {latest && (
          <Panel className="p-4 mt-3.5 bg-panelHi">
            <Label>Latest Transaction</Label>
            <div className="font-mono text-sm">
              <span className="text-textDim">{latest.transaction_id.slice(0, 8)}</span> ·
              <span className="ml-1">{latest.sender_account.slice(0, 8)} → {latest.receiver_account.slice(0, 8)}</span>
              <span className="ml-2 text-textDim">₹{latest.amount.toLocaleString("en-IN")}</span>
              <span className="ml-2 text-textDim">risk={latest.transaction_risk_score.toFixed(1)} ({latest.risk_level})</span>
            </div>
          </Panel>
        )}
      </Panel>

      <Panel>
        <div className="flex justify-between items-start mb-4">
          <div>
            <Label>Recently Flagged</Label>
            <SectionTitle>Flagged Transactions ({alerts.length} live)</SectionTitle>
          </div>
          <PillNote>Select one transaction to investigate in the other tabs</PillNote>
        </div>

        <div className="flex flex-col gap-3">
          {(alerts.length > 0 ? alerts : events).slice(0, 12).map(ev => (
            <div key={ev.transaction_id}
              onClick={() => setSelected(ev.transaction_id, ev)}
              className="border border-border bg-panelHi rounded-2xl p-4 cursor-pointer hover:border-high/40 transition">
              <div className="flex justify-between items-center mb-2">
                <span className="font-mono text-[11px] text-textDim">{ev.transaction_id.slice(0, 14)}</span>
                <Badge level={ev.risk_level} />
              </div>
              <div className="text-base font-bold font-display mb-1">
                {ev.sender_account.slice(0, 10)} → {ev.receiver_account.slice(0, 10)}
              </div>
              <p className="text-textDim text-xs mb-3 line-clamp-2">
                {(ev.explanation || "").split("\n")[0]}
              </p>
              <div className="flex justify-between mb-2.5 text-xs font-mono">
                <span>₹{ev.amount.toLocaleString("en-IN")}</span>
                <span className="text-textDim">{ev.timestamp?.slice(0, 19)}</span>
              </div>
              <div className="flex gap-1.5 text-[10px] font-mono mb-3">
                <span className="px-1.5 py-0.5 bg-layer1/15 text-layer1 rounded">L1 {ev.rule_score.toFixed(0)}</span>
                <span className="px-1.5 py-0.5 bg-layer3/15 text-layer3 rounded">L3 {(ev.supervised_score ?? 0).toFixed(0)}</span>
                <span className="px-1.5 py-0.5 bg-layer4/15 text-layer4 rounded">L4 {(ev.anomaly_score ?? 0).toFixed(0)}</span>
                <span className="px-1.5 py-0.5 bg-layer5/15 text-layer5 rounded">L5 {(ev.tgn_score ?? 0).toFixed(0)}</span>
                <span className="px-1.5 py-0.5 bg-high/15 text-high rounded ml-auto">FUSED {ev.transaction_risk_score.toFixed(0)}</span>
              </div>
              <div className="flex gap-2">
                <Btn onClick={(e) => { e?.stopPropagation(); investigate("case", ev); }}>Investigate</Btn>
                <Btn onClick={(e) => { e?.stopPropagation(); investigate("geo", ev); }}>Open Map</Btn>
                <Btn onClick={(e) => { e?.stopPropagation(); investigate("report", ev); }}>Generate Report</Btn>
              </div>
            </div>
          ))}
          {events.length === 0 && (
            <Panel className="bg-panelHi text-textDim text-sm">
              No live events yet. Press <span className="text-text font-semibold">Start Generator</span> to begin streaming.
            </Panel>
          )}
        </div>
      </Panel>
    </div>
  );
}
