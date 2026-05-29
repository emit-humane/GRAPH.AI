"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useSelectionStore } from "@/stores/selectionStore";
import { Badge, Label, Panel, PillNote, SectionTitle, Stat, ViewHeader } from "@/components/ui";
import ScoreBreakdownBars from "@/components/ScoreBreakdownBars";
import CytoscapeChain from "@/components/CytoscapeChain";

export default function CasePage() {
  const txId = useSelectionStore(s => s.selectedTransactionId);
  const payload = useSelectionStore(s => s.selectedPayload);
  const [caseGraph, setCaseGraph] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!txId) return;
    setErr(null);
    api.caseGraph(txId).then(setCaseGraph).catch(e => setErr(e.message));
  }, [txId]);

  if (!txId || !payload) {
    return (
      <div className="max-w-4xl mx-auto">
        <ViewHeader title="Case Analysis" sub="Select a transaction from the Overview tab first." />
        <Panel className="bg-panelHi text-textDim">
          No case selected. Go to <a href="/overview" className="text-teal underline">Overview</a> and click
          “Investigate” on any flagged transaction.
        </Panel>
      </div>
    );
  }

  const ev = payload;
  return (
    <div className="max-w-5xl mx-auto">
      <ViewHeader
        title="Case Analysis"
        sub="Cross-layer breakdown of the selected transaction, with structural context from the live multigraph."
      />

      {/* Selected Card */}
      <Panel className="mb-4">
        <Label>Selected Transaction</Label>
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-xs px-3 py-1.5 rounded-lg bg-panelHi border border-border">
            {ev.transaction_id}
          </span>
          <Badge level={ev.risk_level} />
        </div>
        <div className="text-lg font-bold font-display mb-1">
          {ev.sender_account.slice(0, 12)} → {ev.receiver_account.slice(0, 12)} · {ev.transaction_type}
        </div>
        <p className="text-textDim text-sm leading-relaxed">{(ev.explanation || "").split("\n")[0]}</p>
      </Panel>

      {/* High-level alert banner */}
      <Panel className="mb-4 bg-high/10 border-high/40">
        <Label>Fraud Alert</Label>
        <div className="flex justify-between items-center">
          <div className="text-base font-bold font-display">
            {ev.transaction_id.slice(0, 14)} flagged as {ev.risk_level} (fused = {ev.transaction_risk_score.toFixed(1)})
          </div>
          <span className="w-2.5 h-2.5 rounded-full bg-high shadow-[0_0_12px_#e23d6e]" />
        </div>
      </Panel>

      {/* Top-line stats */}
      <div className="grid grid-cols-4 gap-3.5 mb-4">
        <Panel className="p-4"><Stat label="Amount" value={`₹${ev.amount.toLocaleString("en-IN")}`} /></Panel>
        <Panel className="p-4"><Stat label="Patterns" value={ev.triggered_patterns?.length ?? 0} /></Panel>
        <Panel className="p-4"><Stat label="Transaction Risk" value={ev.transaction_risk_score.toFixed(1)} valueColor="#e23d6e" /></Panel>
        <Panel className="p-4"><Stat label="Group Risk" value={ev.group_risk_score.toFixed(1)} valueColor="#d99a2b" /></Panel>
      </div>

      {/* THE 5-layer breakdown — this is the page's centrepiece */}
      <ScoreBreakdownBars
        scoreBreakdown={ev.score_breakdown}
        topShapFeatures={ev.top_shap_features}
        triggeredRules={ev.triggered_rules}
        anomalyDrivers={ev.anomaly_drivers || ev.score_breakdown?.anomaly_drivers}
        temporalExplanations={ev.score_breakdown?.temporal_graph_explanations || (ev.explanation || "").split("\n").slice(1, 4)}
        fusedScore={ev.transaction_risk_score}
        riskLevel={ev.risk_level}
      />

      {/* Cytoscape — case-specific 2-hop chain */}
      <Panel className="mb-4">
        <div className="flex justify-between items-center mb-3">
          <div>
            <Label>Fraud Network</Label>
            <SectionTitle>Case Network Graph</SectionTitle>
          </div>
          <PillNote>2-hop neighbourhood from D2</PillNote>
        </div>
        {err && <div className="text-high text-sm font-mono mb-2">{err}</div>}
        {caseGraph ? (
          <>
            <CytoscapeChain
              elements={caseGraph.elements}
              focal={caseGraph.focal}
              layout="cose"
              height={420}
            />
            <div className="grid grid-cols-3 gap-3 mt-3 text-xs font-mono">
              <Panel className="p-3">
                <Label>Cycle</Label>
                <div className="text-text">{caseGraph.metadata?.edge_creates_cycle ? `Yes (${caseGraph.metadata.cycle_length}-hop)` : "No"}</div>
              </Panel>
              <Panel className="p-3">
                <Label>Shared Community</Label>
                <div className="text-text">{caseGraph.metadata?.shared_community ? "Yes" : "No"}</div>
              </Panel>
              <Panel className="p-3">
                <Label>2-hop nodes</Label>
                <div className="text-text">{(caseGraph.elements?.nodes?.length ?? 0)}</div>
              </Panel>
            </div>
          </>
        ) : (
          <div className="text-textDim text-sm">Loading graph …</div>
        )}
      </Panel>

      {/* Investigation Notes */}
      <Panel>
        <Label>Investigation Notes</Label>
        <SectionTitle>Why This Was Flagged</SectionTitle>
        <ul className="space-y-2">
          {(ev.explanation || "").split("\n").slice(0, 6).map((line: string, i: number) => (
            <li key={i} className="border border-border rounded-lg px-4 py-3 text-sm font-display">{line}</li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
