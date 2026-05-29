"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Label, Panel, SectionTitle, Stat, ViewHeader } from "@/components/ui";

const RISK_COLORS: Record<string, string> = {
  Low: "#2bb673",
  Medium: "#d99a2b",
  High: "#e23d6e",
  Critical: "#a21caf",
};

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    let cancel = false;
    const tick = () => api.evaluation().then(d => !cancel && setData(d)).catch(() => {});
    tick();
    const id = setInterval(tick, 4000);
    return () => { cancel = true; clearInterval(id); };
  }, []);

  if (!data) return <div className="max-w-4xl mx-auto"><ViewHeader title="Analytics" sub="Loading …" /></div>;

  const dist = data.risk_distribution || {};
  const total = (dist.Low || 0) + (dist.Medium || 0) + (dist.High || 0) + (dist.Critical || 0);

  return (
    <div className="max-w-5xl mx-auto">
      <ViewHeader
        title="Analytics"
        sub="Live aggregates from the in-memory event buffer + persisted alert log."
      />

      <div className="grid grid-cols-4 gap-3.5 mb-4">
        <Panel className="p-4"><Stat label="Events Seen" value={data.events_seen?.toLocaleString() ?? 0} /></Panel>
        <Panel className="p-4"><Stat label="Alerts Total" value={data.alerts_total?.toLocaleString() ?? 0} valueColor="#e23d6e" /></Panel>
        <Panel className="p-4"><Stat label="Mean Risk" value={(data.mean_risk_score ?? 0).toFixed(1)} /></Panel>
        <Panel className="p-4"><Stat label="Max Risk" value={(data.max_risk_score ?? 0).toFixed(1)} valueColor="#d99a2b" /></Panel>
      </div>

      <Panel className="mb-4">
        <Label>Risk Distribution</Label>
        <SectionTitle>By Risk Level</SectionTitle>
        <div className="space-y-2">
          {(["Low", "Medium", "High", "Critical"]).map(k => {
            const n = dist[k] || 0;
            const pct = total > 0 ? (n / total) * 100 : 0;
            return (
              <div key={k}>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span style={{ color: RISK_COLORS[k] }}>{k}</span>
                  <span className="text-textDim">{n} ({pct.toFixed(1)}%)</span>
                </div>
                <div className="h-2 bg-panelHi rounded-full overflow-hidden">
                  <div className="h-full transition-all duration-500"
                    style={{ width: `${pct}%`, background: RISK_COLORS[k] }} />
                </div>
              </div>
            );
          })}
        </div>
      </Panel>

      <div className="grid grid-cols-2 gap-3.5">
        <Panel>
          <Label>Rule Firing Counts</Label>
          <SectionTitle>Layer 1</SectionTitle>
          {Object.entries(data.rule_counts || {}).slice(0, 15).map(([k, v]: any) => (
            <div key={k} className="flex justify-between text-xs font-mono py-1 border-b border-border">
              <span className="text-layer1">{k}</span>
              <span className="text-textDim">{v}</span>
            </div>
          ))}
          {Object.keys(data.rule_counts || {}).length === 0 && (
            <div className="text-textDim text-sm">No rules fired yet.</div>
          )}
        </Panel>

        <Panel>
          <Label>Cross-Layer Patterns</Label>
          <SectionTitle>Consolidated</SectionTitle>
          {Object.entries(data.pattern_counts || {}).slice(0, 15).map(([k, v]: any) => (
            <div key={k} className="flex justify-between text-xs font-mono py-1 border-b border-border">
              <span className="text-text">{k}</span>
              <span className="text-textDim">{v}</span>
            </div>
          ))}
          {Object.keys(data.pattern_counts || {}).length === 0 && (
            <div className="text-textDim text-sm">No patterns yet.</div>
          )}
        </Panel>
      </div>
    </div>
  );
}
