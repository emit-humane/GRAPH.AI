"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useSelectionStore } from "@/stores/selectionStore";
import { Badge, Btn, Label, Panel, SectionTitle, Stat, ViewHeader } from "@/components/ui";
import ScoreBreakdownBars from "@/components/ScoreBreakdownBars";

export default function ReportPage() {
  const txId = useSelectionStore(s => s.selectedTransactionId);
  const [report, setReport] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!txId) return;
    setErr(null);
    api.report(txId).then(setReport).catch(e => setErr(e.message));
  }, [txId]);

  if (!txId) {
    return (
      <div className="max-w-4xl mx-auto">
        <ViewHeader title="FIU Reporting" sub="Select a transaction first." />
        <Panel className="bg-panelHi text-textDim">
          No case selected. Go to <a href="/overview" className="text-teal underline">Overview</a> first.
        </Panel>
      </div>
    );
  }

  if (err) {
    return (
      <div className="max-w-4xl mx-auto">
        <ViewHeader title="FIU Reporting" sub="Could not load report." />
        <Panel className="bg-high/10 border-high/40 text-high text-sm font-mono">{err}</Panel>
      </div>
    );
  }

  const tx = report?.transaction || {};
  return (
    <div className="max-w-5xl mx-auto">
      <ViewHeader
        title="FIU Reporting"
        sub="Build a regulator-ready report for the selected transaction. Downloads include the full evidence package."
      />

      <Panel className="mb-4">
        <div className="flex justify-between items-start mb-5">
          <div>
            <Label>FIU Filing Preview</Label>
            <SectionTitle>Fraud Report &amp; Evidence Package</SectionTitle>
          </div>
          <div className="flex flex-col gap-2.5 items-end">
            <a href={api.reportZipUrl(txId)} target="_blank">
              <Btn variant="accent">Download Evidence Package (.zip)</Btn>
            </a>
            <a href={api.reportPdfUrl(txId)} target="_blank">
              <Btn variant="primary">Download Report (.pdf)</Btn>
            </a>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-3.5 mb-4">
          <Panel className="p-4"><Stat label="Case ID" value={<span className="font-mono text-base">{report?.case_id}</span>} /></Panel>
          <Panel className="p-4"><Stat label="Transaction" value={<span className="font-mono text-base">{tx.transaction_id?.slice(0, 14)}</span>} /></Panel>
          <Panel className="p-4"><Stat label="Amount" value={`₹${(tx.amount ?? 0).toLocaleString("en-IN")}`} /></Panel>
          <Panel className="p-4"><Stat label="Risk" value={(tx.transaction_risk_score ?? 0).toFixed(1)} valueColor="#e23d6e" /></Panel>
        </div>

        <ReportRow label="Risk Level">
          <Badge level={tx.risk_level || "Low"} />
        </ReportRow>

        <ReportRow label="Triggered Rules (Layer 1)">
          <div className="flex flex-wrap gap-2">
            {(tx.triggered_rules || []).map((r: string) => (
              <span key={r} className="px-2 py-0.5 rounded text-[11px] font-mono bg-layer1/15 text-layer1 border border-layer1/30">
                {r}
              </span>
            ))}
          </div>
        </ReportRow>

        <ReportRow label="Triggered Patterns (consolidated)">
          <div className="flex flex-wrap gap-2">
            {(tx.triggered_patterns || []).map((p: string) => (
              <span key={p} className="px-3 py-1 rounded-full bg-accent/15 text-violet-300 font-display text-xs font-semibold border border-accent/30">
                {p}
              </span>
            ))}
          </div>
        </ReportRow>

        <ReportRow label="Recommendation">
          <div className="text-base font-bold font-display mb-2">Continue tracing downstream beneficiaries</div>
          <p className="text-textDim text-sm">{report?.filing_notes}</p>
        </ReportRow>
      </Panel>

      <ScoreBreakdownBars
        scoreBreakdown={tx.score_breakdown}
        topShapFeatures={tx.top_shap_features}
        triggeredRules={tx.triggered_rules}
        fusedScore={tx.transaction_risk_score}
        riskLevel={tx.risk_level}
      />

      <Panel>
        <Label>Suggested Filing Notes</Label>
        <p className="text-textDim text-sm leading-relaxed">{report?.filing_notes}</p>
      </Panel>
    </div>
  );
}

function ReportRow({ label, children }: { label: string; children: any }) {
  return (
    <div className="border-t border-border pt-4 mt-4">
      <Label>{label}</Label>
      {children}
    </div>
  );
}
