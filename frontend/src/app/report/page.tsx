"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useSelectionStore } from "@/stores/selectionStore";
import { Badge, Btn, Label, Panel, PillNote, SectionTitle, Stat, ViewHeader } from "@/components/ui";
import ScoreBreakdownBars from "@/components/ScoreBreakdownBars";
import RiskGauge from "@/components/RiskGauge";

interface Indicator {
  rule_id: string;
  title: string;
  severity: "Low" | "Medium" | "High" | "Critical";
  source: string;
  regulatory?: string;
  detail?: string;
}
interface TimelineItem {
  transaction_id: string;
  timestamp: string;
  sender_account: string;
  receiver_account: string;
  amount: number;
  transaction_type?: string;
  risk_score: number;
  risk_level?: string;
  alerted: boolean;
  is_focal?: boolean;
}
interface Recommendation {
  id: string;
  title: string;
  detail: string;
  priority: "Low" | "Medium" | "High" | "Critical";
}
interface Citation {
  code: string;
  title: string;
  detail: string;
}

const SEVERITY_COLOR: Record<string, { bg: string; border: string; text: string; dot: string }> = {
  Critical: { bg: "rgba(226,61,110,0.12)", border: "#e23d6e55", text: "#e23d6e", dot: "#e23d6e" },
  High:     { bg: "rgba(226,61,110,0.08)", border: "#e23d6e44", text: "#e23d6e", dot: "#e23d6e" },
  Medium:   { bg: "rgba(217,154,43,0.10)", border: "#d99a2b44", text: "#fbbf24", dot: "#d99a2b" },
  Low:      { bg: "rgba(43,182,115,0.10)", border: "#2bb67344", text: "#34d399", dot: "#2bb673" },
};

function humanAmount(amount: number): string {
  if (!Number.isFinite(amount)) return "—";
  if (amount >= 1e7) return `₹${(amount / 1e7).toFixed(2)} Cr`;
  if (amount >= 1e5) return `₹${(amount / 1e5).toFixed(2)} L`;
  return `₹${amount.toLocaleString("en-IN")}`;
}

function fmtTime(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(+d)) return iso;
  return d.toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "medium" });
}

export default function ReportPage() {
  const txId = useSelectionStore(s => s.selectedTransactionId);
  const [report, setReport] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState<Record<string, boolean>>({});

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

  if (!report) {
    return (
      <div className="max-w-4xl mx-auto">
        <ViewHeader title="FIU Reporting" sub="Loading evidence package …" />
        <Panel className="text-textDim text-sm">Loading …</Panel>
      </div>
    );
  }

  const tx = report.transaction || {};
  const narrative = report.narrative || {};
  const indicators: Indicator[] = report.indicators || [];
  const timeline: TimelineItem[] = report.timeline || [];
  const peer = report.peer_comparison || { available: false };
  const recommendations: Recommendation[] = report.recommendations || [];
  const citations: Citation[] = report.regulatory_context || [];

  return (
    <div className="max-w-6xl mx-auto pb-12">
      <ViewHeader
        title="FIU Reporting"
        sub="Regulator-ready summary of the selected case, with red-flag evidence, timeline, peer comparison, and recommended next actions."
      />

      {/* HERO — Case ID + risk gauge + key facts + download buttons */}
      <Panel className="mb-4">
        <div className="flex flex-wrap items-start gap-6 justify-between">
          <div className="flex-1 min-w-[260px]">
            <Label>Case Card</Label>
            <SectionTitle>
              <span className="font-mono">{report.case_id}</span>
            </SectionTitle>
            <div className="flex gap-2 mb-3">
              <Badge level={tx.risk_level || "Low"} />
              {tx.alerted && (
                <span className="px-3 py-1 rounded-full text-[11px] font-mono bg-high/15 text-high border border-high/40">
                  ALERT PERSISTED
                </span>
              )}
              <PillNote>
                Group risk {Number(tx.group_risk_score ?? 0).toFixed(0)} ({tx.risk_level_group || "Low"})
              </PillNote>
            </div>
            <div className="text-sm text-textDim leading-relaxed">
              {narrative.what_happened}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5 mt-5">
              <Stat label="Amount" value={humanAmount(tx.amount ?? 0)} />
              <Stat label="Type" value={tx.transaction_type || "—"} />
              <Stat label="Channel" value={tx.payment_channel || "—"} />
              <Stat label="When" value={<span className="text-base font-mono">{fmtTime(tx.timestamp).split(",")[1]?.trim()}</span>} />
            </div>
          </div>

          <div className="flex flex-col items-center gap-4">
            <RiskGauge
              score={Number(tx.transaction_risk_score ?? 0)}
              level={tx.risk_level}
              caption="FUSED RISK · 0-100"
            />
            <div className="flex flex-col gap-2 w-[200px]">
              <a href={api.reportPdfUrl(txId)} target="_blank" rel="noreferrer">
                <Btn variant="primary">Download PDF</Btn>
              </a>
              <a href={api.reportZipUrl(txId)} target="_blank" rel="noreferrer">
                <Btn variant="accent">Evidence ZIP</Btn>
              </a>
            </div>
          </div>
        </div>
      </Panel>

      {/* PARTIES — sender + arrow + receiver */}
      <Panel className="mb-4">
        <Label>Parties</Label>
        <SectionTitle>Counterparties</SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-center">
          <PartyCard
            role="Sender"
            account={tx.sender_account || "—"}
            bank={tx.sender_bank || "—"}
            country={tx.sender_country || "—"}
          />
          <div className="flex flex-col items-center px-2">
            <div className="text-xs font-mono text-textDim mb-1">{humanAmount(tx.amount ?? 0)}</div>
            <div className="text-2xl" style={{ color: "#e23d6e" }}>▶</div>
            <div className="text-[10px] font-mono text-textFaint mt-1">{tx.transaction_type || ""}</div>
          </div>
          <PartyCard
            role="Receiver"
            account={tx.receiver_account || "—"}
            bank={tx.receiver_bank || "—"}
            country={tx.receiver_country || "—"}
          />
        </div>
      </Panel>

      {/* WHY FLAGGED — narrative + rule list (human-readable) */}
      <Panel className="mb-4">
        <Label>Why this was flagged</Label>
        <SectionTitle>Executive summary</SectionTitle>
        <div className="space-y-3 text-sm leading-relaxed text-text">
          <p>{narrative.why_flagged}</p>
          <p className="text-textDim">{narrative.which_rules}</p>
          <p className="text-textDim italic">{narrative.next_steps}</p>
        </div>
      </Panel>

      {/* INDICATORS — red-flag grid */}
      <Panel className="mb-4">
        <Label>Red-flag indicators</Label>
        <SectionTitle>What we found ({indicators.length})</SectionTitle>
        {indicators.length === 0 ? (
          <div className="text-textDim text-sm">No structured indicators raised. See Layer-scores below.</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {indicators.map((ind, i) => {
              const sev = SEVERITY_COLOR[ind.severity] ?? SEVERITY_COLOR.Low;
              return (
                <div
                  key={`${ind.rule_id}-${i}`}
                  className="rounded-xl border p-4"
                  style={{ background: sev.bg, borderColor: sev.border }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ background: sev.dot, boxShadow: `0 0 8px ${sev.dot}` }}
                      />
                      <span className="text-xs font-mono text-textDim">{ind.rule_id}</span>
                    </div>
                    <span className="text-[10px] tracking-wider font-mono uppercase" style={{ color: sev.text }}>
                      {ind.severity}
                    </span>
                  </div>
                  <div className="font-bold font-display text-sm leading-snug mb-1.5" style={{ color: sev.text }}>
                    {ind.title}
                  </div>
                  {ind.detail && (
                    <div className="text-xs text-textDim leading-relaxed mb-2">{ind.detail}</div>
                  )}
                  <div className="flex flex-wrap gap-2 text-[10px] font-mono text-textFaint mt-2">
                    <span className="bg-bg/60 px-2 py-0.5 rounded">{ind.source}</span>
                    {ind.regulatory && (
                      <span className="bg-bg/60 px-2 py-0.5 rounded">{ind.regulatory}</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      {/* PEER COMPARISON */}
      {peer.available && (
        <Panel className="mb-4">
          <Label>Peer comparison</Label>
          <SectionTitle>How this differs from the sender's normal</SectionTitle>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3.5">
            <Stat label="This tx" value={humanAmount(peer.this_amount)} valueColor="#e23d6e" />
            <Stat label="Sender mean" value={humanAmount(peer.mean_amount)} />
            <Stat label="Sender median" value={humanAmount(peer.median_amount)} />
            <Stat label="Sender max" value={humanAmount(peer.max_amount)} />
            <Stat
              label="× of mean"
              value={`${peer.multiple_of_mean.toFixed(1)}×`}
              valueColor={peer.is_outlier ? "#e23d6e" : "#34d399"}
            />
          </div>
          {peer.is_outlier && (
            <div className="mt-3 text-sm text-high">
              ⚠ This transaction is <b>{peer.multiple_of_mean.toFixed(1)}×</b> the sender's recent mean transaction size — material outlier.
            </div>
          )}
        </Panel>
      )}

      {/* SCORE BREAKDOWN — keep existing component */}
      <ScoreBreakdownBars
        scoreBreakdown={tx.score_breakdown}
        topShapFeatures={tx.top_shap_features}
        triggeredRules={tx.triggered_rules}
        fusedScore={tx.transaction_risk_score}
        riskLevel={tx.risk_level}
      />

      {/* TIMELINE */}
      {timeline.length > 0 && (
        <Panel className="mb-4">
          <Label>Counterparty activity timeline</Label>
          <SectionTitle>Recent activity for sender / receiver ({timeline.length} events)</SectionTitle>
          <div className="relative pl-5 border-l-2 border-border space-y-3">
            {timeline.map((t, i) => (
              <div key={`${t.transaction_id}-${i}`} className="relative">
                <span
                  className="absolute -left-[26px] top-1.5 w-3 h-3 rounded-full"
                  style={{
                    background: t.is_focal ? "#e23d6e" : (t.alerted ? "#d99a2b" : "#3b82f6"),
                    boxShadow: t.is_focal ? "0 0 10px #e23d6e" : undefined,
                  }}
                />
                <div className={`rounded-lg p-3 text-xs ${t.is_focal ? "bg-high/10 border border-high/30" : "bg-panelHi border border-border"}`}>
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="font-mono text-textDim">{fmtTime(t.timestamp)}</span>
                    <span className="font-mono text-textFaint">{t.transaction_id?.slice(0, 10)}…</span>
                  </div>
                  <div className="font-mono">
                    <span className={t.is_focal ? "text-high font-bold" : ""}>{t.sender_account?.slice(0, 10)}…</span>
                    <span className="text-textDim mx-1.5">→</span>
                    <span className={t.is_focal ? "text-high font-bold" : ""}>{t.receiver_account?.slice(0, 10)}…</span>
                    <span className="mx-2 text-textFaint">·</span>
                    <span>{humanAmount(t.amount)}</span>
                    <span className="mx-2 text-textFaint">·</span>
                    <span style={{
                      color: (t.risk_level === "High" || t.risk_level === "Critical") ? "#e23d6e" :
                        t.risk_level === "Medium" ? "#fbbf24" : "#34d399",
                    }}>
                      risk={t.risk_score.toFixed(0)} ({t.risk_level || "Low"})
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* RECOMMENDATIONS — checklist */}
      <Panel className="mb-4">
        <Label>Recommended actions</Label>
        <SectionTitle>Investigator checklist</SectionTitle>
        <div className="space-y-2">
          {recommendations.map(rec => {
            const sev = SEVERITY_COLOR[rec.priority] ?? SEVERITY_COLOR.Medium;
            const checked = !!done[rec.id];
            return (
              <label
                key={rec.id}
                className="flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition"
                style={{
                  background: checked ? "rgba(43,182,115,0.06)" : "transparent",
                  borderColor: checked ? "#2bb67344" : "#1f2940",
                }}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={e => setDone(d => ({ ...d, [rec.id]: e.target.checked }))}
                  className="mt-1 accent-teal"
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span
                      className={`font-bold font-display text-sm ${checked ? "line-through text-textFaint" : ""}`}
                    >
                      {rec.title}
                    </span>
                    <span
                      className="text-[10px] font-mono tracking-wider uppercase px-1.5 py-0.5 rounded"
                      style={{ color: sev.text, background: sev.bg, border: `1px solid ${sev.border}` }}
                    >
                      {rec.priority}
                    </span>
                  </div>
                  <div className={`text-xs text-textDim leading-relaxed ${checked ? "line-through" : ""}`}>
                    {rec.detail}
                  </div>
                </div>
              </label>
            );
          })}
        </div>
        {Object.values(done).filter(Boolean).length > 0 && (
          <div className="mt-3 text-xs font-mono text-textDim">
            {Object.values(done).filter(Boolean).length} / {recommendations.length} actions completed
          </div>
        )}
      </Panel>

      {/* REGULATORY CONTEXT */}
      {citations.length > 0 && (
        <Panel className="mb-4">
          <Label>Regulatory context</Label>
          <SectionTitle>Citations</SectionTitle>
          <div className="space-y-3">
            {citations.map((c, i) => (
              <div key={i} className="border-l-4 pl-3 py-1" style={{ borderColor: "#a78bfa" }}>
                <div className="font-mono text-xs text-violet-300 mb-0.5">{c.code}</div>
                <div className="font-bold font-display text-sm">{c.title}</div>
                <div className="text-xs text-textDim leading-relaxed mt-1">{c.detail}</div>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {/* FILING NOTES — terse one-liner for the FIU template */}
      <Panel>
        <Label>Suggested filing notes (for STR free-text field)</Label>
        <p className="text-textDim text-sm leading-relaxed font-mono">{report.filing_notes}</p>
      </Panel>
    </div>
  );
}

function PartyCard({ role, account, bank, country }: { role: string; account: string; bank: string; country: string }) {
  return (
    <div className="rounded-xl border border-border bg-panelHi p-4">
      <div className="text-[10px] tracking-[0.25em] uppercase text-textDim font-mono mb-2">{role}</div>
      <div className="font-mono text-sm font-bold mb-2 break-all">{account.slice(0, 24)}{account.length > 24 ? "…" : ""}</div>
      <div className="flex gap-2 text-[11px] font-mono">
        <span className="px-2 py-0.5 rounded bg-bg/60 text-textDim border border-border">{bank}</span>
        <span
          className="px-2 py-0.5 rounded border"
          style={{
            background: country === "IN" ? "rgba(43,182,115,0.10)" : "rgba(217,154,43,0.10)",
            color: country === "IN" ? "#34d399" : "#fbbf24",
            borderColor: country === "IN" ? "#2bb67355" : "#d99a2b55",
          }}
        >
          {country}
        </span>
      </div>
    </div>
  );
}
