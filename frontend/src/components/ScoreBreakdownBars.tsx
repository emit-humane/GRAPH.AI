"use client";

import { Label, Panel, SectionTitle } from "./ui";

interface Props {
  scoreBreakdown: any;
  topShapFeatures?: string[];
  triggeredRules?: string[];
  anomalyDrivers?: string[];
  temporalExplanations?: string[];
  fusedScore?: number;
  riskLevel?: string;
}

const LAYER_ROWS = [
  // v2 fusion (rule + graph dominant): Layer 1 is PRIMARY at 0.30 weight.
  { key: "rule_score", name: "Rule Engine", layer: "L1", color: "#22d3ee", rationale: "PRIMARY auditable AML rules", primary: true },
  { key: "graph_score", name: "Graph Analytics", layer: "L2", color: "#a78bfa", rationale: "structural AML" },
  { key: "supervised_score", name: "Supervised ML", layer: "L3", color: "#34d399", rationale: "calibrated ML detector" },
  { key: "anomaly_score", name: "Behavioural Anomaly", layer: "L4", color: "#fbbf24", rationale: "unseen behaviour" },
  { key: "tgn_score", name: "TGN / GNN", layer: "L5", color: "#f472b6", rationale: "temporal specialist" },
];

export default function ScoreBreakdownBars({
  scoreBreakdown, topShapFeatures, triggeredRules, anomalyDrivers, temporalExplanations,
  fusedScore, riskLevel,
}: Props) {
  const scores = scoreBreakdown?.scores || {};
  const weights = scoreBreakdown?.weights || {};
  const contribs = scoreBreakdown?.weighted_contributions || {};

  return (
    <Panel className="mb-4">
      <div className="flex justify-between items-start mb-4">
        <div>
          <Label>Cross-Layer Score Breakdown</Label>
          <SectionTitle>
            Transaction Risk = {fusedScore != null ? fusedScore.toFixed(1) : "—"}
            {riskLevel && (
              <span className="ml-3 text-xs font-mono text-textDim">({riskLevel})</span>
            )}
          </SectionTitle>
        </div>
        <span className="px-3 py-1.5 rounded-lg text-xs font-mono bg-panelHi border border-border text-textDim">
          weights sum = 1.00
        </span>
      </div>

      <div className="space-y-3.5">
        {LAYER_ROWS.map(row => {
          const score = Number(scores[row.key] ?? 0);
          const weight = Number(weights[row.key] ?? 0);
          const contrib = Number(contribs[row.key] ?? 0);
          const widthPct = Math.max(0, Math.min(100, score));
          return (
            <div key={row.key}>
              <div className="flex justify-between text-xs font-mono mb-1.5">
                <span className="text-textDim">
                  <span style={{ color: row.color }}>{row.layer}</span>
                  <span className="ml-2">{row.name}</span>
                  {row.primary && (
                    <span className="ml-2 px-1.5 py-0.5 text-[10px] bg-teal/20 text-teal rounded">PRIMARY</span>
                  )}
                  <span className="ml-2 text-textFaint">— {row.rationale}</span>
                </span>
                <span>
                  <span style={{ color: row.color }}>{score.toFixed(1)}</span>
                  <span className="text-textFaint mx-1">·</span>
                  <span className="text-textDim">w={weight.toFixed(2)}</span>
                  <span className="text-textFaint mx-1">→</span>
                  <span className="text-text">+{contrib.toFixed(2)}</span>
                </span>
              </div>
              <div className="h-2 bg-panelHi rounded-full overflow-hidden">
                <div
                  className="h-full transition-all duration-500"
                  style={{ width: `${widthPct}%`, background: row.color, boxShadow: row.primary ? `0 0 12px ${row.color}` : undefined }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {(topShapFeatures && topShapFeatures.length > 0) && (
        <div className="mt-5 pt-4 border-t border-border">
          <Label>Top SHAP Features (Layer 3 — Primary)</Label>
          <div className="flex flex-wrap gap-2">
            {topShapFeatures.slice(0, 8).map(f => (
              <span key={f} className="px-2.5 py-1 rounded-md text-[11px] font-mono bg-teal/10 text-teal border border-teal/30">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {(triggeredRules && triggeredRules.length > 0) && (
        <div className="mt-4">
          <Label>Triggered Rules (Layer 1)</Label>
          <div className="flex flex-wrap gap-2">
            {triggeredRules.map(r => (
              <span key={r} className="px-2 py-0.5 rounded text-[11px] font-mono bg-layer1/15 text-layer1 border border-layer1/30">
                {r}
              </span>
            ))}
          </div>
        </div>
      )}

      {(anomalyDrivers && anomalyDrivers.length > 0) && (
        <div className="mt-4">
          <Label>Anomaly Drivers (Layer 4)</Label>
          <div className="flex flex-wrap gap-2">
            {anomalyDrivers.slice(0, 6).map(d => (
              <span key={d} className="px-2 py-0.5 rounded text-[11px] font-mono bg-layer4/15 text-layer4 border border-layer4/30">
                {d}
              </span>
            ))}
          </div>
        </div>
      )}

      {(temporalExplanations && temporalExplanations.length > 0) && (
        <div className="mt-4">
          <Label>Structural / Temporal (Layer 5)</Label>
          <ul className="text-xs text-textDim space-y-1 font-mono">
            {temporalExplanations.slice(0, 3).map((e, i) => (
              <li key={i}>· {e}</li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}
