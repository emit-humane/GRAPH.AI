"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { Btn, Label, Panel, PillNote, SectionTitle, Stat, ViewHeader } from "@/components/ui";
import clsx from "@/components/clsx";

// ----------------------------------------------------------------------------
// Local types — match the backend catalog payload exactly
// ----------------------------------------------------------------------------
type ParamSpec = {
  label: string; min: number; max: number; default: number; step: number; integer: boolean;
};
type TypologySpec = {
  key: string; label: string; pattern: string; blurb: string;
  target_rules: string[]; color: string; params: Record<string, ParamSpec>;
};
type PlanResp = {
  typology: string;
  params: Record<string, any>;
  plans: Array<{
    ring_id: string;
    typology: string;
    target_rules: string[];
    members: string[];
    edges: Array<{ sender: string; receiver: string; amount: number; timestamp_offset_s: number; sender_country: string; receiver_country: string; is_international: boolean; note?: string; }>;
    summary: { edge_count: number; account_count: number; total_value: number; };
    metadata: Record<string, any>;
  }>;
};
type ActivityResp = {
  items: Array<{ timestamp: string; msg: string; kind: string }>;
  queue_depth: number;
  export_buffer: number;
};


export default function ScenarioStudioPage() {
  const [catalog, setCatalog] = useState<TypologySpec[] | null>(null);
  const [active, setActive] = useState<string>("layering_chain");
  const [params, setParams] = useState<Record<string, number>>({});
  const [seed, setSeed] = useState(42);
  const [count, setCount] = useState(1);
  const [mode, setMode] = useState<"live" | "export">("live");
  const [preview, setPreview] = useState<PlanResp | null>(null);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [activity, setActivity] = useState<ActivityResp | null>(null);
  const [busy, setBusy] = useState(false);

  // Load catalog once + tail activity log
  useEffect(() => {
    api.studioCatalog().then(c => {
      const typologies = c.typologies as TypologySpec[];
      setCatalog(typologies);
      const layering = typologies.find(t => t.key === "layering_chain") ?? typologies[0];
      setActive(layering.key);
      setParams(Object.fromEntries(Object.entries(layering.params).map(([k, p]) => [k, (p as ParamSpec).default])));
    }).catch(e => setPreviewErr(`catalog: ${e.message}`));
  }, []);

  useEffect(() => {
    let cancel = false;
    const tick = () => api.studioActivity().then(a => !cancel && setActivity(a)).catch(() => {});
    tick();
    const id = setInterval(tick, 2500);
    return () => { cancel = true; clearInterval(id); };
  }, []);

  // Recompute preview whenever params/typology/seed change.
  useEffect(() => {
    if (!catalog || !active || Object.keys(params).length === 0) return;
    let cancel = false;
    api.studioPreview(active, params, seed, 1)
      .then(p => { if (!cancel) { setPreview(p); setPreviewErr(null); } })
      .catch(e => !cancel && setPreviewErr(e.message));
    return () => { cancel = true; };
  }, [catalog, active, params, seed]);

  const activeSpec = useMemo<TypologySpec | null>(
    () => (catalog?.find(t => t.key === active) ?? null),
    [catalog, active],
  );

  const selectTypology = (key: string) => {
    setActive(key);
    const t = catalog?.find(x => x.key === key);
    if (t) setParams(Object.fromEntries(Object.entries(t.params).map(([k, p]) => [k, p.default])));
  };

  const fire = async () => {
    if (!activeSpec) return;
    setBusy(true);
    try {
      if (mode === "live") {
        await api.studioInject(active, params, seed, count);
      } else {
        await api.studioExport([{ typology: active, params, seed, count }]);
      }
      const a = await api.studioActivity();
      setActivity(a);
    } catch (e: any) {
      console.warn(e);
    } finally {
      setBusy(false);
    }
  };

  const downloadExport = () => {
    const url = api.studioExportDownloadUrl();
    window.open(url, "_blank");
  };

  const clearExport = async () => {
    await api.studioClearExport();
    const a = await api.studioActivity();
    setActivity(a);
  };

  if (!catalog) {
    return (
      <div className="max-w-7xl mx-auto">
        <ViewHeader title="Scenario Studio" sub="Loading catalog …" />
        {previewErr && <Panel className="bg-high/10 border-high/40 text-high text-sm font-mono">{previewErr}</Panel>}
      </div>
    );
  }

  const plan0 = preview?.plans?.[0];

  return (
    <div className="max-w-7xl mx-auto">
      <ViewHeader
        title="Scenario Studio"
        sub="Pick a laundering typology, tune its parameters, and either inject it into the live stream or append it to an exportable labelled dataset. Same engine, both entry points."
      />

      <div className="grid grid-cols-[260px_1fr_320px] gap-4">
        {/* Column 1 — typology picker */}
        <Panel>
          <Label>Typology</Label>
          <div className="flex flex-col gap-1.5 mt-1">
            {catalog.map(t => {
              const isActive = active === t.key;
              return (
                <button key={t.key} onClick={() => selectTypology(t.key)}
                  className={clsx(
                    "text-left px-3 py-2.5 rounded-lg border transition",
                    isActive
                      ? "border-high/60 bg-high/10"
                      : "border-border hover:border-accent/40",
                  )}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold font-display">{t.label}</span>
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: t.color }} />
                  </div>
                  <div className="text-[10px] tracking-wider font-mono text-textDim mt-1">
                    {t.target_rules.join(" · ")}
                  </div>
                </button>
              );
            })}
          </div>
        </Panel>

        {/* Column 2 — params + preview */}
        <div className="flex flex-col gap-4">
          {activeSpec && (
            <Panel>
              <div className="flex justify-between items-start mb-2">
                <div>
                  <Label>Pattern</Label>
                  <SectionTitle>{activeSpec.label}</SectionTitle>
                </div>
                <div className="flex flex-wrap gap-1.5 max-w-xs justify-end">
                  {activeSpec.target_rules.map(r => (
                    <span key={r} className="px-2 py-0.5 text-[10px] font-mono bg-layer1/15 text-layer1 border border-layer1/30 rounded">
                      {r}
                    </span>
                  ))}
                </div>
              </div>
              <p className="text-textDim text-sm mb-2">{activeSpec.blurb}</p>
              <div className="text-xs font-mono text-textFaint mb-4">{activeSpec.pattern}</div>

              <div className="grid grid-cols-2 gap-x-6 gap-y-1">
                {Object.entries(activeSpec.params).map(([key, ps]) => {
                  const val = Number(params[key] ?? ps.default);
                  return (
                    <div key={key} className="mb-3">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-display">{ps.label}</span>
                        <span className="text-sm font-mono text-teal">
                          {ps.integer ? Math.round(val).toLocaleString("en-IN") : val.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range" min={ps.min} max={ps.max} step={ps.step} value={val}
                        onChange={e => setParams(p => ({ ...p, [key]: +e.target.value }))}
                        className="w-full accent-accent"
                      />
                    </div>
                  );
                })}
              </div>
            </Panel>
          )}

          <Panel>
            <div className="flex justify-between items-center mb-3">
              <div>
                <Label>Live Preview</Label>
                <SectionTitle>Planned Edges</SectionTitle>
              </div>
              <PillNote>seed = {seed}</PillNote>
            </div>
            {previewErr && (
              <div className="text-high text-sm font-mono mb-2">{previewErr}</div>
            )}
            {plan0 && (
              <>
                <div className="grid grid-cols-3 gap-3 mb-4">
                  <Panel className="p-3"><Stat label="Edges" value={plan0.summary.edge_count} /></Panel>
                  <Panel className="p-3"><Stat label="Accounts" value={plan0.summary.account_count} /></Panel>
                  <Panel className="p-3"><Stat label="Total Value" value={`₹${plan0.summary.total_value.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`} /></Panel>
                </div>
                <div className="max-h-72 overflow-auto scrollbar-thin border border-border rounded-lg bg-bg/50">
                  <table className="w-full text-xs font-mono">
                    <thead className="bg-panelHi sticky top-0">
                      <tr className="text-textDim">
                        <th className="text-left px-3 py-2">#</th>
                        <th className="text-left px-3 py-2">sender → receiver</th>
                        <th className="text-right px-3 py-2">amount</th>
                        <th className="text-right px-3 py-2">+t (s)</th>
                        <th className="text-left px-3 py-2">geo</th>
                      </tr>
                    </thead>
                    <tbody>
                      {plan0.edges.map((e, i) => (
                        <tr key={i} className="border-t border-border">
                          <td className="px-3 py-1.5 text-textFaint">{i + 1}</td>
                          <td className="px-3 py-1.5">
                            <span className="text-text">{e.sender.slice(0, 12)}</span>
                            <span className="text-textDim"> → </span>
                            <span className="text-text">{e.receiver.slice(0, 12)}</span>
                          </td>
                          <td className="px-3 py-1.5 text-right text-teal">
                            ₹{Math.round(e.amount).toLocaleString("en-IN")}
                          </td>
                          <td className="px-3 py-1.5 text-right text-textDim">
                            {Math.round(e.timestamp_offset_s).toLocaleString()}
                          </td>
                          <td className="px-3 py-1.5">
                            {e.sender_country} → {e.receiver_country}
                            {e.is_international && <span className="ml-1 text-medium">×INTL</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </Panel>
        </div>

        {/* Column 3 — action panel */}
        <div className="flex flex-col gap-4">
          <Panel>
            <Label>Mode</Label>
            <div className="flex gap-2 mb-4 mt-1">
              <button onClick={() => setMode("live")}
                className={clsx(
                  "flex-1 px-3 py-2 rounded-lg text-sm font-semibold font-display border transition",
                  mode === "live"
                    ? "border-teal/50 bg-teal/15 text-teal"
                    : "border-border text-textDim hover:text-text",
                )}>
                Live Inject
              </button>
              <button onClick={() => setMode("export")}
                className={clsx(
                  "flex-1 px-3 py-2 rounded-lg text-sm font-semibold font-display border transition",
                  mode === "export"
                    ? "border-accent/50 bg-accent/15 text-violet-300"
                    : "border-border text-textDim hover:text-text",
                )}>
                Export CSV
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <div className="text-xs text-textDim mb-1 font-mono">Instances</div>
                <input type="number" min={1} max={50} value={count} onChange={e => setCount(Math.max(1, +e.target.value))}
                  className="w-full bg-bg border border-border rounded-lg p-2 font-mono text-sm outline-none" />
              </div>
              <div>
                <div className="text-xs text-textDim mb-1 font-mono">Seed</div>
                <input type="number" value={seed} onChange={e => setSeed(+e.target.value)}
                  className="w-full bg-bg border border-border rounded-lg p-2 font-mono text-sm outline-none" />
              </div>
            </div>

            <Btn variant={mode === "live" ? "primary" : "accent"} onClick={fire} disabled={busy}>
              {busy ? "Working …" : (mode === "live" ? `Inject ${count}× into live stream` : `Buffer ${count}× for export`)}
            </Btn>

            {mode === "export" && (
              <div className="grid grid-cols-2 gap-2 mt-3">
                <Btn onClick={downloadExport} disabled={!(activity?.export_buffer ?? 0)}>
                  Download CSV
                </Btn>
                <Btn onClick={clearExport} disabled={!(activity?.export_buffer ?? 0)}>
                  Clear buffer
                </Btn>
              </div>
            )}

            <div className="mt-4 text-xs font-mono text-textDim border-t border-border pt-3">
              <div className="flex justify-between">
                <span>queue depth</span>
                <span>{activity?.queue_depth ?? 0}</span>
              </div>
              <div className="flex justify-between">
                <span>export buffer</span>
                <span>{activity?.export_buffer ?? 0}</span>
              </div>
            </div>
          </Panel>

          <Panel>
            <Label>Activity Log</Label>
            <div className="space-y-2 max-h-[40rem] overflow-auto scrollbar-thin">
              {(activity?.items ?? []).length === 0 && (
                <div className="text-textDim text-sm">Nothing yet. Fire a scenario to populate.</div>
              )}
              {(activity?.items ?? []).map((it, i) => (
                <div key={i} className="border border-border rounded-lg p-2.5 bg-panelHi">
                  <div className="text-[11px] font-mono text-textDim flex justify-between">
                    <span>{it.timestamp.slice(11, 19)}</span>
                    <span className={clsx(
                      it.kind === "inject" && "text-teal",
                      it.kind === "export" && "text-violet-300",
                      it.kind === "download" && "text-cyan",
                      it.kind === "warn" && "text-medium",
                    )}>{it.kind}</span>
                  </div>
                  <div className="text-xs mt-1 leading-relaxed">{it.msg}</div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
