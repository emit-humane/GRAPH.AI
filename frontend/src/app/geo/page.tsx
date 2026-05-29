"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useSelectionStore } from "@/stores/selectionStore";
import { Label, Panel, PillNote, SectionTitle, Stat, ViewHeader } from "@/components/ui";

const IndiaMap = dynamic(() => import("@/components/IndiaMap"), { ssr: false });

export default function GeoPage() {
  const txId = useSelectionStore(s => s.selectedTransactionId);
  const payload = useSelectionStore(s => s.selectedPayload);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!txId) return;
    setErr(null);
    api.caseGeo(txId).then(setData).catch(e => setErr(e.message));
  }, [txId]);

  if (!txId) {
    return (
      <div className="max-w-4xl mx-auto">
        <ViewHeader title="Geographical View" sub="Select a transaction first." />
        <Panel className="bg-panelHi text-textDim">
          No case selected. Go to <a href="/overview" className="text-teal underline">Overview</a> first.
        </Panel>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      <ViewHeader
        title="Geographical View"
        sub="The accounts and edges involved in the selected case, projected onto an India base map."
      />

      <Panel className="mb-4">
        <div className="flex justify-between items-center mb-3">
          <div>
            <Label>Geographical View</Label>
            <SectionTitle>Fraud Geography Map</SectionTitle>
          </div>
          <PillNote>Focal nodes pulse in pink, network in green</PillNote>
        </div>
        {err && <div className="text-high text-sm font-mono mb-2">{err}</div>}
        {data ? (
          <>
            <IndiaMap nodes={data.nodes} edges={data.edges} height={460} />
            <div className="grid grid-cols-3 gap-3.5 mt-4">
              <Panel className="p-4"><Stat label="Active Accounts" value={data.metadata.active_accounts.toLocaleString()} /></Panel>
              <Panel className="p-4"><Stat label="Fraud Accounts" value={data.metadata.fraud_accounts} valueColor="#e23d6e" /></Panel>
              <Panel className="p-4"><Stat label="Fraud Edges" value={data.metadata.fraud_edges} valueColor="#d99a2b" /></Panel>
            </div>
          </>
        ) : (
          <div className="text-textDim text-sm">Loading geography …</div>
        )}
      </Panel>

      <Panel>
        <Label>Highlighted Geography</Label>
        <SectionTitle>Network Locations</SectionTitle>
        {(data?.nodes || []).filter((n: any) => n.is_focal).map((n: any) => (
          <div key={n.account_id} className="border border-border rounded-2xl p-4 mb-2.5 bg-panelHi">
            <div className="font-bold font-display mb-1">{n.account_id.slice(0, 16)} · {n.city}</div>
            <div className="font-mono text-xs text-textDim">lat={n.lat.toFixed(2)} lng={n.lng.toFixed(2)}</div>
          </div>
        ))}
      </Panel>
    </div>
  );
}
