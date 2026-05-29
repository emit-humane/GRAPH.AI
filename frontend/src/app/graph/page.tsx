"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Btn, Label, Panel, SectionTitle, Stat, ViewHeader } from "@/components/ui";
import CytoscapeChain from "@/components/CytoscapeChain";

export default function GraphExplorerPage() {
  const [nodeId, setNodeId] = useState("");
  const [hops, setHops] = useState(2);
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setErr(null);
    try {
      const r = await api.subgraph(nodeId, hops);
      setData(r);
    } catch (e: any) {
      setErr(e.message);
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      <ViewHeader
        title="Graph Explorer"
        sub="Run a k-hop neighbourhood query against the live D2 multigraph. Useful for ad-hoc account investigation outside the alert flow."
      />

      <Panel className="mb-4">
        <Label>Query</Label>
        <div className="grid grid-cols-[1fr_120px_120px] gap-3 items-end mt-2">
          <div>
            <div className="text-xs text-textDim mb-1.5 font-mono">Account ID</div>
            <input
              value={nodeId} onChange={e => setNodeId(e.target.value)}
              placeholder="paste a full account_id from the Overview"
              className="w-full bg-bg border border-border rounded-lg p-2.5 font-mono text-sm outline-none"
            />
          </div>
          <div>
            <div className="text-xs text-textDim mb-1.5 font-mono">Hops</div>
            <select value={hops} onChange={e => setHops(+e.target.value)}
              className="w-full bg-bg border border-border rounded-lg p-2.5 font-mono text-sm outline-none">
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </div>
          <Btn variant="primary" onClick={load} disabled={!nodeId}>Fetch</Btn>
        </div>
        {err && <div className="text-high text-sm font-mono mt-3">{err}</div>}
      </Panel>

      {data && (
        <Panel>
          <div className="flex justify-between items-center mb-3">
            <SectionTitle>{data.node_id.slice(0, 18)} · {data.hops}-hop neighbourhood</SectionTitle>
            <div className="flex gap-2 text-xs font-mono text-textDim">
              <span>{data.elements.nodes.length} nodes</span>
              <span>·</span>
              <span>{data.elements.edges.length} edges</span>
            </div>
          </div>
          <CytoscapeChain elements={data.elements} layout="cose" height={500} />
        </Panel>
      )}
    </div>
  );
}
