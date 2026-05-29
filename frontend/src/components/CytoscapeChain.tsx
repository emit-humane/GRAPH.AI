"use client";

import { useEffect, useRef } from "react";
import cytoscape, { ElementsDefinition } from "cytoscape";

interface Props {
  elements: ElementsDefinition;
  focal?: { sender: string; receiver: string };
  layout?: "dagre" | "cose" | "circle" | "grid";
  height?: number;
}

export default function CytoscapeChain({ elements, focal, layout = "cose", height = 360 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: { name: layout === "dagre" ? "breadthfirst" : layout, padding: 24, animate: false } as any,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#34d399",
            "label": "data(label)",
            "color": "#e6ebf5",
            "text-valign": "bottom",
            "text-margin-y": 6,
            "font-family": "JetBrains Mono",
            "font-size": 10,
            "width": 18,
            "height": 18,
          },
        },
        {
          selector: 'node[?is_focal]',
          style: {
            "background-color": "#e23d6e",
            "border-color": "#fff",
            "border-width": 2,
            "width": 24,
            "height": 24,
          },
        },
        {
          selector: 'node[?is_sender]',
          style: { "background-color": "#fbbf24", "border-color": "#fff", "border-width": 2 },
        },
        {
          selector: 'node[?is_receiver]',
          style: { "background-color": "#22d3ee", "border-color": "#fff", "border-width": 2 },
        },
        {
          selector: "edge",
          style: {
            "line-color": "#1f2940",
            "width": 1.5,
            "target-arrow-color": "#1f2940",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
        {
          selector: 'edge[?is_focal]',
          style: { "line-color": "#e23d6e", "target-arrow-color": "#e23d6e", "width": 2.5 },
        },
      ],
    });
    cyRef.current = cy;
    cy.fit(undefined, 30);
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [elements, layout]);

  return (
    <div
      ref={containerRef}
      style={{ height, width: "100%", background: "#0d1320", borderRadius: 12, border: "1px solid #1f2940" }}
    />
  );
}
