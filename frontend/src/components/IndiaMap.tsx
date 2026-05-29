"use client";

// @ts-ignore — CSS side-effect import, no types needed
import "leaflet/dist/leaflet.css";
import { useEffect, useRef } from "react";

interface GeoNode { account_id: string; city: string; lat: number; lng: number; is_focal: boolean; }
interface GeoEdge { source: string; target: string; amount?: number; is_focal?: boolean; }

interface Props {
  nodes: GeoNode[];
  edges: GeoEdge[];
  height?: number;
}

// Lazy-load Leaflet on the client only (react-leaflet doesn't SSR).
export default function IndiaMap({ nodes, edges, height = 460 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const layersRef = useRef<any[]>([]);

  useEffect(() => {
    if (!ref.current) return;

    let cancelled = false;
    (async () => {
      const L = (await import("leaflet")).default;
      if (cancelled) return;

      if (!mapRef.current) {
        const m = L.map(ref.current!, {
          center: [22.5, 80],
          zoom: 4,
          minZoom: 3,
          maxZoom: 8,
          attributionControl: false,
        });
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", {
          attribution: "&copy; OSM &copy; CARTO",
        }).addTo(m);
        mapRef.current = m;
      }
      const map = mapRef.current;

      // Clear previous layers
      layersRef.current.forEach(l => map.removeLayer(l));
      layersRef.current = [];

      // Build node lookup
      const byId: Record<string, GeoNode> = {};
      nodes.forEach(n => { byId[n.account_id] = n; });

      // Edges
      edges.forEach(e => {
        const a = byId[e.source], b = byId[e.target];
        if (!a || !b) return;
        const line = L.polyline([[a.lat, a.lng], [b.lat, b.lng]], {
          color: e.is_focal ? "#e23d6e" : "#d99a2b",
          weight: e.is_focal ? 3 : 1.5,
          opacity: 0.85,
          dashArray: e.is_focal ? undefined : "5 4",
        }).addTo(map);
        layersRef.current.push(line);
      });

      // Nodes
      nodes.forEach(n => {
        const marker = L.circleMarker([n.lat, n.lng], {
          radius: n.is_focal ? 9 : 6,
          color: n.is_focal ? "#e23d6e" : "#34d399",
          fillColor: n.is_focal ? "#e23d6e" : "#34d399",
          fillOpacity: 0.8,
          weight: n.is_focal ? 3 : 1,
        }).bindTooltip(
          `<div style="font-family:JetBrains Mono,monospace;font-size:11px;">
             <div style="font-weight:600;color:${n.is_focal ? "#e23d6e" : "#34d399"};">
               ${n.account_id.slice(0, 12)}
             </div>
             <div style="color:#8a96b0;">${n.city}</div>
           </div>`,
          { permanent: false, direction: "right" } as any,
        ).addTo(map);
        layersRef.current.push(marker);
      });

      // Fit bounds to all nodes
      if (nodes.length > 0) {
        const bounds = L.latLngBounds(nodes.map(n => [n.lat, n.lng] as [number, number]));
        map.fitBounds(bounds.pad(0.25));
      }
    })();

    return () => { cancelled = true; };
  }, [nodes, edges]);

  return (
    <div
      ref={ref}
      style={{ height, width: "100%", borderRadius: 12, border: "1px solid #1f2940", overflow: "hidden" }}
    />
  );
}
