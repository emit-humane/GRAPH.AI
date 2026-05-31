"use client";

// @ts-ignore — CSS side-effect import, no types needed
import "leaflet/dist/leaflet.css";
import { useEffect, useRef } from "react";

interface GeoNode {
  account_id: string;
  city: string;
  lat: number;
  lng: number;
  is_focal: boolean;
  /** "focal" | "neighbour" | "context" — drives the marker tier */
  kind?: "focal" | "neighbour" | "context";
  country?: string | null;
}
interface GeoEdge {
  source: string;
  target: string;
  amount?: number;
  is_focal?: boolean;
  /** "focal" | "context" — drives the line/arrow tier */
  kind?: "focal" | "context";
  risk_level?: string;
  alerted?: boolean;
}

interface Props {
  nodes: GeoNode[];
  edges: GeoEdge[];
  height?: number;
}

const COLOR = {
  focal: "#e23d6e",       // hot pink — the case under investigation
  neighbour: "#34d399",   // mint — 2-hop network around the case
  context_alerted: "#fbbf24", // amber — other transactions that ALSO alerted
  context: "#3b82f6",     // soft blue — other transactions (background traffic)
};

// Bearing (deg, 0 = north, clockwise) from (lat1,lng1) to (lat2,lng2).
function bearing(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const Δλ = ((lng2 - lng1) * Math.PI) / 180;
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
  const θ = Math.atan2(y, x);
  return (θ * 180) / Math.PI; // -180..180
}

function midpoint(lat1: number, lng1: number, lat2: number, lng2: number, t = 0.62) {
  // Point at parameter t along the great-circle. For the small distances we
  // care about, linear interpolation is visually indistinguishable and far
  // cheaper than spherical interp.
  return [lat1 + (lat2 - lat1) * t, lng1 + (lng2 - lng1) * t] as [number, number];
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
          zoom: 3,
          minZoom: 2,
          maxZoom: 8,
          worldCopyJump: true,
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

      // ---- Edges (three tiers) -------------------------------------------- //
      // Draw context edges FIRST so the focal edge sits on top.
      const sortedEdges = [...edges].sort((a, b) => {
        const rank = (e: GeoEdge) => (e.is_focal || e.kind === "focal" ? 2 : e.alerted ? 1 : 0);
        return rank(a) - rank(b);
      });

      sortedEdges.forEach(e => {
        const a = byId[e.source], b = byId[e.target];
        if (!a || !b) return;
        // Skip degenerate zero-length edges (sender == receiver projected to same point)
        if (a.lat === b.lat && a.lng === b.lng) return;

        const isFocal = e.is_focal || e.kind === "focal";
        const isAlertedCtx = !isFocal && e.alerted;
        const color = isFocal ? COLOR.focal : isAlertedCtx ? COLOR.context_alerted : COLOR.context;
        const weight = isFocal ? 3.2 : isAlertedCtx ? 2 : 1.3;
        const opacity = isFocal ? 0.95 : isAlertedCtx ? 0.85 : 0.55;
        const dash = isFocal ? undefined : isAlertedCtx ? "6 3" : "4 5";

        const line = L.polyline([[a.lat, a.lng], [b.lat, b.lng]], {
          color, weight, opacity,
          dashArray: dash,
        }).addTo(map);
        layersRef.current.push(line);

        // ---- Direction arrow at ~62% along the line --------------------- //
        const [mlat, mlng] = midpoint(a.lat, a.lng, b.lat, b.lng, 0.62);
        const θ = bearing(a.lat, a.lng, b.lat, b.lng);
        // Convert "0deg = north, clockwise" to CSS "0deg = right, clockwise"
        const cssRot = θ - 90;
        const arrowSize = isFocal ? 14 : isAlertedCtx ? 12 : 10;
        const arrowIcon = L.divIcon({
          className: "geo-arrow",
          html: `<div style="
              width:${arrowSize}px;height:${arrowSize}px;
              transform: rotate(${cssRot}deg);
              transform-origin: 50% 50%;
              color:${color};
              line-height:${arrowSize}px;
              text-align:center;
              font-size:${arrowSize}px;
              filter: drop-shadow(0 0 2px rgba(0,0,0,0.7));
            ">&#9654;</div>`,
          iconSize: [arrowSize, arrowSize],
          iconAnchor: [arrowSize / 2, arrowSize / 2],
        });
        const arrow = L.marker([mlat, mlng], { icon: arrowIcon, interactive: false }).addTo(map);
        layersRef.current.push(arrow);
      });

      // ---- Nodes (three tiers) -------------------------------------------- //
      // Draw context nodes first, then neighbours, then focals on top.
      const nodeRank = (n: GeoNode) =>
        (n.is_focal || n.kind === "focal") ? 2 : n.kind === "neighbour" ? 1 : 0;
      const sortedNodes = [...nodes].sort((a, b) => nodeRank(a) - nodeRank(b));

      sortedNodes.forEach(n => {
        const isFocal = n.is_focal || n.kind === "focal";
        const isContext = n.kind === "context";
        const color = isFocal ? COLOR.focal : isContext ? COLOR.context : COLOR.neighbour;
        const radius = isFocal ? 9 : isContext ? 4 : 6;
        const fillOpacity = isFocal ? 0.85 : isContext ? 0.55 : 0.8;
        const weight = isFocal ? 3 : isContext ? 0.8 : 1;

        const marker = L.circleMarker([n.lat, n.lng], {
          radius, color, fillColor: color, fillOpacity, weight,
        }).bindTooltip(
          `<div style="font-family:JetBrains Mono,monospace;font-size:11px;">
             <div style="font-weight:600;color:${color};">
               ${n.account_id.slice(0, 12)}
             </div>
             <div style="color:#8a96b0;">${n.city}${n.country && n.country !== "IN" ? ` · ${n.country}` : ""}</div>
             <div style="color:#5e6b85;font-size:10px;margin-top:2px;">
               ${isFocal ? "focal account" : isContext ? "background traffic" : "2-hop neighbour"}
             </div>
           </div>`,
          { permanent: false, direction: "right" } as any,
        ).addTo(map);
        layersRef.current.push(marker);
      });

      // ---- Legend (rebuild every render) ---------------------------------- //
      const legendDiv = L.DomUtil.create("div");
      legendDiv.style.cssText = `
        background: rgba(13, 19, 32, 0.92);
        border: 1px solid #1f2940;
        border-radius: 8px;
        color: #e6ebf5;
        font-family: JetBrains Mono, monospace;
        font-size: 10px;
        padding: 8px 10px;
        line-height: 1.6;`;
      legendDiv.innerHTML = `
        <div style="font-weight:600;letter-spacing:0.05em;margin-bottom:4px;">LEGEND</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COLOR.focal};margin-right:6px;"></span>focal case</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COLOR.neighbour};margin-right:6px;"></span>2-hop neighbour</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COLOR.context_alerted};margin-right:6px;"></span>other alert</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COLOR.context};margin-right:6px;"></span>background tx</div>
        <div style="margin-top:4px;color:#8a96b0;">▶ direction of transfer</div>`;
      const Legend = L.Control.extend({
        options: { position: "bottomright" },
        onAdd: () => legendDiv,
      });
      const legend = new (Legend as any)();
      legend.addTo(map);
      layersRef.current.push({ removeFrom: (m: any) => m.removeControl(legend) });
      // The control isn't a layer but Leaflet's removeLayer is permissive about
      // non-layers (silently ignores). Add an explicit removeControl on the
      // next render — handled via a sentinel-style cleanup below.

      // Fit bounds to all nodes
      if (nodes.length > 0) {
        const bounds = L.latLngBounds(nodes.map(n => [n.lat, n.lng] as [number, number]));
        map.fitBounds(bounds.pad(0.2), { animate: false });
      }

      // Stash the legend control so the next render can remove it cleanly.
      (map as any).__amlLegend && map.removeControl((map as any).__amlLegend);
      (map as any).__amlLegend = legend;
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
