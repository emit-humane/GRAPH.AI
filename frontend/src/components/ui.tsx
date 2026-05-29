"use client";

import { ReactNode } from "react";
import clsx from "./clsx";

// Tiny atomic UI primitives ported from docs/ui_reference.jsx, restyled
// with Tailwind. The reference uses inline-style theming; here we centralise
// the same palette in tailwind.config.js → keeps the per-component CSS
// surface tiny.

export function Label({ children }: { children: ReactNode }) {
  return (
    <div className="text-[10px] tracking-[0.25em] uppercase text-textDim font-mono mb-2">
      {children}
    </div>
  );
}

export function Panel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={clsx("bg-panel border border-border rounded-2xl p-5", className)}>
      {children}
    </div>
  );
}

export function Stat({ label, value, valueColor }: { label: ReactNode; value: ReactNode; valueColor?: string }) {
  return (
    <div>
      <Label>{label}</Label>
      <div className="text-3xl font-bold font-display leading-none" style={{ color: valueColor }}>
        {value}
      </div>
    </div>
  );
}

export type Severity = "Low" | "Medium" | "High" | "Critical";

export function Badge({ level }: { level: Severity | string }) {
  const map: Record<string, { c: string; bg: string }> = {
    High: { c: "#e23d6e", bg: "rgba(226,61,110,0.12)" },
    Critical: { c: "#e23d6e", bg: "rgba(226,61,110,0.20)" },
    Medium: { c: "#d99a2b", bg: "rgba(217,154,43,0.12)" },
    Low: { c: "#2bb673", bg: "rgba(43,182,115,0.12)" },
  };
  const s = map[level] || map.Low;
  return (
    <span
      style={{ color: s.c, background: s.bg, border: `1px solid ${s.c}55` }}
      className="px-3.5 py-1 rounded-full text-xs font-semibold font-display tracking-wide"
    >
      {level}
    </span>
  );
}

export function Btn({
  children, onClick, variant = "ghost", disabled = false,
}: {
  children: ReactNode;
  onClick?: (e?: any) => void;
  variant?: "ghost" | "primary" | "accent" | "danger";
  disabled?: boolean;
}) {
  const styles: Record<string, string> = {
    ghost: "bg-transparent text-text border-border hover:brightness-125",
    primary: "bg-high text-white border-high hover:brightness-110",
    accent: "bg-accent/15 text-violet-300 border-accent/40 hover:brightness-125",
    danger: "bg-red-600 text-white border-red-600 hover:brightness-110",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "border rounded-lg px-4 py-2 text-sm font-semibold font-display transition",
        styles[variant],
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      {children}
    </button>
  );
}

export function PillNote({ children }: { children: ReactNode }) {
  return (
    <span className="px-3.5 py-2 rounded-lg text-xs font-semibold font-display bg-teal/10 text-teal border border-teal/30 max-w-[220px] leading-tight">
      {children}
    </span>
  );
}

export function ViewHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="mb-6">
      <div className="text-[11px] tracking-[0.3em] font-mono text-textDim mb-2">G.R.A.P.H. AI</div>
      <h1 className="text-3xl font-bold font-display m-0">{title}</h1>
      <p className="text-textDim text-sm mt-2 max-w-2xl leading-relaxed">{sub}</p>
    </div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return <div className="text-lg font-bold font-display mb-3">{children}</div>;
}

export function ConnectionDot({ connected }: { connected: boolean }) {
  return (
    <span
      className={clsx(
        "w-2.5 h-2.5 rounded-full",
        connected ? "bg-teal shadow-[0_0_10px_#34d399]" : "bg-textFaint",
      )}
    />
  );
}
