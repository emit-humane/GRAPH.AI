"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "./clsx";

const NAV = [
  { href: "/overview", label: "Overview" },
  { href: "/case", label: "Case Analysis" },
  { href: "/geo", label: "Geographical View" },
  { href: "/report", label: "Fraud Report" },
  { href: "/graph", label: "Graph Explorer" },
  { href: "/analytics", label: "Analytics" },
];

export default function Sidebar() {
  const pathname = usePathname() || "";
  return (
    <aside className="w-60 border-r border-border p-6 sticky top-0 h-screen flex flex-col bg-bg">
      <div className="flex gap-2.5 items-center mb-8">
        <div className="w-10 h-10 rounded-[10px] bg-gradient-to-br from-[#1e3a5f] to-[#2bb673] grid place-items-center font-bold font-display text-sm">
          GA
        </div>
        <div>
          <div className="text-[10px] tracking-[0.2em] font-mono text-textDim">G.R.A.P.H. AI</div>
          <div className="text-[13px] font-bold leading-tight">Graph-based Risk Analysis &amp; Pattern Hunting</div>
        </div>
      </div>
      <nav className="flex flex-col gap-2">
        {NAV.map(n => {
          const active = pathname.startsWith(n.href);
          return (
            <Link
              key={n.href}
              href={n.href}
              className={clsx(
                "text-left px-4 py-3 rounded-[10px] text-sm font-semibold font-display transition",
                active
                  ? "bg-teal/10 text-text border-l-2 border-teal"
                  : "text-textDim border-l-2 border-transparent hover:text-text",
              )}
            >
              {n.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto pt-6 text-[10px] font-mono text-textFaint tracking-wide">
        v2 · 5-layer detection
      </div>
    </aside>
  );
}
