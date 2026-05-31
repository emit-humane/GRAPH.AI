"use client";

/**
 * Circular risk gauge — 0..100, colour by level.
 *
 * Pure SVG, no charting deps. Renders an arc from -135° to +135° (a 270° sweep)
 * showing the fused transaction-risk score in the centre and the risk level
 * below. Designed to be the hero element of the fraud report.
 */
interface Props {
  score: number;             // 0..100
  level?: string;            // Low | Medium | High | Critical
  size?: number;             // px
  caption?: string;          // optional second line
}

const LEVEL_COLOR: Record<string, string> = {
  Low:      "#34d399",
  Medium:   "#d99a2b",
  High:     "#e23d6e",
  Critical: "#e23d6e",
};

export default function RiskGauge({ score, level, size = 168, caption }: Props) {
  const s = Math.max(0, Math.min(100, Number(score) || 0));
  const lvl = (level && LEVEL_COLOR[level]) ? level : "Low";
  const color = LEVEL_COLOR[lvl];

  // Geometry — semicircle arc from -135° to +135° (270° sweep)
  const cx = size / 2, cy = size / 2;
  const r = size / 2 - 16;
  const startA = -135 * (Math.PI / 180);
  const endA = (Math.PI / 180) * (-135 + 270 * (s / 100));
  const baseEndA = 135 * (Math.PI / 180);

  const pt = (a: number) => [cx + r * Math.cos(a), cy + r * Math.sin(a)] as const;
  const arc = (a0: number, a1: number) => {
    const [x0, y0] = pt(a0);
    const [x1, y1] = pt(a1);
    const sweep = a1 - a0;
    const large = Math.abs(sweep) > Math.PI ? 1 : 0;
    return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
  };

  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <svg width={size} height={size}>
        {/* Background arc */}
        <path
          d={arc(startA, baseEndA)}
          fill="none"
          stroke="#1f2940"
          strokeWidth={12}
          strokeLinecap="round"
        />
        {/* Score arc */}
        {s > 0 && (
          <path
            d={arc(startA, endA)}
            fill="none"
            stroke={color}
            strokeWidth={12}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${color}88)` }}
          />
        )}
        {/* Centre score */}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          fill={color}
          fontFamily="'Space Grotesk', 'Inter', system-ui, sans-serif"
          fontSize={size * 0.34}
          fontWeight={800}
        >
          {s.toFixed(0)}
        </text>
        <text
          x={cx}
          y={cy + size * 0.18}
          textAnchor="middle"
          fill="#8a96b0"
          fontFamily="'JetBrains Mono', monospace"
          fontSize={11}
          letterSpacing={2}
        >
          {(level || "LOW").toUpperCase()}
        </text>
      </svg>
      {caption && (
        <div className="text-[11px] tracking-[0.2em] font-mono text-textDim mt-1">
          {caption}
        </div>
      )}
    </div>
  );
}
