/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0e17",
        panel: "#111725",
        panelHi: "#161d2e",
        border: "#1f2940",
        text: "#e6ebf5",
        textDim: "#8a96b0",
        textFaint: "#5a6680",
        accent: "#7c3aed",
        high: "#e23d6e",
        medium: "#d99a2b",
        low: "#2bb673",
        teal: "#34d399",
        cyan: "#22d3ee",
        // 5 layer palette
        layer1: "#22d3ee",   // rule
        layer2: "#a78bfa",   // graph
        layer3: "#34d399",   // supervised — PRIMARY
        layer4: "#fbbf24",   // anomaly
        layer5: "#f472b6",   // tgn
      },
      fontFamily: {
        display: ["Space Grotesk", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
