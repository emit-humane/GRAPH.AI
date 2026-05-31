/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Proxy /api/* to the backend. `NEXT_PUBLIC_BACKEND_URL` is set per
  // environment:
  //   - local dev:        http://localhost:8000
  //   - Vercel preview:   https://graphai-backend-staging.onrender.com
  //   - Vercel prod:      https://graphai-backend.onrender.com
  //
  // CRITICAL: Vercel buffers HTTP responses by default. The proxied
  // /stream/events route is Server-Sent Events and MUST stream — that's
  // why we set `Cache-Control: no-store` on /api/* in vercel.json AND
  // emit `X-Accel-Buffering: no` here. Without this, the live stream
  // appears frozen on the deployed dashboard.
  async rewrites() {
    const backend = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${backend}/:path*` },
    ];
  },

  async headers() {
    return [
      {
        source: "/api/stream/events",
        headers: [
          { key: "X-Accel-Buffering", value: "no" },
          { key: "Cache-Control", value: "no-store, no-transform" },
          { key: "Connection", value: "keep-alive" },
        ],
      },
    ];
  },
};
module.exports = nextConfig;
