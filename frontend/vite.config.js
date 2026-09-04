import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Content-Security-Policy for the production build.
//
// The built app is fully self-hosted: one external module script, one stylesheet,
// woff2 fonts, and XHR to the API - all same-origin (see `BASE = "/api"` in
// src/api.js). So every directive is 'self', which blocks injected <script>,
// remote script/style, and framing.
//
// - script-src has no 'unsafe-inline': that is the point, and the prod build
//   emits no inline script. Dev does (React Fast Refresh), which is why this is
//   injected at build time only - see the plugin below.
// - style-src keeps 'unsafe-inline': low risk, and Vite / a runtime can still
//   emit a <style> tag or inline style attribute.
// - connect-src is 'self' plus the API origin when the backend is on a different
//   host (VITE_API_BASE, the Render Static Site + Web Service split). Derived
//   from the same env var api.js uses, so the two never drift.
// - frame-ancestors is ignored in a <meta> CSP; also set X-Frame-Options or a
//   real CSP header at the static host for clickjacking protection.
function buildProdCsp() {
  let apiOrigin = "";
  try {
    if (process.env.VITE_API_BASE) apiOrigin = new URL(process.env.VITE_API_BASE).origin;
  } catch {
    // not a full URL (e.g. "/api") - stays same-origin, nothing to add
  }
  return [
    "default-src 'self'",
    "base-uri 'none'",
    "object-src 'none'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    `connect-src 'self'${apiOrigin ? " " + apiOrigin : ""}`,
    "form-action 'self'",
    "frame-ancestors 'none'",
    "upgrade-insecure-requests",
  ].join("; ");
}

// /api is proxied to FastAPI so the browser sees one origin in development.
export default defineConfig({
  plugins: [
    react(),
    {
      name: "coverpay-prod-csp",
      transformIndexHtml: {
        order: "post",
        handler(html, ctx) {
          // ctx.server is set only under `vite`/`vite dev`; skip so Fast Refresh
          // (an inline script) still runs. Build and `vite preview` get the CSP.
          if (ctx.server) return html;
          return html.replace(
            "</head>",
            `  <meta http-equiv="Content-Security-Policy" content="${buildProdCsp()}" />\n  </head>`,
          );
        },
      },
    },
  ],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
