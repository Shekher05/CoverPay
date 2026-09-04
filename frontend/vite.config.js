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
// - connect-src 'self' assumes the API is same-origin. If you split the frontend
//   and backend onto different hosts (VITE_API_BASE), add that origin here.
// - frame-ancestors is ignored in a <meta> CSP; also set X-Frame-Options or a
//   real CSP header at the static host for clickjacking protection.
const PROD_CSP = [
  "default-src 'self'",
  "base-uri 'none'",
  "object-src 'none'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
  "upgrade-insecure-requests",
].join("; ");

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
            `  <meta http-equiv="Content-Security-Policy" content="${PROD_CSP}" />\n  </head>`,
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
