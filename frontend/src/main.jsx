import React from "react";
import { createRoot } from "react-dom/client";
// Self-hosted variable fonts. No CDN request at runtime, and the console
// keeps its typeface offline, which an internal ops tool needs.
import "@fontsource-variable/geist/wght.css";
import "@fontsource-variable/geist-mono/wght.css";
import App from "./App.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
