import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/schema": "http://127.0.0.1:8000",
      "/metrics": "http://127.0.0.1:8000",
      "/patients": "http://127.0.0.1:8000",
      "/predict": "http://127.0.0.1:8000",
      "/explain": "http://127.0.0.1:8000",
      "/plots": "http://127.0.0.1:8000",
    },
  },
});
