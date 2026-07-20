import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs on :8000. All /api calls are proxied in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
