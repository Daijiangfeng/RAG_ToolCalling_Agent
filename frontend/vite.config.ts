import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs on :8000. All /api calls are proxied in dev.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // 拆分供应商 chunk，提升长期缓存与首屏加载。
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          antd: ["antd", "@ant-design/icons"],
          markdown: ["react-markdown"],
          vendor: ["zustand", "axios"],
        },
      },
    },
  },
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
