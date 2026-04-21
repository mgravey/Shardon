import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const adminUiPort = Number(process.env.SHARDON_ADMIN_UI_PORT ?? "5173");

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: adminUiPort,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8081",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
