import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Only /api is proxied. Report and prediction artifacts are mounted under /api by
// the backend, so proxying bare /reports or /predictions would shadow the SPA
// client routes of the same name and break direct navigation to those pages.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})