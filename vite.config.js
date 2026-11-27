import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// The API port is read from the environment so `PORT=9000 npm start` does not
// silently break the dev proxy the way a hardcoded 8787 did.
const apiPort = process.env.PORT || '8787';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': `http://localhost:${apiPort}`,
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // three + react is a big bundle; split it so the viewer code is not
    // re-downloaded every time the UI changes.
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
});
