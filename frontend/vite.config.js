import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/advice': 'http://localhost:5050',
      '/evaluate': 'http://localhost:5050',
    },
  },
})
