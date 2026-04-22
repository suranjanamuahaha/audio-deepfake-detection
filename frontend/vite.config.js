import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],

  extend: {
  keyframes: {
    wave: {
      "0%, 100%": { transform: "scaleY(0.4)" },
      "50%": { transform: "scaleY(1.5)" },
    },
  },
  animation: {
    wave: "wave 1s ease-in-out infinite",
  },
}
})
