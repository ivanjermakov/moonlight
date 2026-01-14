import { defineConfig } from 'vite'

export default defineConfig({
    plugins: [],
    server: {
        port: 3000,
        hmr: false,
        watch: undefined
    },
    build: {
        target: 'esnext'
    }
})
