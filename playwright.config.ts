import { defineConfig } from '@playwright/experimental-ct-react';
import react from '@vitejs/plugin-react';

export default defineConfig({
  testDir: './e2e',
  ctViteConfig: {
    plugins: [react()],
  },
  use: {
    headless: true,
    trace: 'retain-on-failure',
  },
});
