import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  retries: 1,
  use: {
    headless: true,
    trace: 'on-first-retry',
  },
});
