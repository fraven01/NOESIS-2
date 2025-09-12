import { test, expect } from '@playwright/test';
import path from 'path';

const file = path.join(__dirname, 'pages', 'theme-toggle.html');

/** Theme Toggle Persistenz */
test('theme choice persists after reload', async ({ page }) => {
  await page.goto('file://' + file);
  const button = page.getByRole('button', { name: 'Toggle theme' });
  await button.click();
  await expect(page.locator('html')).toHaveClass(/dark/);
  await page.reload();
  await expect(page.locator('html')).toHaveClass(/dark/);
});
