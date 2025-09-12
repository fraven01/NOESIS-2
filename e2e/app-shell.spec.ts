import { test, expect } from '@playwright/test';
import path from 'path';

const file = path.join(__dirname, 'pages', 'app-shell.html');

/** AppShell Render */
test('app shell renders navigation and content', async ({ page }) => {
  await page.goto('file://' + file);
  await expect(page.getByRole('navigation', { name: 'Haupt' })).toBeVisible();
  await expect(page.getByRole('main')).toHaveText('Dashboard');
});
