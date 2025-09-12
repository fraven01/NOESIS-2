import { test, expect } from '@playwright/test';
import path from 'path';

const file = path.join(__dirname, 'pages', 'list-sort.html');

/** List Sort */
test('sorts items when header clicked', async ({ page }) => {
  await page.goto('file://' + file);
  const header = page.getByRole('columnheader', { name: 'Name' });
  const firstRow = () => page.locator('tbody tr').first();
  await expect(firstRow()).toHaveText('Beta');
  await header.click();
  await expect(firstRow()).toHaveText('Alpha');
  await header.click();
  await expect(firstRow()).toHaveText('Gamma');
});
