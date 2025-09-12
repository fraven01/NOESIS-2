import { test, expect } from '@playwright/test';
import path from 'path';

const file = path.join(__dirname, 'pages', 'dialog-confirm.html');

/** Dialog Confirm Flow */
test('dialog confirms after typing text', async ({ page }) => {
  await page.goto('file://' + file);
  await page.getByRole('button', { name: 'Delete' }).click();
  const dialog = page.getByRole('dialog', { name: 'Delete item?' });
  await dialog.getByLabel('Type to confirm').fill('DELETE');
  await dialog.getByRole('button', { name: 'Delete' }).click();
  await expect(page.getByText('confirmed')).toBeVisible();
});
