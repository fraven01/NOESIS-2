import React from 'react';
import { test, expect } from '@playwright/experimental-ct-react';
import { ThemeSwitch } from '../theme/components/ThemeSwitch';

test('theme toggle persists across reloads', async ({ mount, page }) => {
  await mount(<ThemeSwitch />);
  const button = page.getByRole('button');
  await button.click();
  await expect(page.locator('html')).toHaveClass(/dark/);
  await page.reload();
  await mount(<ThemeSwitch />);
  await expect(page.locator('html')).toHaveClass(/dark/);
  const stored = await page.evaluate(() => window.localStorage.getItem('theme'));
  expect(stored).toBe('dark');
});
