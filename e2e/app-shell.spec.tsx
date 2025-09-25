import React from 'react';
import { test, expect } from '@playwright/experimental-ct-react';
import { AppShell } from '../theme/components/layouts/AppShell';
import { checkA11y } from './axe';

test('AppShell renders navigation and content', async ({ mount, page }) => {
  await mount(
    <AppShell breadcrumbs={<span>Home</span>} actions={<button>Act</button>} buildInfo="1.0">
      <p>Hallo Welt</p>
    </AppShell>
  );
  await expect(page.getByRole('navigation', { name: 'Haupt' })).toBeVisible();
  await expect(page.getByText('Hallo Welt')).toBeVisible();
  await checkA11y(page);
});
