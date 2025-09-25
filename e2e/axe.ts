import { AxeBuilder } from '@axe-core/playwright';
import type { Page } from '@playwright/test';

export async function checkA11y(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  return results;
}
