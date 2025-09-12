import type { Page } from '@playwright/test';
import { injectAxe, checkA11y, Options } from '@axe-core/playwright';

/**
 * Runs axe-core accessibility checks on the current page.
 */
export async function runAxe(page: Page, options?: Options) {
  await injectAxe(page);
  await checkA11y(page, undefined, options);
}
