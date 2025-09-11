import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import { AppShell } from './AppShell';

test('keyboard toggles mobile sidebar and returns focus', async () => {
  const user = userEvent.setup();
  render(
    <AppShell breadcrumbs={<span>Home</span>} buildInfo="1.0">
      <p>Content</p>
    </AppShell>
  );
  const toggle = screen.getByLabelText(/toggle navigation/i);
  toggle.focus();
  await user.keyboard('{Enter}');
  const nav = screen.getByTestId('mobile-nav');
  expect(nav).toBeVisible();
  await user.keyboard('{Escape}');
  expect(toggle).toHaveFocus();
});

test('has no accessibility violations', async () => {
  const { container } = render(
    <AppShell breadcrumbs={<span>Home</span>} buildInfo="1.0">
      <p>Content</p>
    </AppShell>
  );
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
