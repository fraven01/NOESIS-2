import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe, toHaveNoViolations } from 'jest-axe';
expect.extend(toHaveNoViolations);
import AppShell from './AppShell';

describe('AppShell', () => {
  test('Escape closes mobile sidebar and returns focus', async () => {
    const user = userEvent.setup();
    render(
      <AppShell>
        <div>Content</div>
      </AppShell>
    );
    const toggle = screen.getByLabelText(/open sidebar/i);
    await user.click(toggle);
    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeInTheDocument();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(toggle).toHaveFocus();
  });

  test('has no axe violations', async () => {
    const { container } = render(
      <AppShell>
        <div>Content</div>
      </AppShell>
    );
    const results = await axe(container);
    // @ts-expect-error matchers provided by jest-axe
    expect(results).toHaveNoViolations();
  });
});
