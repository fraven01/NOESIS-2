import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe, toHaveNoViolations } from 'jest-axe';
expect.extend(toHaveNoViolations);
import Button from './Button';

describe('Button', () => {
  test('triggers onClick', async () => {
    const user = userEvent.setup();
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click</Button>);
    await user.click(screen.getByRole('button', { name: /click/i }));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  test('shows loading state', () => {
    render(<Button isLoading>Load</Button>);
    const btn = screen.getByRole('button', { name: /load/i });
    expect(btn).toHaveAttribute('aria-busy', 'true');
    expect(btn).toBeDisabled();
  });

  test('has no axe violations', async () => {
    const { container } = render(<Button>Click</Button>);
    const results = await axe(container);
    // @ts-expect-error matchers provided by jest-axe
    expect(results).toHaveNoViolations();
  });
});
