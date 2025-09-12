import * as React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe, toHaveNoViolations } from 'jest-axe';
import { List, Project } from './List';

expect.extend(toHaveNoViolations);

const sample = (): Project[] => [
  { id: '1', name: 'Alpha', owner: 'Alice' },
  { id: '2', name: 'Beta', owner: 'Bob' },
];

describe('List', () => {
  test('sorts by column', async () => {
    render(<List projects={sample()} />);
    const nameHeader = screen.getByRole('button', { name: 'Name' });
    await userEvent.click(nameHeader);
    const rows = screen.getAllByRole('row');
    expect(rows[1]).toHaveTextContent('Alpha');
    await userEvent.click(nameHeader);
    const rowsDesc = screen.getAllByRole('row');
    expect(rowsDesc[1]).toHaveTextContent('Beta');
  });

  test('selection toggles indeterminate state', async () => {
    render(<List projects={sample()} />);
    const selectAll = screen.getByLabelText('Select all');
    await userEvent.click(selectAll);
    expect(selectAll).toBeChecked();
    const rowCheckboxes = screen.getAllByLabelText(/Select/);
    await userEvent.click(rowCheckboxes[1]);
    expect(selectAll).toHaveAttribute('aria-checked', 'mixed');
  });

  test('keyboard tab order', async () => {
    render(<List projects={sample()} />);
    const search = screen.getByLabelText('Search');
    const filter = screen.getByLabelText('Filter');
    const selectAll = screen.getByLabelText('Select all');
    await userEvent.tab();
    expect(search).toHaveFocus();
    await userEvent.tab();
    expect(filter).toHaveFocus();
    await userEvent.tab();
    expect(selectAll).toHaveFocus();
  });

  test('has no accessibility violations', async () => {
    const { container } = render(<List projects={sample()} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
