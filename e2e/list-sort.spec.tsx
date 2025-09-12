import React from 'react';
import { test, expect } from '@playwright/experimental-ct-react';
import { DataTable, Row } from '../theme/components/data/DataTable';

test('sorts list ascending and descending', async ({ mount, page }) => {
  const rows: Row[] = [
    { id: '1', name: 'Charlie' },
    { id: '2', name: 'Alice' },
    { id: '3', name: 'Bob' },
  ];
  const TableWrapper = () => {
    const [selected, setSelected] = React.useState<string[]>([]);
    return <DataTable rows={rows} selected={selected} onSelectedChange={setSelected} />;
  };
  await mount(<TableWrapper />);
  const header = page.getByRole('columnheader', { name: 'Name' });
  await expect(page.locator('tbody tr:first-child td:last-child')).toHaveText('Charlie');
  await header.click();
  await expect(page.locator('tbody tr:first-child td:last-child')).toHaveText('Alice');
  await header.click();
  await expect(page.locator('tbody tr:first-child td:last-child')).toHaveText('Charlie');
});
