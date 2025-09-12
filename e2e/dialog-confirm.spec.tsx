import React from 'react';
import { test, expect } from '@playwright/experimental-ct-react';
import { ConfirmDialog } from '../theme/components/ui/ConfirmDialog';

test('dialog confirm flow', async ({ mount, page }) => {
  const Wrapper = () => {
    const [open, setOpen] = React.useState(true);
    const [confirmed, setConfirmed] = React.useState(false);
    return (
      <>
        <ConfirmDialog
          open={open}
          onClose={() => setOpen(false)}
          onConfirm={() => setConfirmed(true)}
          title="Delete"
          confirmText="DELETE"
        />
        {confirmed && <p data-testid="result">confirmed</p>}
      </>
    );
  };
  await mount(<Wrapper />);
  await page.getByLabel('Type to confirm').fill('DELETE');
  await page.getByRole('button', { name: 'Delete' }).click();
  await expect(page.getByTestId('result')).toHaveText('confirmed');
});
