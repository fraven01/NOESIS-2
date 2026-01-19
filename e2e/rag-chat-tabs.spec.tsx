import React from 'react';
import { test, expect } from '@playwright/experimental-ct-react';

const RagChatDebugTabs = () => {
  const [tab, setTab] = React.useState('answer');

  return (
    <div>
      <div className="flex gap-2">
        <button type="button" onClick={() => setTab('answer')}>
          Final Answer
        </button>
        <button type="button" onClick={() => setTab('reasoning')}>
          Thinking Process
        </button>
        <button type="button" onClick={() => setTab('sources')}>
          Sources & Evidence
        </button>
      </div>
      <div hidden={tab !== 'answer'}>
        <p>Answer content</p>
      </div>
      <div hidden={tab !== 'reasoning'}>
        <p>Reasoning content</p>
      </div>
      <div hidden={tab !== 'sources'}>
        <p>Sources content</p>
      </div>
    </div>
  );
};

test('RAG chat tabs toggle visibility', async ({ mount, page }) => {
  await mount(<RagChatDebugTabs />);

  await expect(page.getByText('Answer content')).toBeVisible();
  await expect(page.getByText('Reasoning content')).toBeHidden();
  await expect(page.getByText('Sources content')).toBeHidden();

  await page.getByRole('button', { name: 'Thinking Process' }).click();
  await expect(page.getByText('Reasoning content')).toBeVisible();
  await expect(page.getByText('Answer content')).toBeHidden();

  await page.getByRole('button', { name: 'Sources & Evidence' }).click();
  await expect(page.getByText('Sources content')).toBeVisible();
  await expect(page.getByText('Reasoning content')).toBeHidden();
});
