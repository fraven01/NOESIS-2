import type { Meta, StoryObj } from '@storybook/react';
import AppShell from './AppShell';

const meta: Meta<typeof AppShell> = {
  title: 'Layouts/AppShell',
  component: AppShell,
  parameters: { layout: 'fullscreen' },
};
export default meta;
type Story = StoryObj<typeof AppShell>;

export const Default: Story = {
  render: () => (
    <AppShell>
      <div className="p-4">Inhalt</div>
    </AppShell>
  ),
};

export const Mobile: Story = {
  render: Default.render,
  parameters: { viewport: { defaultViewport: 'mobile1' } },
};

export const Dark: Story = {
  render: () => (
    <div className="dark">
      <AppShell>
        <div className="p-4 text-fg">Inhalt</div>
      </AppShell>
    </div>
  ),
};
