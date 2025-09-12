import type { Meta, StoryObj } from '@storybook/react';
import { AppShell } from './AppShell';

const meta: Meta<typeof AppShell> = {
  component: AppShell,
  title: 'Layouts/AppShell',
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof AppShell>;

export const Default: Story = {
  render: () => (
    <AppShell breadcrumbs={<span>Home</span>} actions={<button className="rounded bg-accent px-2 py-1">Aktion</button>} buildInfo="v1.0">
      <p>Inhalt</p>
    </AppShell>
  ),
};

export const Mobile: Story = {
  parameters: { viewport: { defaultViewport: 'iphone5' } },
  render: Default.render,
};

export const Dark: Story = {
  parameters: { backgrounds: { default: 'dark' } },
  render: () => (
    <div className="dark">
      <AppShell breadcrumbs={<span>Home</span>} actions={<button className="rounded bg-accent px-2 py-1">Aktion</button>} buildInfo="v1.0">
        <p className="text-fg">Inhalt</p>
      </AppShell>
    </div>
  ),
};
