import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'UI/Button',
  component: Button,
};
export default meta;

type Story = StoryObj<typeof Button>;

export const Default: Story = {
  render: () => <Button>Click me</Button>,
};

export const States: Story = {
  render: () => (
    <div className="space-x-2">
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="destructive">Destructive</Button>
      <Button isLoading>Loading</Button>
      <Button disabled>Disabled</Button>
    </div>
  ),
};

export const Dark: Story = {
  render: () => (
    <div className="dark p-4 space-x-2 bg-bg">
      <Button variant="primary">Primary</Button>
    </div>
  ),
};

export const RTL: Story = {
  render: Default.render,
  parameters: { direction: 'rtl' },
};
