import type { Meta, StoryObj } from '@storybook/react';
import { List, Project } from './List';

const meta: Meta<typeof List> = {
  title: 'Pages/Projects/List',
  component: List,
};

export default meta;

const sample = (count: number): Project[] =>
  Array.from({ length: count }, (_, i) => ({
    id: String(i + 1),
    name: `Project ${i + 1}`,
    owner: `Owner ${i + 1}`,
  }));

type Story = StoryObj<typeof List>;

export const Default: Story = {
  args: { projects: sample(3) },
};

export const ManyRows: Story = {
  args: { projects: sample(50) },
};

export const Empty: Story = {
  args: { projects: [] },
};

export const Error: Story = {
  args: { projects: sample(3), error: 'Failed to load' },
};

export const Loading: Story = {
  args: { projects: sample(3), loading: true },
};
