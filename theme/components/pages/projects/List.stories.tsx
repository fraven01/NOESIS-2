import type { Meta, StoryObj } from "@storybook/react";
import { List } from "./List";

const meta: Meta<typeof List> = {
  title: "Pages/Projects/List",
  component: List,
};
export default meta;
type Story = StoryObj<typeof List>;

const sample = [
  { id: "1", name: "Alpha" },
  { id: "2", name: "Beta" },
  { id: "3", name: "Gamma" },
];

export const Default: Story = {
  args: {
    projects: sample,
  },
};

export const ManyRows: Story = {
  args: {
    projects: Array.from({ length: 15 }, (_, i) => ({ id: String(i), name: `Project ${i}` })),
  },
};

export const Empty: Story = {
  args: {
    projects: [],
  },
};

export const Error: Story = {
  args: {
    projects: sample,
    status: "error",
  },
};

export const Loading: Story = {
  args: {
    projects: sample,
    status: "loading",
  },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <List {...args} />
    </div>
  ),
  args: {
    projects: sample,
  },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <List {...args} />
    </div>
  ),
  args: {
    projects: sample,
  },
};
