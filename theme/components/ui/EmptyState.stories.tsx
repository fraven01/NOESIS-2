import type { Meta, StoryObj } from "@storybook/react";
import { EmptyState } from "./EmptyState";

const meta: Meta<typeof EmptyState> = {
  component: EmptyState,
};
export default meta;
type Story = StoryObj<typeof EmptyState>;

export const Default: Story = {
  args: {
    title: "No items",
    text: "Add one",
    cta: { label: "Add", onClick: () => {} },
  },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <EmptyState {...args} />
    </div>
  ),
  args: {
    title: "No items",
    text: "Add one",
  },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <EmptyState {...args} />
    </div>
  ),
  args: {
    title: "No items",
    text: "Add one",
  },
};
