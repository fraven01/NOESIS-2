import type { Meta, StoryObj } from "@storybook/react";
import { Skeleton } from "./Skeleton";

const meta: Meta<typeof Skeleton> = {
  component: Skeleton,
};
export default meta;
type Story = StoryObj<typeof Skeleton>;

export const Default: Story = {
  args: { className: "h-4 w-32" },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <Skeleton {...args} />
    </div>
  ),
  args: { className: "h-4 w-32" },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <Skeleton {...args} />
    </div>
  ),
  args: { className: "h-4 w-32" },
};
