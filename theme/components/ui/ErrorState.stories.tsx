import type { Meta, StoryObj } from "@storybook/react";
import { ErrorState } from "./ErrorState";

const meta: Meta<typeof ErrorState> = {
  component: ErrorState,
};
export default meta;
type Story = StoryObj<typeof ErrorState>;

export const Default: Story = {
  args: {
    title: "Error",
    text: "Something went wrong",
    cta: { label: "Retry", onClick: () => {} },
  },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <ErrorState {...args} />
    </div>
  ),
  args: {
    title: "Error",
    text: "Something went wrong",
  },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <ErrorState {...args} />
    </div>
  ),
  args: {
    title: "Error",
    text: "Something went wrong",
  },
};
