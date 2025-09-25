import type { Meta, StoryObj } from "@storybook/react";
import { Button } from "./Button";

const meta: Meta<typeof Button> = {
  component: Button,
  args: { children: "Button" },
};
export default meta;
type Story = StoryObj<typeof Button>;

export const Default: Story = {};

export const States: Story = {
  render: () => (
    <div className="flex gap-2">
      <Button>Default</Button>
      <Button disabled>Disabled</Button>
      <Button loading>Loading</Button>
    </div>
  ),
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark p-4 bg-bg text-fg">
      <Button {...args}>Dark</Button>
    </div>
  ),
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <Button {...args}>RTL</Button>
    </div>
  ),
};
