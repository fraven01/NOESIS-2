import type { Meta, StoryObj } from "@storybook/react";
import { Tooltip } from "./Tooltip";
import { Button } from "./Button";

const meta: Meta<typeof Tooltip> = { component: Tooltip };
export default meta;
type Story = StoryObj<typeof Tooltip>;

export const Default: Story = {
  render: () => (
    <Tooltip content="Info">
      <Button>Hover</Button>
    </Tooltip>
  ),
};
export const States: Story = { render: Default.render };
export const Dark: Story = {
  render: () => (
    <div className="dark p-4 bg-bg text-fg">
      <Tooltip content="Info">
        <Button>Hover</Button>
      </Tooltip>
    </div>
  ),
};
export const RTL: Story = {
  render: () => (
    <div dir="rtl">
      <Tooltip content="Info">
        <Button>Hover</Button>
      </Tooltip>
    </div>
  ),
};
