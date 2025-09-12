import type { Meta, StoryObj } from "@storybook/react";
import { ThemeSwitch } from "./ThemeSwitch";

const meta: Meta<typeof ThemeSwitch> = {
  component: ThemeSwitch,
  title: "Theme/ThemeSwitch",
};
export default meta;
type Story = StoryObj<typeof ThemeSwitch>;

export const Default: Story = {
  render: () => <ThemeSwitch />,
};

export const Disabled: Story = {
  render: () => <ThemeSwitch disabled />, 
};

export const Dark: Story = {
  parameters: { backgrounds: { default: "dark" } },
  render: () => (
    <div className="dark p-4 bg-bg text-fg">
      <ThemeSwitch />
    </div>
  ),
};

export const RTL: Story = {
  render: () => (
    <div dir="rtl">
      <ThemeSwitch />
    </div>
  ),
};
