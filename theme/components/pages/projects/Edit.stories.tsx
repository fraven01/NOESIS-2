import type { Meta, StoryObj } from "@storybook/react";
import { Edit } from "./Edit";

const meta: Meta<typeof Edit> = {
  title: "Pages/Projects/Edit",
  component: Edit,
};
export default meta;
type Story = StoryObj<typeof Edit>;

export const Default: Story = {
  args: {
    initial: { name: "Alpha", description: "First project" },
  },
};

export const Invalid: Story = {
  args: {
    initial: { name: "", description: "" },
    initialErrors: { name: "Name is required" },
  },
};

export const Loading: Story = {
  render: (args) => <Edit {...args} onSubmit={() => new Promise((r) => setTimeout(r, 1000))} />,
  args: {
    initial: { name: "Alpha" },
  },
};

export const Dark: Story = {
  render: (args) => (
    <div className="dark bg-bg p-4 text-fg">
      <Edit {...args} />
    </div>
  ),
  args: {
    initial: { name: "Alpha" },
  },
};

export const RTL: Story = {
  render: (args) => (
    <div dir="rtl">
      <Edit {...args} />
    </div>
  ),
  args: {
    initial: { name: "Alpha" },
  },
};
