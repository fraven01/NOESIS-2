import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { Toast } from "./Toast";
import { Button } from "./Button";

const meta: Meta<typeof Toast> = { component: Toast };
export default meta;
type Story = StoryObj<typeof Toast>;

const Template: React.FC = () => {
  const [open, setOpen] = React.useState(false);
  return (
    <div>
      <Button onClick={() => setOpen(true)}>Show</Button>
      <Toast open={open} onClose={() => setOpen(false)}>
        Saved
      </Toast>
    </div>
  );
};

export const Default: Story = { render: () => <Template /> };
export const States: Story = { render: () => <Toast open={true} onClose={() => {}}>Saved</Toast> };
export const Dark: Story = {
  render: () => (
    <div className="dark p-4 bg-bg text-fg">
      <Template />
    </div>
  ),
};
export const RTL: Story = {
  render: () => (
    <div dir="rtl">
      <Template />
    </div>
  ),
};
