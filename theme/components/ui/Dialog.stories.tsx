import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { Dialog } from "./Dialog";
import { Button } from "./Button";

const meta: Meta<typeof Dialog> = { component: Dialog };
export default meta;
type Story = StoryObj<typeof Dialog>;

const Template: React.FC = () => {
  const [open, setOpen] = React.useState(false);
  return (
    <div>
      <Button onClick={() => setOpen(true)}>Open</Button>
      <Dialog open={open} onClose={() => setOpen(false)}>
        <p>Dialog content</p>
        <Button onClick={() => setOpen(false)}>Close</Button>
      </Dialog>
    </div>
  );
};

export const Default: Story = { render: () => <Template /> };
export const States: Story = { render: () => <Dialog open={true} onClose={() => {}}><p>Dialog</p></Dialog> };
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
