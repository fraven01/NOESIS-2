import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { ConfirmDialog } from "./ConfirmDialog";
import { Button } from "./Button";

const meta: Meta<typeof ConfirmDialog> = { component: ConfirmDialog };
export default meta;
type Story = StoryObj<typeof ConfirmDialog>;

const Template: React.FC = () => {
  const [open, setOpen] = React.useState(false);
  return (
    <div>
      <Button onClick={() => setOpen(true)}>Open</Button>
      <ConfirmDialog
        open={open}
        onClose={() => setOpen(false)}
        onConfirm={() => setOpen(false)}
        title="Delete item"
        description="This action cannot be undone"
        confirmText="DELETE"
      />
    </div>
  );
};

export const Default: Story = { render: () => <Template /> };
export const Loading: Story = {
  render: () => (
    <ConfirmDialog
      open
      onClose={() => {}}
      onConfirm={() => {}}
      title="Please wait"
      confirmLabel="Deleting..."
      className="opacity-50"
    />
  ),
};
export const Empty: Story = {
  render: () => (
    <ConfirmDialog open onClose={() => {}} onConfirm={() => {}} title="Delete" />
  ),
};
export const Error: Story = {
  render: () => (
    <ConfirmDialog
      open
      onClose={() => {}}
      onConfirm={() => {}}
      title="Delete"
      description="Error deleting item"
    />
  ),
};
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
