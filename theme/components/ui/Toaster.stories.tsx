import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { Toaster, useToaster } from "./Toaster";
import { Button } from "./Button";

const meta: Meta<typeof Toaster> = { component: Toaster };
export default meta;
type Story = StoryObj<typeof Toaster>;

const Template: React.FC = () => {
  const { push } = useToaster();
  return (
    <Button onClick={() => push("Saved")}>Notify</Button>
  );
};

export const Default: Story = {
  render: () => (
    <Toaster>
      <Template />
    </Toaster>
  ),
};
export const Loading: Story = {
  render: () => (
    <Toaster>
      <Button disabled>Notify</Button>
    </Toaster>
  ),
};
export const Empty: Story = { render: () => <Toaster /> };
export const Error: Story = {
  render: () => (
    <Toaster>
      <Button onClick={() => { throw new Error("fail"); }}>Notify</Button>
    </Toaster>
  ),
};
export const Dark: Story = {
  render: () => (
    <div className="dark p-4 bg-bg text-fg">
      <Toaster>
        <Template />
      </Toaster>
    </div>
  ),
};
export const RTL: Story = {
  render: () => (
    <div dir="rtl">
      <Toaster>
        <Template />
      </Toaster>
    </div>
  ),
};
