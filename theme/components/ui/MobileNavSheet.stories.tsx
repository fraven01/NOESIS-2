import * as React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { MobileNavSheet } from "./MobileNavSheet";
import { Button } from "./Button";

const meta: Meta<typeof MobileNavSheet> = { component: MobileNavSheet };
export default meta;
type Story = StoryObj<typeof MobileNavSheet>;

const links = [
  { label: "Home", href: "#home" },
  { label: "About", href: "#about" },
];

const Template: React.FC = () => {
  const [open, setOpen] = React.useState(false);
  return (
    <div>
      <Button onClick={() => setOpen(true)}>Menu</Button>
      <MobileNavSheet open={open} onClose={() => setOpen(false)} links={links} />
    </div>
  );
};

export const Default: Story = { render: () => <Template /> };
export const Loading: Story = {
  render: () => (
    <MobileNavSheet open onClose={() => {}} links={[]} className="opacity-50" />
  ),
};
export const Empty: Story = {
  render: () => (
    <MobileNavSheet open onClose={() => {}} links={[]} />
  ),
};
export const Error: Story = {
  render: () => (
    <MobileNavSheet open onClose={() => {}} links={[]} className="border border-danger" />
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
